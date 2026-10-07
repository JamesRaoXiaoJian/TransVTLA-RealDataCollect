#!/usr/bin/env python3
"""Lightweight validation for the tactile-patched TransVTLA RLDS shards.

This avoids full tf.train.Example deserialization, which expands every depth
and image payload. It validates TFRecord framing plus the fields that should be
affected by the tactile patch.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from patch_rlds_tactile import (
    TACTILE_FEATURE,
    feature_int64_values,
    find_feature_message,
    find_tactile_float_bytes,
    iter_fields,
    iter_tfrecord_records,
    read_varint,
)


DATASET_NAME = "transvtla_annotated_rlds"
SHARD_GLOB = f"{DATASET_NAME}-train.tfrecord-*-of-*"
STATE_FEATURE = "steps/observation/state"
ACTION_FEATURE = "steps/action"
DEPTH_FEATURES = [
    "steps/observation/primary_depth",
    "steps/observation/wrist_depth",
]
IMAGE_FEATURES = [
    "steps/observation/primary_image",
    "steps/observation/wrist_image",
]
STEP_INT_FIELDS = [
    "steps/is_first",
    "steps/is_last",
    "steps/is_terminal",
    "steps/stage_label",
    "steps/use_tactile",
]
STEP_TEXT_FIELDS = [
    "steps/action_prompt",
    "steps/item_type",
    "steps/language_instruction",
    "steps/material",
    "steps/object_label",
    "steps/object_type",
    "steps/phase",
]
METADATA_FIELDS = [
    "episode_metadata/annotation_key",
    "episode_metadata/data_source",
    "episode_metadata/episode_id",
    "episode_metadata/needs_review",
    "episode_metadata/review_reasons",
    "episode_metadata/slot_pair",
    "episode_metadata/source_file",
    "episode_metadata/split",
    "episode_metadata/success",
    "episode_metadata/task_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--source-version-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def load_expected_shard_lengths(version_dir: Path) -> list[int] | None:
    info_path = version_dir / "dataset_info.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for split in info.get("splits", []):
        if split.get("name") == "train":
            return [int(x) for x in split.get("shardLengths", [])]
    return None


def is_image_payload_plausible(prefix: bytes) -> bool:
    return (
        prefix.startswith(b"\xff\xd8\xff")
        or prefix.startswith(b"\x89PNG\r\n\x1a\n")
        or prefix.startswith(b"GIF87a")
        or prefix.startswith(b"GIF89a")
        or prefix.startswith(b"BM")
    )


def find_packed_float_bytes(record: bytes | bytearray, feature_key: str) -> tuple[int, int]:
    feature_start, feature_end = find_feature_message(record, feature_key)
    record_view = memoryview(record)
    for field_no, wire_type, float_list_start, float_list_end in iter_fields(
        record_view, feature_start, feature_end
    ):
        if field_no != 2 or wire_type != 2:
            continue
        packed_ranges: list[tuple[int, int]] = []
        for list_field, list_wire, value_start, value_end in iter_fields(
            record_view, float_list_start, float_list_end
        ):
            if list_field == 1 and list_wire == 2:
                packed_ranges.append((value_start, value_end))
        if len(packed_ranges) != 1:
            raise ValueError(f"expected one packed float range for {feature_key}, found {len(packed_ranges)}")
        return packed_ranges[0]
    raise KeyError(f"{feature_key} float_list")


def float_array(record: bytes, feature_key: str) -> np.ndarray:
    start, end = find_packed_float_bytes(record, feature_key)
    byte_len = end - start
    if byte_len % 4 != 0:
        raise ValueError(f"{feature_key} byte length {byte_len} is not divisible by 4")
    return np.frombuffer(record, dtype="<f4", count=byte_len // 4, offset=start)


def int64_values(record: bytes | bytearray, feature_key: str) -> list[int]:
    return [int(x) for x in feature_int64_values(record, feature_key)]


def bytes_list_stats(record: bytes | bytearray, feature_key: str, *, image_magic: bool = False) -> dict[str, int]:
    feature_start, feature_end = find_feature_message(record, feature_key)
    record_view = memoryview(record)
    stats = {"count": 0, "empty": 0, "bad_magic": 0, "invalid_utf8": 0}
    found = False
    for field_no, wire_type, bytes_list_start, bytes_list_end in iter_fields(
        record_view, feature_start, feature_end
    ):
        if field_no != 1 or wire_type != 2:
            continue
        found = True
        for list_field, list_wire, value_start, value_end in iter_fields(
            record_view, bytes_list_start, bytes_list_end
        ):
            if list_field != 1 or list_wire != 2:
                continue
            stats["count"] += 1
            if value_start == value_end:
                stats["empty"] += 1
                continue
            if image_magic:
                prefix = bytes(record_view[value_start : min(value_end, value_start + 8)])
                if not is_image_payload_plausible(prefix):
                    stats["bad_magic"] += 1
            else:
                try:
                    bytes(record_view[value_start:value_end]).decode("utf-8")
                except UnicodeDecodeError:
                    stats["invalid_utf8"] += 1
    if not found:
        raise KeyError(f"{feature_key} bytes_list")
    return stats


def ensure_feature(record: bytes | bytearray, feature_key: str) -> None:
    find_feature_message(record, feature_key)


def update_minmax(minmax: dict[str, float], prefix: str, values: np.ndarray) -> None:
    if values.size == 0:
        return
    minmax[f"{prefix}_min"] = min(minmax.get(f"{prefix}_min", float("inf")), float(values.min()))
    minmax[f"{prefix}_max"] = max(minmax.get(f"{prefix}_max", float("-inf")), float(values.max()))


def compare_source_sizes(source_version_dir: Path | None, shards: list[Path]) -> list[str]:
    if source_version_dir is None:
        return []
    mismatches: list[str] = []
    for shard in shards:
        source = source_version_dir / shard.name
        if not source.is_file():
            mismatches.append(f"missing source shard for {shard.name}")
            continue
        if source.stat().st_size != shard.stat().st_size:
            mismatches.append(
                f"{shard.name}: size output={shard.stat().st_size} source={source.stat().st_size}"
            )
    return mismatches


def main() -> int:
    args = parse_args()
    version_dir = args.version_dir
    if not version_dir.is_dir():
        raise SystemExit(f"Version directory not found: {version_dir}")

    shards = sorted(version_dir.glob(SHARD_GLOB))
    expected_lengths = load_expected_shard_lengths(version_dir)
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[str] = Counter()
    minmax: dict[str, float] = {}
    shard_example_counts: list[int] = []
    episode_ids: Counter[int] = Counter()

    if not shards:
        errors.append("no TFRecord shards found")
    if expected_lengths is not None and len(shards) != len(expected_lengths):
        errors.append(f"shard count mismatch: files={len(shards)} dataset_info={len(expected_lengths)}")

    for mismatch in compare_source_sizes(args.source_version_dir, shards):
        errors.append(f"source/output size mismatch: {mismatch}")

    for shard_idx, shard in enumerate(shards):
        shard_examples = 0
        try:
            for record in iter_tfrecord_records(shard):
                shard_examples += 1
                counts["examples"] += 1

                missing = [key for key in METADATA_FIELDS if not _has_feature(record, key)]
                if missing:
                    errors.append(f"{shard.name} example {shard_examples}: missing metadata {missing}")
                    continue

                episode_id = int64_values(record, "episode_metadata/episode_id")
                if len(episode_id) != 1:
                    errors.append(f"{shard.name} example {shard_examples}: episode_id length={len(episode_id)}")
                else:
                    episode_ids[episode_id[0]] += 1

                tactile_start, tactile_end = find_tactile_float_bytes(record)
                tactile_byte_len = tactile_end - tactile_start
                if tactile_byte_len % (16 * 20 * 4) != 0:
                    errors.append(
                        f"{shard.name} example {shard_examples}: tactile bytes={tactile_byte_len} "
                        "not divisible by one frame"
                    )
                    continue
                tactile = np.frombuffer(record, dtype="<f4", count=tactile_byte_len // 4, offset=tactile_start)
                n_steps = int(tactile.size // (16 * 20))
                if n_steps <= 0:
                    errors.append(f"{shard.name} example {shard_examples}: zero tactile steps")
                    continue
                counts["steps"] += n_steps
                counts["tactile_values"] += int(tactile.size)
                counts["tactile_nonzero"] += int(np.count_nonzero(tactile))
                if not bool(np.all(np.isfinite(tactile))):
                    errors.append(f"{shard.name} example {shard_examples}: tactile contains NaN or Inf")
                update_minmax(minmax, "tactile", tactile)
                if tactile.size and (float(tactile.min()) < -1e-6 or float(tactile.max()) > 1.0 + 1e-6):
                    errors.append(
                        f"{shard.name} example {shard_examples}: tactile range "
                        f"{float(tactile.min())}..{float(tactile.max())}"
                    )

                state = float_array(record, STATE_FEATURE)
                action = float_array(record, ACTION_FEATURE)
                if state.size != n_steps * 7:
                    errors.append(
                        f"{shard.name} example {shard_examples}: state values={state.size}, expected={n_steps * 7}"
                    )
                if action.size != n_steps * 7:
                    errors.append(
                        f"{shard.name} example {shard_examples}: action values={action.size}, expected={n_steps * 7}"
                    )
                if state.size and not bool(np.all(np.isfinite(state))):
                    errors.append(f"{shard.name} example {shard_examples}: state contains NaN or Inf")
                if action.size and not bool(np.all(np.isfinite(action))):
                    errors.append(f"{shard.name} example {shard_examples}: action contains NaN or Inf")
                update_minmax(minmax, "state", state)
                update_minmax(minmax, "action", action)

                for key in STEP_INT_FIELDS:
                    values = int64_values(record, key)
                    if len(values) != n_steps:
                        errors.append(f"{shard.name} example {shard_examples}: {key} length={len(values)}, expected={n_steps}")
                    if key in {"steps/is_first", "steps/is_last", "steps/is_terminal", "steps/use_tactile"}:
                        bad_bool = sum(1 for value in values if value not in (0, 1))
                        if bad_bool:
                            errors.append(f"{shard.name} example {shard_examples}: {key} has {bad_bool} non-bool values")
                    if key == "steps/use_tactile":
                        counts["use_tactile_true"] += sum(1 for value in values if value == 1)

                for key in STEP_TEXT_FIELDS:
                    stats = bytes_list_stats(record, key)
                    if stats["count"] != n_steps:
                        errors.append(f"{shard.name} example {shard_examples}: {key} count={stats['count']}, expected={n_steps}")
                    counts[f"{key}_empty"] += stats["empty"]
                    counts[f"{key}_invalid_utf8"] += stats["invalid_utf8"]

                for key in IMAGE_FEATURES:
                    stats = bytes_list_stats(record, key, image_magic=True)
                    if stats["count"] != n_steps:
                        errors.append(f"{shard.name} example {shard_examples}: {key} count={stats['count']}, expected={n_steps}")
                    counts[f"{key}_payloads"] += stats["count"]
                    counts[f"{key}_empty"] += stats["empty"]
                    counts[f"{key}_bad_magic"] += stats["bad_magic"]

                for key in DEPTH_FEATURES:
                    ensure_feature(record, key)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{shard.name}: TFRecord/feature read failed: {exc}")

        shard_example_counts.append(shard_examples)
        if expected_lengths is not None and shard_idx < len(expected_lengths):
            expected = expected_lengths[shard_idx]
            if shard_examples != expected:
                errors.append(f"{shard.name}: shard examples={shard_examples}, expected={expected}")
        if (
            (shard_idx + 1) == 1
            or (args.progress_every > 0 and (shard_idx + 1) % args.progress_every == 0)
            or (shard_idx + 1) == len(shards)
        ):
            print(
                f"[{shard_idx + 1:04d}/{len(shards):04d}] "
                f"examples={counts['examples']} steps={counts['steps']} "
                f"tactile_nonzero={counts['tactile_nonzero']} errors={len(errors)}",
                flush=True,
            )

    duplicate_episode_ids = {k: v for k, v in episode_ids.items() if v > 1}
    if duplicate_episode_ids:
        warnings.append(f"duplicate episode_id values: {duplicate_episode_ids}")
    if expected_lengths is not None and counts["examples"] != sum(expected_lengths):
        errors.append(f"total examples={counts['examples']}, expected={sum(expected_lengths)}")
    if counts["tactile_nonzero"] == 0:
        errors.append("tactile values are all zero")

    summary: dict[str, Any] = {
        "version_dir": str(version_dir),
        "source_version_dir": str(args.source_version_dir) if args.source_version_dir else None,
        "num_shards": len(shards),
        "expected_shards_from_dataset_info": len(expected_lengths) if expected_lengths is not None else None,
        "num_examples": int(counts["examples"]),
        "num_steps": int(counts["steps"]),
        "tactile_values": int(counts["tactile_values"]),
        "tactile_nonzero": int(counts["tactile_nonzero"]),
        "use_tactile_true": int(counts["use_tactile_true"]),
        "image_payload_counts": {key: int(counts[f"{key}_payloads"]) for key in IMAGE_FEATURES},
        "image_empty_counts": {key: int(counts[f"{key}_empty"]) for key in IMAGE_FEATURES},
        "image_bad_magic_counts": {key: int(counts[f"{key}_bad_magic"]) for key in IMAGE_FEATURES},
        "text_empty_counts": {
            key: int(counts[f"{key}_empty"])
            for key in STEP_TEXT_FIELDS
            if counts[f"{key}_empty"]
        },
        "text_invalid_utf8_counts": {
            key: int(counts[f"{key}_invalid_utf8"])
            for key in STEP_TEXT_FIELDS
            if counts[f"{key}_invalid_utf8"]
        },
        "minmax": {k: (None if math.isinf(v) else v) for k, v in sorted(minmax.items())},
        "duplicate_episode_ids": duplicate_episode_ids,
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors[:200],
        "warnings": warnings[:200],
        "shard_example_counts": shard_example_counts,
        "notes": [
            "Depth payload values are not expanded; depth feature presence is checked.",
            "Image payloads are checked for count and magic bytes but not decoded.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"shard_example_counts", "errors", "warnings"}}, indent=2, sort_keys=True))
    if errors:
        print(f"TACTILE_PATCH_VALIDATION_FAILED errors={len(errors)} warnings={len(warnings)}")
        return 1
    print(f"TACTILE_PATCH_VALIDATION_OK warnings={len(warnings)}")
    return 0


def _has_feature(record: bytes | bytearray, feature_key: str) -> bool:
    try:
        find_feature_message(record, feature_key)
    except KeyError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
