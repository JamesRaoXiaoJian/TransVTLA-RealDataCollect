#!/usr/bin/env python3
"""Validate a TransVTLA TFDS/RLDS directory.

The validator scans every TFRecord example and checks:
  - shard naming/counts against dataset_info.json
  - TFRecord CRC/readability and Example deserialization
  - required fields, sequence lengths, scalar metadata lengths
  - finite float values for action/state/scalar/tactile fields
  - gripper normalization for state[6] and action[6]
  - action[6] alignment to next state[6]
  - non-empty image payloads with PNG/JPEG/GIF/BMP magic bytes

Image decoding can be enabled for sampled or all images, but the default is a
full structural scan without decoding every image payload.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


DEFAULT_VERSION_DIR = Path(
    "/media/files/TransVTLA-RealDataCollect/dataset/"
    "rlds_real1000_gripper01_strict/transvtla_annotated_rlds/3.0.0"
)

DATASET_NAME = "transvtla_annotated_rlds"
SHARD_PREFIX = f"{DATASET_NAME}-train.tfrecord-"

METADATA_FIELDS = {
    "episode_metadata/annotation_key": "bytes_list",
    "episode_metadata/data_source": "bytes_list",
    "episode_metadata/episode_id": "int64_list",
    "episode_metadata/needs_review": "int64_list",
    "episode_metadata/review_reasons": "bytes_list",
    "episode_metadata/slot_pair": "bytes_list",
    "episode_metadata/source_file": "bytes_list",
    "episode_metadata/split": "bytes_list",
    "episode_metadata/success": "int64_list",
    "episode_metadata/task_id": "bytes_list",
}

STEP_SCALAR_FLOAT_FIELDS = [
    "steps/depth_uncertainty",
    "steps/force_max",
    "steps/force_min",
    "steps/fragility",
    "steps/reward",
    "steps/slip_risk",
]

STEP_SCALAR_INT_FIELDS = [
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

SEQUENCE_FLOAT_FIELDS_WITH_MULTIPLIER = {
    "steps/action": 7,
    "steps/observation/state": 7,
    "steps/observation/tactile": 16 * 20,
}

SEQUENCE_INT_FIELDS_WITH_MULTIPLIER = {
    "steps/observation/primary_depth": 224 * 224,
    "steps/observation/wrist_depth": 224 * 224,
}

IMAGE_FIELDS = [
    "steps/observation/primary_image",
    "steps/observation/wrist_image",
]

REQUIRED_KEYS = (
    set(METADATA_FIELDS)
    | set(STEP_SCALAR_FLOAT_FIELDS)
    | set(STEP_SCALAR_INT_FIELDS)
    | set(STEP_TEXT_FIELDS)
    | set(SEQUENCE_FLOAT_FIELDS_WITH_MULTIPLIER)
    | set(SEQUENCE_INT_FIELDS_WITH_MULTIPLIER)
    | set(IMAGE_FIELDS)
)


def feature_kind(feature: tf.train.Feature) -> str | None:
    return feature.WhichOneof("kind")


def list_len(feature: tf.train.Feature) -> int:
    kind = feature_kind(feature)
    if kind == "bytes_list":
        return len(feature.bytes_list.value)
    if kind == "float_list":
        return len(feature.float_list.value)
    if kind == "int64_list":
        return len(feature.int64_list.value)
    return 0


def is_image_payload_plausible(payload: bytes) -> bool:
    return (
        payload.startswith(b"\xff\xd8\xff")
        or payload.startswith(b"\x89PNG\r\n\x1a\n")
        or payload.startswith(b"GIF87a")
        or payload.startswith(b"GIF89a")
        or payload.startswith(b"BM")
    )


def decode_image(payload: bytes) -> tuple[int, int, int]:
    decoded = tf.io.decode_image(payload, channels=3, expand_animations=False)
    shape = decoded.shape.as_list()
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"decoded image shape is not HxWx3: {shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def decode_utf8(value: bytes) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return "<invalid-utf8>"


def finite_check(values: Iterable[float]) -> bool:
    arr = np.asarray(values, dtype=np.float32)
    return bool(np.all(np.isfinite(arr)))


def update_minmax(counter: dict[str, float], prefix: str, values: np.ndarray) -> None:
    counter[f"{prefix}_min"] = min(counter.get(f"{prefix}_min", float("inf")), float(values.min()))
    counter[f"{prefix}_max"] = max(counter.get(f"{prefix}_max", float("-inf")), float(values.max()))


def load_expected_shard_lengths(version_dir: Path) -> list[int] | None:
    info_path = version_dir / "dataset_info.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for split in info.get("splits", []):
        if split.get("name") == "train":
            return [int(x) for x in split.get("shardLengths", [])]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-dir", type=Path, default=DEFAULT_VERSION_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--decode-images",
        choices=["none", "sample", "all"],
        default="sample",
        help="Decode no images, the first image per image field per episode, or all images.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version_dir = args.version_dir
    if not version_dir.is_dir():
        raise SystemExit(f"Version directory not found: {version_dir}")

    shards = sorted(version_dir.glob(f"{SHARD_PREFIX}*-of-*"))
    expected_lengths = load_expected_shard_lengths(version_dir)

    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[str] = Counter()
    minmax: dict[str, float] = {}
    text_empty_counts: Counter[str] = Counter()
    text_invalid_utf8_counts: Counter[str] = Counter()
    episode_ids: Counter[int] = Counter()
    shard_example_counts: list[int] = []
    image_payload_counts: Counter[str] = Counter()
    image_decode_counts: Counter[str] = Counter()

    if expected_lengths is not None and len(shards) != len(expected_lengths):
        errors.append(f"shard count mismatch: files={len(shards)} dataset_info={len(expected_lengths)}")
    if not shards:
        errors.append("no TFRecord shards found")

    for shard_idx, shard in enumerate(shards):
        shard_examples = 0
        try:
            iterator = tf.compat.v1.io.tf_record_iterator(str(shard))
            for raw in iterator:
                counts["examples"] += 1
                shard_examples += 1
                try:
                    example = tf.train.Example.FromString(raw)
                except Exception as exc:  # noqa: BLE001 - record exact validation failure.
                    errors.append(f"{shard.name}: Example parse failed at example {shard_examples}: {exc}")
                    continue

                features = example.features.feature
                keys = set(features.keys())
                missing = sorted(REQUIRED_KEYS - keys)
                extra = sorted(keys - REQUIRED_KEYS)
                if missing:
                    errors.append(f"{shard.name} example {shard_examples}: missing keys {missing}")
                    continue
                if extra:
                    warnings.append(f"{shard.name} example {shard_examples}: unexpected keys {extra}")

                for key, expected_kind in METADATA_FIELDS.items():
                    kind = feature_kind(features[key])
                    length = list_len(features[key])
                    if kind != expected_kind:
                        errors.append(f"{shard.name} example {shard_examples}: {key} kind={kind}, expected={expected_kind}")
                    if length != 1:
                        errors.append(f"{shard.name} example {shard_examples}: {key} length={length}, expected=1")

                episode_id_values = features["episode_metadata/episode_id"].int64_list.value
                if len(episode_id_values) == 1:
                    episode_ids[int(episode_id_values[0])] += 1

                state_values = features["steps/observation/state"].float_list.value
                action_values = features["steps/action"].float_list.value
                if len(state_values) % 7 != 0:
                    errors.append(f"{shard.name} example {shard_examples}: state length {len(state_values)} not divisible by 7")
                    continue
                n_steps = len(state_values) // 7
                if n_steps <= 0:
                    errors.append(f"{shard.name} example {shard_examples}: zero-step episode")
                    continue
                counts["steps"] += n_steps

                for key, multiplier in SEQUENCE_FLOAT_FIELDS_WITH_MULTIPLIER.items():
                    feature = features[key]
                    kind = feature_kind(feature)
                    length = list_len(feature)
                    expected = n_steps * multiplier
                    if kind != "float_list":
                        errors.append(f"{shard.name} example {shard_examples}: {key} kind={kind}, expected=float_list")
                    if length != expected:
                        errors.append(f"{shard.name} example {shard_examples}: {key} length={length}, expected={expected}")
                    if length and not finite_check(feature.float_list.value):
                        errors.append(f"{shard.name} example {shard_examples}: {key} contains NaN or Inf")

                for key, multiplier in SEQUENCE_INT_FIELDS_WITH_MULTIPLIER.items():
                    feature = features[key]
                    kind = feature_kind(feature)
                    length = list_len(feature)
                    expected = n_steps * multiplier
                    if kind != "int64_list":
                        errors.append(f"{shard.name} example {shard_examples}: {key} kind={kind}, expected=int64_list")
                    if length != expected:
                        errors.append(f"{shard.name} example {shard_examples}: {key} length={length}, expected={expected}")

                for key in STEP_SCALAR_FLOAT_FIELDS:
                    feature = features[key]
                    kind = feature_kind(feature)
                    length = list_len(feature)
                    if kind != "float_list":
                        errors.append(f"{shard.name} example {shard_examples}: {key} kind={kind}, expected=float_list")
                    if length != n_steps:
                        errors.append(f"{shard.name} example {shard_examples}: {key} length={length}, expected={n_steps}")
                    if length and not finite_check(feature.float_list.value):
                        errors.append(f"{shard.name} example {shard_examples}: {key} contains NaN or Inf")

                for key in STEP_SCALAR_INT_FIELDS:
                    feature = features[key]
                    kind = feature_kind(feature)
                    length = list_len(feature)
                    if kind != "int64_list":
                        errors.append(f"{shard.name} example {shard_examples}: {key} kind={kind}, expected=int64_list")
                    if length != n_steps:
                        errors.append(f"{shard.name} example {shard_examples}: {key} length={length}, expected={n_steps}")

                for key in STEP_TEXT_FIELDS:
                    feature = features[key]
                    kind = feature_kind(feature)
                    length = list_len(feature)
                    if kind != "bytes_list":
                        errors.append(f"{shard.name} example {shard_examples}: {key} kind={kind}, expected=bytes_list")
                    if length != n_steps:
                        errors.append(f"{shard.name} example {shard_examples}: {key} length={length}, expected={n_steps}")
                    for value in feature.bytes_list.value:
                        if not value:
                            text_empty_counts[key] += 1
                        if decode_utf8(value) == "<invalid-utf8>":
                            text_invalid_utf8_counts[key] += 1

                for key in IMAGE_FIELDS:
                    feature = features[key]
                    kind = feature_kind(feature)
                    length = list_len(feature)
                    if kind != "bytes_list":
                        errors.append(f"{shard.name} example {shard_examples}: {key} kind={kind}, expected=bytes_list")
                    if length != n_steps:
                        errors.append(f"{shard.name} example {shard_examples}: {key} length={length}, expected={n_steps}")
                    payloads = feature.bytes_list.value
                    for payload_idx, payload in enumerate(payloads):
                        image_payload_counts[key] += 1
                        if not payload:
                            errors.append(f"{shard.name} example {shard_examples}: {key}[{payload_idx}] is empty")
                            continue
                        if not is_image_payload_plausible(payload):
                            errors.append(f"{shard.name} example {shard_examples}: {key}[{payload_idx}] has unknown image magic")
                            continue
                        should_decode = args.decode_images == "all" or (
                            args.decode_images == "sample" and payload_idx == 0
                        )
                        if should_decode:
                            try:
                                h, w, c = decode_image(payload)
                                image_decode_counts[key] += 1
                                if (h, w, c) != (224, 224, 3):
                                    errors.append(
                                        f"{shard.name} example {shard_examples}: {key}[{payload_idx}] "
                                        f"decoded shape={(h, w, c)}, expected=(224, 224, 3)"
                                    )
                            except Exception as exc:  # noqa: BLE001
                                errors.append(f"{shard.name} example {shard_examples}: {key}[{payload_idx}] decode failed: {exc}")

                state = np.asarray(state_values, dtype=np.float32).reshape(n_steps, 7)
                action = np.asarray(action_values, dtype=np.float32).reshape(n_steps, 7)
                state6 = state[:, 6]
                action6 = action[:, 6]
                update_minmax(minmax, "state6", state6)
                update_minmax(minmax, "action6", action6)
                update_minmax(minmax, "state_all", state)
                update_minmax(minmax, "action_all", action)

                counts["state6_below_0"] += int(np.count_nonzero(state6 < -1e-6))
                counts["state6_above_1"] += int(np.count_nonzero(state6 > 1.0 + 1e-6))
                counts["action6_below_0"] += int(np.count_nonzero(action6 < -1e-6))
                counts["action6_above_1"] += int(np.count_nonzero(action6 > 1.0 + 1e-6))

                target_action6 = np.concatenate([state6[1:], state6[-1:]])
                bad_align = int(np.count_nonzero(np.abs(action6 - target_action6) > 1e-6))
                counts["action6_next_state6_alignment_bad"] += bad_align

                is_first = np.asarray(features["steps/is_first"].int64_list.value, dtype=np.int64)
                is_last = np.asarray(features["steps/is_last"].int64_list.value, dtype=np.int64)
                is_terminal = np.asarray(features["steps/is_terminal"].int64_list.value, dtype=np.int64)
                for key, arr in [
                    ("steps/is_first", is_first),
                    ("steps/is_last", is_last),
                    ("steps/is_terminal", is_terminal),
                ]:
                    if int(np.count_nonzero((arr != 0) & (arr != 1))) > 0:
                        errors.append(f"{shard.name} example {shard_examples}: {key} has non-bool values")
                if not (is_first[0] == 1 and int(is_first.sum()) == 1):
                    errors.append(f"{shard.name} example {shard_examples}: invalid is_first flags")
                if not (is_last[-1] == 1 and int(is_last.sum()) == 1):
                    errors.append(f"{shard.name} example {shard_examples}: invalid is_last flags")
                if not (is_terminal[-1] == 1 and int(is_terminal.sum()) == 1):
                    errors.append(f"{shard.name} example {shard_examples}: invalid is_terminal flags")
        except Exception as exc:  # noqa: BLE001 - catches TFRecord CRC/read failures.
            errors.append(f"{shard.name}: TFRecord read failed: {exc}")

        shard_example_counts.append(shard_examples)
        if expected_lengths is not None and shard_idx < len(expected_lengths):
            expected_examples = expected_lengths[shard_idx]
            if shard_examples != expected_examples:
                errors.append(f"{shard.name}: shard examples={shard_examples}, expected={expected_examples}")
        if (shard_idx + 1) == 1 or (shard_idx + 1) % args.progress_every == 0 or (shard_idx + 1) == len(shards):
            print(
                f"[{shard_idx + 1:04d}/{len(shards):04d}] "
                f"examples={counts['examples']} steps={counts['steps']} errors={len(errors)}",
                flush=True,
            )

    duplicate_episode_ids = {k: v for k, v in episode_ids.items() if v > 1}
    if duplicate_episode_ids:
        warnings.append(f"duplicate episode_id values: {duplicate_episode_ids}")
    if expected_lengths is not None and counts["examples"] != sum(expected_lengths):
        errors.append(f"total examples={counts['examples']}, expected={sum(expected_lengths)}")

    summary = {
        "version_dir": str(version_dir),
        "num_shards": len(shards),
        "expected_shards_from_dataset_info": len(expected_lengths) if expected_lengths is not None else None,
        "num_examples": int(counts["examples"]),
        "num_steps": int(counts["steps"]),
        "shard_example_counts": shard_example_counts,
        "minmax": {k: (None if math.isinf(v) else v) for k, v in sorted(minmax.items())},
        "gripper_checks": {
            "state6_below_0": int(counts["state6_below_0"]),
            "state6_above_1": int(counts["state6_above_1"]),
            "action6_below_0": int(counts["action6_below_0"]),
            "action6_above_1": int(counts["action6_above_1"]),
            "action6_next_state6_alignment_bad": int(counts["action6_next_state6_alignment_bad"]),
        },
        "image_payload_counts": dict(image_payload_counts),
        "image_decode_counts": dict(image_decode_counts),
        "text_empty_counts": dict(text_empty_counts),
        "text_invalid_utf8_counts": dict(text_invalid_utf8_counts),
        "duplicate_episode_ids": duplicate_episode_ids,
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors[:200],
        "warnings": warnings[:200],
        "decode_images": args.decode_images,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k not in {"shard_example_counts", "errors", "warnings"}}, indent=2))
    if errors:
        print(f"VALIDATION_FAILED errors={len(errors)} warnings={len(warnings)}")
        return 1
    print(f"VALIDATION_OK warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
