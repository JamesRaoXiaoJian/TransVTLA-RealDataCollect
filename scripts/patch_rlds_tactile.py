#!/usr/bin/env python3
"""Patch tactile tensors in an existing TransVTLA TFDS/RLDS dataset.

The original rlds_real1000_strict conversion wrote zero tactile tensors because
the pressure npz files were missing at conversion time. Rebuilding the full RLDS
dataset is slow because it decodes and re-encodes all image/depth arrays. This
script keeps the existing TFRecord examples and replaces only the packed
float32 bytes for steps/observation/tactile.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from session_schema import gripper_state_csv_path, robot_state_csv_path  # noqa: E402
from scripts.build_annotated_rlds import (  # noqa: E402
    EpisodeRow,
    RGB_EXT,
    TACTILE_SHAPE,
    aligned_state,
    aligned_tactile,
    load_gripper_csv,
    load_numpy,
    load_robot_csv,
    parse_bool,
    repo_path,
    resolve_session_layout_safe,
    safe_stems,
    select_rows,
    stage_arrays,
    value_or,
)


TACTILE_FEATURE = "steps/observation/tactile"
SOURCE_FILE_FEATURE = "episode_metadata/source_file"
ANNOTATION_KEY_FEATURE = "episode_metadata/annotation_key"
EPISODE_ID_FEATURE = "episode_metadata/episode_id"
TFRECORD_NAME = "transvtla_annotated_rlds"


@dataclass(frozen=True)
class PatchStats:
    frames: int
    nonzero: int
    minimum: float
    maximum: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch TransVTLA RLDS tactile tensors.")
    parser.add_argument("--manifest", required=True, help="annotation manifest CSV path")
    parser.add_argument("--source-rlds", required=True, help="existing TFDS data_dir to read")
    parser.add_argument("--output-rlds", required=True, help="new TFDS data_dir to write")
    parser.add_argument("--version", default="3.0.0", help="TFDS version directory")
    parser.add_argument("--data-source", choices=["all", "real", "sim"], default="real")
    parser.add_argument("--split", default="", help="Optional exact split filter")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--allow-review", action="store_true")
    parser.add_argument("--include-unusable", action="store_true")
    parser.add_argument("--max-records", type=int, default=None, help="Patch only the first N records")
    parser.add_argument("--summary-every", type=int, default=25)
    parser.add_argument("--allow-missing-tactile", action="store_true")
    parser.add_argument("--no-mask-tactile-before-stage1", action="store_true")
    return parser.parse_args()


def selected_episodes(args: argparse.Namespace) -> list[EpisodeRow]:
    selector_args = SimpleNamespace(
        manifest=args.manifest,
        data_source=args.data_source,
        split=args.split,
        max_episodes=args.max_episodes,
        allow_review=args.allow_review,
        include_unusable=args.include_unusable,
    )
    return select_rows(selector_args)


def version_dir(data_dir: Path, version: str) -> Path:
    return data_dir / TFRECORD_NAME / version


def legacy_rlds_visual_stems(session_dir: Path) -> tuple[list[str], Any | None, str]:
    """Reproduce the frame selection used by the existing strict RLDS shards.

    The current builder prefers frames.csv when present. The already-built
    rlds_real1000_strict shards were generated from the common RGB file stem
    intersection instead, so patching must use the same count to replace the
    packed tactile bytes without changing any other field.
    """
    layout = resolve_session_layout_safe(session_dir)
    if layout is None:
        return [], None, "missing dual RealSense or legacy RGB layout"

    world = safe_stems(layout.world.rgb, RGB_EXT)
    wrist = safe_stems(layout.wrist.rgb, RGB_EXT)
    stems = sorted(world & wrist)
    if not stems:
        return [], layout, "missing common RGB frames"
    return stems, layout, ""


def tfrecord_files(data_dir: Path, version: str) -> list[Path]:
    files = sorted(version_dir(data_dir, version).glob(f"{TFRECORD_NAME}-train.tfrecord-*"))
    if not files:
        raise SystemExit(f"No TFRecord shards found under {version_dir(data_dir, version)}")
    return files


def copy_metadata_files(src_version_dir: Path, dst_version_dir: Path) -> None:
    for path in src_version_dir.rglob("*"):
        if not path.is_file():
            continue
        if ".tfrecord-" in path.name:
            continue
        rel = path.relative_to(src_version_dir)
        dst = dst_version_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def read_varint(buf: bytes | bytearray | memoryview, pos: int, limit: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < limit:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift >= 70:
            break
    raise ValueError("invalid protobuf varint")


def iter_fields(
    buf: bytes | bytearray | memoryview,
    start: int,
    end: int,
) -> Iterable[tuple[int, int, int, int]]:
    pos = start
    while pos < end:
        tag, pos = read_varint(buf, pos, end)
        field_no = tag >> 3
        wire_type = tag & 0x07
        if field_no <= 0:
            raise ValueError(f"invalid protobuf field number {field_no}")

        if wire_type == 0:
            value_start = pos
            _, pos = read_varint(buf, pos, end)
            yield field_no, wire_type, value_start, pos
        elif wire_type == 1:
            value_start = pos
            pos += 8
            if pos > end:
                raise ValueError("truncated fixed64 protobuf field")
            yield field_no, wire_type, value_start, pos
        elif wire_type == 2:
            length, pos = read_varint(buf, pos, end)
            value_start = pos
            value_end = pos + length
            if value_end > end:
                raise ValueError("truncated length-delimited protobuf field")
            yield field_no, wire_type, value_start, value_end
            pos = value_end
        elif wire_type == 5:
            value_start = pos
            pos += 4
            if pos > end:
                raise ValueError("truncated fixed32 protobuf field")
            yield field_no, wire_type, value_start, pos
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")


def find_feature_message(record: bytes | bytearray, feature_key: str) -> tuple[int, int]:
    key_bytes = feature_key.encode("utf-8")
    record_view = memoryview(record)
    for field_no, wire_type, features_start, features_end in iter_fields(record_view, 0, len(record_view)):
        if field_no != 1 or wire_type != 2:
            continue
        for map_field, map_wire, entry_start, entry_end in iter_fields(record_view, features_start, features_end):
            if map_field != 1 or map_wire != 2:
                continue
            key: bytes | None = None
            feature_range: tuple[int, int] | None = None
            for entry_field, entry_wire, value_start, value_end in iter_fields(record_view, entry_start, entry_end):
                if entry_field == 1 and entry_wire == 2:
                    key = bytes(record_view[value_start:value_end])
                elif entry_field == 2 and entry_wire == 2:
                    feature_range = (value_start, value_end)
            if key == key_bytes and feature_range is not None:
                return feature_range
    raise KeyError(feature_key)


def find_tactile_float_bytes(record: bytes | bytearray) -> tuple[int, int]:
    feature_start, feature_end = find_feature_message(record, TACTILE_FEATURE)
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
            raise ValueError(f"expected one packed float range for {TACTILE_FEATURE}, found {len(packed_ranges)}")
        return packed_ranges[0]
    raise KeyError(f"{TACTILE_FEATURE} float_list")


def feature_bytes_values(record: bytes | bytearray, feature_key: str) -> list[bytes]:
    feature_start, feature_end = find_feature_message(record, feature_key)
    record_view = memoryview(record)
    values: list[bytes] = []
    for field_no, wire_type, bytes_list_start, bytes_list_end in iter_fields(
        record_view, feature_start, feature_end
    ):
        if field_no != 1 or wire_type != 2:
            continue
        for list_field, list_wire, value_start, value_end in iter_fields(
            record_view, bytes_list_start, bytes_list_end
        ):
            if list_field == 1 and list_wire == 2:
                values.append(bytes(record_view[value_start:value_end]))
    return values


def feature_text(record: bytes | bytearray, feature_key: str) -> str:
    values = feature_bytes_values(record, feature_key)
    if not values:
        raise KeyError(feature_key)
    return values[0].decode("utf-8")


def feature_int64_values(record: bytes | bytearray, feature_key: str) -> list[int]:
    feature_start, feature_end = find_feature_message(record, feature_key)
    record_view = memoryview(record)
    values: list[int] = []
    for field_no, wire_type, int64_list_start, int64_list_end in iter_fields(
        record_view, feature_start, feature_end
    ):
        if field_no != 3 or wire_type != 2:
            continue
        for list_field, list_wire, value_start, value_end in iter_fields(
            record_view, int64_list_start, int64_list_end
        ):
            if list_field != 1:
                continue
            if list_wire == 0:
                value, _ = read_varint(record_view, value_start, value_end)
                values.append(value)
            elif list_wire == 2:
                pos = value_start
                while pos < value_end:
                    value, pos = read_varint(record_view, pos, value_end)
                    values.append(value)
            else:
                raise ValueError(f"unsupported int64_list wire type {list_wire}")
    return values


def iter_tfrecord_records(path: Path) -> Iterable[bytes]:
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError(f"truncated TFRecord length header in {path}")
            length = struct.unpack("<Q", header)[0]
            length_crc = f.read(4)
            if len(length_crc) != 4:
                raise ValueError(f"truncated TFRecord length CRC in {path}")
            data = f.read(length)
            if len(data) != length:
                raise ValueError(f"truncated TFRecord data in {path}")
            data_crc = f.read(4)
            if len(data_crc) != 4:
                raise ValueError(f"truncated TFRecord data CRC in {path}")
            yield data


def build_tactile_payload(
    episode: EpisodeRow,
    np: Any,
    *,
    mask_tactile_before_stage1: bool,
    allow_missing_tactile: bool,
) -> tuple[bytes, PatchStats]:
    stems, layout, reason = legacy_rlds_visual_stems(episode.session_dir)
    if not stems or layout is None:
        raise RuntimeError(f"{episode.rel_path}: {reason}")

    robot_ts, robot_poses = load_robot_csv(robot_state_csv_path(episode.session_dir), np)
    if len(robot_ts) == 0 or len(robot_poses) == 0:
        raise RuntimeError(f"{episode.rel_path}: missing robot_state")
    gripper_ts, gripper_pos = load_gripper_csv(gripper_state_csv_path(episode.session_dir), np)

    timestamps, poses, gripper = aligned_state(
        episode.session_dir, stems, robot_ts, robot_poses, gripper_ts, gripper_pos, np
    )
    n = min(len(stems), len(poses), len(gripper), len(timestamps))
    if n < 2:
        raise RuntimeError(f"{episode.rel_path}: too few aligned frames ({n})")
    timestamps = timestamps[:n]

    tactile = aligned_tactile(episode.session_dir, timestamps, np)
    if tactile is None:
        if not allow_missing_tactile:
            raise RuntimeError(f"{episode.rel_path}: missing aligned tactile")
        tactile_array = np.zeros((n, *TACTILE_SHAPE), dtype=np.float32)
    else:
        tactile_array = np.asarray(tactile[:n], dtype=np.float32)
        if tactile_array.shape != (n, *TACTILE_SHAPE):
            raise RuntimeError(f"{episode.rel_path}: unexpected tactile shape {tactile_array.shape}")

    if mask_tactile_before_stage1:
        _, use_tactile, _ = stage_arrays(episode.row, n, np)
        tactile_array = tactile_array.copy()
        tactile_array[~use_tactile[:n]] = 0.0

    if not np.isfinite(tactile_array).all():
        raise RuntimeError(f"{episode.rel_path}: non-finite tactile values")
    minimum = float(tactile_array.min()) if tactile_array.size else 0.0
    maximum = float(tactile_array.max()) if tactile_array.size else 0.0
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise RuntimeError(f"{episode.rel_path}: tactile range out of bounds min={minimum} max={maximum}")

    tactile_array = np.ascontiguousarray(tactile_array.astype("<f4", copy=False))
    stats = PatchStats(
        frames=n,
        nonzero=int(np.count_nonzero(tactile_array)),
        minimum=minimum,
        maximum=maximum,
    )
    return tactile_array.tobytes(order="C"), stats


def patch_record(
    record: bytes,
    episode: EpisodeRow,
    np: Any,
    *,
    mask_tactile_before_stage1: bool,
    allow_missing_tactile: bool,
) -> tuple[bytes, PatchStats]:
    payload, stats = build_tactile_payload(
        episode,
        np,
        mask_tactile_before_stage1=mask_tactile_before_stage1,
        allow_missing_tactile=allow_missing_tactile,
    )
    start, end = find_tactile_float_bytes(record)
    old_len = end - start
    if old_len != len(payload):
        raise RuntimeError(
            f"{episode.rel_path}: tactile byte length mismatch old={old_len} new={len(payload)} "
            f"frames={stats.frames}"
        )
    patched = bytearray(record)
    patched[start:end] = payload
    return bytes(patched), stats


def import_tfrecord_writer() -> Any:
    try:
        from tensorflow.python.lib.io.tf_record import TFRecordWriter  # type: ignore
    except Exception as exc:
        raise SystemExit("TensorFlow is required for TFRecord writing") from exc
    return TFRecordWriter


def patch_dataset(args: argparse.Namespace) -> int:
    np = load_numpy()
    TFRecordWriter = import_tfrecord_writer()

    source_root = repo_path(args.source_rlds)
    output_root = repo_path(args.output_rlds)
    src_version_dir = version_dir(source_root, args.version)
    dst_version_dir = version_dir(output_root, args.version)
    if output_root.exists():
        raise SystemExit(f"Output path already exists: {output_root}")
    dst_version_dir.mkdir(parents=True, exist_ok=False)
    copy_metadata_files(src_version_dir, dst_version_dir)

    episodes = selected_episodes(args)
    if not episodes:
        raise SystemExit("No episodes selected from manifest")

    record_count = 0
    total_frames = 0
    total_nonzero = 0
    global_min = 1.0
    global_max = 0.0
    mask_tactile_before_stage1 = not args.no_mask_tactile_before_stage1

    files = tfrecord_files(source_root, args.version)
    for src_file in files:
        dst_file = dst_version_dir / src_file.name
        writer = TFRecordWriter(str(dst_file))
        try:
            for record in iter_tfrecord_records(src_file):
                annotation_key = feature_text(record, ANNOTATION_KEY_FEATURE)
                source_file = feature_text(record, SOURCE_FILE_FEATURE)
                episode_ids = feature_int64_values(record, EPISODE_ID_FEATURE)
                if len(episode_ids) != 1:
                    raise RuntimeError(f"Expected one episode_id, found {len(episode_ids)}")
                episode_id = episode_ids[0]
                if episode_id < 0 or episode_id >= len(episodes):
                    raise RuntimeError(f"Record episode_id out of range: {episode_id}")
                episode = episodes[episode_id]
                if value_or(episode.rel_path) != source_file:
                    raise RuntimeError(
                        "Record source_file does not match manifest: "
                        f"annotation_key={annotation_key} record={source_file} manifest={episode.rel_path}"
                    )
                if value_or(episode.annotation_key) != annotation_key:
                    raise RuntimeError(
                        "Record annotation_key does not match manifest: "
                        f"episode_id={episode_id} record={annotation_key} manifest={episode.annotation_key}"
                    )

                patched_record, stats = patch_record(
                    record,
                    episode,
                    np,
                    mask_tactile_before_stage1=mask_tactile_before_stage1,
                    allow_missing_tactile=args.allow_missing_tactile,
                )
                writer.write(patched_record)

                record_count += 1
                total_frames += stats.frames
                total_nonzero += stats.nonzero
                global_min = min(global_min, stats.minimum)
                global_max = max(global_max, stats.maximum)
                if args.summary_every > 0 and record_count % args.summary_every == 0:
                    print(
                        f"patched records={record_count} frames={total_frames} "
                        f"nonzero={total_nonzero} range={global_min:.6f}..{global_max:.6f}",
                        flush=True,
                    )
                if args.max_records is not None and record_count >= args.max_records:
                    break
        finally:
            writer.close()
        if args.max_records is not None and record_count >= args.max_records:
            break

    expected = len(episodes)
    if args.max_records is None and record_count != expected:
        raise RuntimeError(f"Patched record count {record_count} does not match selected episodes {expected}")
    if args.max_records is None and total_nonzero == 0:
        raise RuntimeError("Patched tactile data is still all zero")

    print(f"source={source_root}")
    print(f"output={output_root}")
    print(f"selected_episodes={expected}")
    print(f"patched_records={record_count}")
    print(f"frames={total_frames}")
    print(f"tactile_nonzero={total_nonzero}")
    print(f"tactile_range={global_min:.6f}..{global_max:.6f}")
    return 0


def main() -> int:
    args = parse_args()
    return patch_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
