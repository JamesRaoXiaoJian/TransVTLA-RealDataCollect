#!/usr/bin/env python3
"""Apply only Task 4 language-instruction refinements to an RLDS version.

The script deliberately leaves every feature except
``steps/language_instruction`` unchanged.  Each changed TFRecord shard is
replaced atomically and its original is retained as a hard-link backup, so a
failed run cannot partially corrupt a shard and rollback needs no dataset copy.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shutil
import struct
import tempfile
from collections import Counter
from pathlib import Path


TASK4 = "task4_test_tube_rack_insertion"
UNREFINED_TASK4_INSTRUCTION = (
    "Pick up the test tube from slot 1 and insert it into slot 2 of the test tube rack."
)
EPISODE_ID = "episode_metadata/episode_id"
TASK_ID = "episode_metadata/task_id"
LANGUAGE = "steps/language_instruction"
CRC32C_TABLE = tuple(
    (lambda value: value)(
        __import__("functools").reduce(
            lambda value, _: (value >> 1) ^ (0x82F63B78 if value & 1 else 0),
            range(8),
            index,
        )
    )
    for index in range(256)
)


def _crc32c(data: bytes) -> int:
    """Return the Castagnoli CRC used by the TFRecord framing format."""

    value = 0xFFFFFFFF
    for byte in data:
        value = CRC32C_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return ((crc >> 15) | (crc << 17) & 0xFFFFFFFF) + 0xA282EAD8 & 0xFFFFFFFF


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("invalid protobuf varint")


def _fields(data: bytes, start: int, end: int):
    offset = start
    while offset < end:
        key, offset = _varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            _, value_end = _varint(data, offset)
            yield number, wire_type, offset, value_end
            offset = value_end
        elif wire_type == 2:
            size, value_start = _varint(data, offset)
            value_end = value_start + size
            if value_end > end:
                raise ValueError("truncated protobuf length-delimited field")
            yield number, wire_type, value_start, value_end
            offset = value_end
        elif wire_type == 5:
            value_end = offset + 4
            yield number, wire_type, offset, value_end
            offset = value_end
        elif wire_type == 1:
            value_end = offset + 8
            yield number, wire_type, offset, value_end
            offset = value_end
        else:
            raise ValueError(f"unsupported protobuf wire type: {wire_type}")


def _feature_block(example: bytes, name: str) -> tuple[int, int]:
    """Locate one named Feature message in a raw tf.train.Example."""

    features_blocks = [
        (start, end)
        for number, wire, start, end in _fields(example, 0, len(example))
        if number == 1 and wire == 2
    ]
    if len(features_blocks) != 1:
        raise ValueError("Example must contain one Features message")
    (features_start, features_end), = features_blocks
    for number, wire, entry_start, entry_end in _fields(
        example, features_start, features_end
    ):
        if number != 1 or wire != 2:
            continue
        key = None
        feature_block = None
        for entry_number, entry_wire, value_start, value_end in _fields(
            example, entry_start, entry_end
        ):
            if entry_number == 1 and entry_wire == 2:
                key = example[value_start:value_end]
            elif entry_number == 2 and entry_wire == 2:
                feature_block = (value_start, value_end)
        if key != name.encode("utf-8") or feature_block is None:
            continue
        return feature_block
    raise ValueError(f"Example is missing {name}")


def _bytes_value_offsets(example: bytes, name: str) -> list[tuple[int, int]]:
    feature_start, feature_end = _feature_block(example, name)
    bytes_list_blocks = [
            (start, end)
            for feature_number, feature_wire, start, end in _fields(
                example, feature_start, feature_end
            )
            if feature_number == 1 and feature_wire == 2
        ]
    if len(bytes_list_blocks) != 1:
        raise ValueError(f"{name} Feature must contain one BytesList")
    bytes_start, bytes_end = bytes_list_blocks[0]
    return [
        (start, end)
        for value_number, value_wire, start, end in _fields(
            example, bytes_start, bytes_end
        )
        if value_number == 1 and value_wire == 2
    ]


def _bytes_values(example: bytes, name: str) -> list[bytes]:
    return [example[start:end] for start, end in _bytes_value_offsets(example, name)]


def _int64_values(example: bytes, name: str) -> list[int]:
    feature_start, feature_end = _feature_block(example, name)
    blocks = [
        (start, end)
        for number, wire, start, end in _fields(example, feature_start, feature_end)
        if number == 3 and wire == 2
    ]
    if len(blocks) != 1:
        raise ValueError(f"{name} Feature must contain one Int64List")
    start, end = blocks[0]
    values: list[int] = []
    for number, wire, value_start, value_end in _fields(example, start, end):
        if number != 1:
            continue
        if wire == 0:
            values.append(_varint(example, value_start)[0])
        elif wire == 2:  # proto3 packs repeated numeric fields by default.
            offset = value_start
            while offset < value_end:
                value, offset = _varint(example, offset)
                values.append(value)
        else:
            raise ValueError(f"unsupported Int64List wire type: {wire}")
    return values


def scalar_bytes(features, name: str) -> bytes:
    feature = features.get(name)
    if feature is None or feature.WhichOneof("kind") != "bytes_list":
        raise ValueError(f"missing bytes feature: {name}")
    values = feature.bytes_list.value
    if len(values) != 1:
        raise ValueError(f"{name} must be scalar")
    return bytes(values[0])


def scalar_int(features, name: str) -> int:
    feature = features.get(name)
    if feature is None or feature.WhichOneof("kind") != "int64_list":
        raise ValueError(f"missing int64 feature: {name}")
    values = feature.int64_list.value
    if len(values) != 1:
        raise ValueError(f"{name} must be scalar")
    return int(values[0])


def load_task4_languages(manifest: Path) -> dict[int, str]:
    selected: list[dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("data_source", "").strip() == "real"
                and row.get("usable_for_aligned_training", "").strip().lower()
                in {"1", "true", "yes", "y"}
                and row.get("needs_review", "").strip().lower()
                not in {"1", "true", "yes", "y"}
                and row.get("rel_path", "").strip()
            ):
                selected.append(row)
    if len(selected) != 1000:
        raise ValueError(f"expected 1000 selected rows, found {len(selected)}")
    result = {
        episode_id: row["language_instruction"].strip()
        for episode_id, row in enumerate(selected)
        if (
            row.get("task_id", "").strip() == TASK4
            and row["language_instruction"].strip() != UNREFINED_TASK4_INSTRUCTION
        )
    }
    if not result:
        raise ValueError("manifest contains no selected Task 4 rows")
    return result


def shards(version: Path) -> list[Path]:
    return sorted(path for path in version.iterdir() if path.is_file() and ".tfrecord-" in path.name)


def task4_shards(version: Path, expected_episode_ids: set[int]) -> list[Path]:
    """Return only shards whose dataset-info record range intersects Task 4."""

    info = json.loads((version / "dataset_info.json").read_text(encoding="utf-8"))
    train = next(split for split in info["splits"] if split["name"] == "train")
    lengths = [int(value) for value in train["shardLengths"]]
    all_shards = shards(version)
    if len(lengths) != len(all_shards):
        raise ValueError("dataset_info shard lengths do not match TFRecord shards")
    selected: list[Path] = []
    first_episode_id = 0
    for shard, length in zip(all_shards, lengths, strict=True):
        shard_ids = range(first_episode_id, first_episode_id + length)
        if any(episode_id in expected_episode_ids for episode_id in shard_ids):
            selected.append(shard)
        first_episode_id += length
    if first_episode_id != 1000:
        raise ValueError(f"expected 1000 examples in dataset info, found {first_episode_id}")
    return selected


def patch_version(version: Path, manifest: Path, backup_root: Path, report: Path) -> None:
    expected = load_task4_languages(manifest)
    backup = backup_root / version.name
    backup.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    changed_episodes: list[int] = []
    changed_shards: list[str] = []

    # TFDS sharding is not ordered by episode_id; scan all records to locate
    # the requested 100 refined Task 4 episodes.  Only matching language bytes
    # are ever written.
    all_shards = shards(version)
    print(f"scanning {len(all_shards)} shards for Task 4 refinements in {version}", flush=True)
    for shard_index, shard in enumerate(all_shards, start=1):
        if shard_index == 1 or shard_index % 25 == 0:
            print(f"scanning shard {shard_index}/{len(all_shards)}", flush=True)
        shard_changed = False
        with shard.open("r+b") as handle:
            while header := handle.read(12):
                if len(header) != 12:
                    raise ValueError(f"truncated TFRecord header: {shard.name}")
                length = struct.unpack("<Q", header[:8])[0]
                data_offset = handle.tell()
                raw = handle.read(length)
                data_crc_offset = handle.tell()
                if len(raw) != length or len(handle.read(4)) != 4:
                    raise ValueError(f"truncated TFRecord payload: {shard.name}")
                episode_ids = _int64_values(raw, EPISODE_ID)
                if len(episode_ids) != 1:
                    raise ValueError(f"{EPISODE_ID} must be scalar")
                episode_id = episode_ids[0]
                target = expected.get(episode_id)
                if target is None:
                    continue
                task_ids = _bytes_values(raw, TASK_ID)
                if len(task_ids) != 1 or task_ids[0].decode("utf-8") != TASK4:
                    raise ValueError(f"episode {episode_id} is not Task 4 in {shard.name}")
                old_values = _bytes_values(raw, LANGUAGE)
                if not old_values:
                    raise ValueError(f"episode {episode_id} has an empty language sequence")
                new_value = target.encode("utf-8")
                seen.add(episode_id)
                if all(value == new_value for value in old_values):
                    continue
                offsets = _bytes_value_offsets(raw, LANGUAGE)
                if len(offsets) != len(old_values):
                    raise ValueError(
                        f"language value count mismatch for episode {episode_id}"
                    )
                replacement = bytearray(raw)
                for (start, end), old_value in zip(offsets, old_values, strict=True):
                    if raw[start:end] != old_value or len(old_value) != len(new_value):
                        raise ValueError(
                            f"language bytes are not safely patchable for episode {episode_id}"
                        )
                    replacement[start:end] = new_value
                backup_shard = backup / shard.name
                if not shard_changed:
                    if backup_shard.exists():
                        if os.stat(backup_shard).st_ino != os.stat(shard).st_ino:
                            if not filecmp.cmp(backup_shard, shard, shallow=False):
                                raise FileExistsError(
                                    f"backup already exists for a changed shard: {backup_shard}"
                                )
                            # The source was restored from its original backup;
                            # it is already independent and safe to patch.
                            shard_changed = True
                            changed_shards.append(shard.name)
                            
                            # Do not create a second backup link.
                            continue_backup = True
                        else:
                            continue_backup = False
                        if not continue_backup:
                            # A previous interrupted run made the backup link but
                            # had not written any bytes.  Break that link while
                            # retaining the original as the backup.
                            fd, temporary_name = tempfile.mkstemp(
                                prefix=f".{shard.name}.", dir=shard.parent
                            )
                            os.close(fd)
                            temporary = Path(temporary_name)
                            try:
                                shutil.copy2(shard, temporary)
                                os.replace(temporary, shard)
                            finally:
                                temporary.unlink(missing_ok=True)
                    else:
                        os.link(shard, backup_shard)
                    shard_changed = True
                    changed_shards.append(shard.name)
                for start, end in offsets:
                    handle.seek(data_offset + start)
                    handle.write(new_value)
                handle.seek(data_crc_offset)
                handle.write(struct.pack("<I", _masked_crc32c(bytes(replacement))))
                handle.seek(data_crc_offset + 4)
                changed_episodes.append(episode_id)
            if shard_changed:
                handle.flush()
                os.fsync(handle.fileno())

    if seen != set(expected):
        raise ValueError(f"Task 4 episode mismatch: missing={sorted(set(expected) - seen)}")
    result = {
        "version": str(version),
        "manifest": str(manifest),
        "feature_changed": LANGUAGE,
        "task4_episodes_checked": len(expected),
        "episodes_changed": len(changed_episodes),
        "changed_slot_distribution": dict(Counter(expected[item] for item in changed_episodes)),
        "changed_shards": changed_shards,
        "backup_directory": str(backup),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--backup-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    patch_version(args.version, args.manifest, args.backup_root, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
