#!/usr/bin/env python3
"""Safely patch semantic annotation fields in TransVTLA RLDS examples.

This module starts with the pure manifest and ``tf.train.Example`` mutation
core.  Dataset streaming, candidate verification, and publication are kept out
of these helpers so their safety contracts can be tested without reading an
RLDS dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


EPISODE_ID_FEATURE = "episode_metadata/episode_id"
ANNOTATION_KEY_FEATURE = "episode_metadata/annotation_key"
SOURCE_FILE_FEATURE = "episode_metadata/source_file"
TASK_ID_FEATURE = "episode_metadata/task_id"
SLOT_PAIR_FEATURE = "episode_metadata/slot_pair"
LANGUAGE_INSTRUCTION_FEATURE = "steps/language_instruction"
ACTION_PROMPT_FEATURE = "steps/action_prompt"
STEP_COUNT_FEATURE = "steps/is_first"

ALLOWED_PATCH_FEATURES = {
    TASK_ID_FEATURE,
    SLOT_PAIR_FEATURE,
    LANGUAGE_INSTRUCTION_FEATURE,
    ACTION_PROMPT_FEATURE,
}

ACTION_PROMPT_PREFIX = b"What action should the robot take to execute: "
CTX_PREFIX = b"CTX:"
TRUTHY = {"1", "true", "yes", "y"}
DEFAULT_EXPECTED_EPISODES = 1000

CANONICAL_TASK_DISTRIBUTION = Counter(
    {
        "task1_simple_put_object_in_box": 230,
        "task2_complex_put_object_or_beaker_tripod": 291,
        "task3_simple_beaker_tripod": 225,
        "task4_test_tube_rack_insertion": 254,
    }
)

CANONICAL_INSTRUCTION_DISTRIBUTION = Counter(
    {
        "Pick up the beaker and place it on the tripod.": 225,
        "Pick up the test tube from slot 1 and insert it into slot 2 of the test tube rack.": 154,
        "Find the large beaker in the current environment, pick it up, and place it into the box.": 126,
        "Pick up the small hollow transparent cube and place it into the box.": 89,
        "Pick up the big transparent cube and place it into the box.": 82,
        "Find the big transparent cube in the current environment, pick it up, and place it into the box.": 81,
        "Find the small hollow transparent cube in the current environment, pick it up, and place it into the box.": 58,
        "Pick up the test tube from slot 1 and insert it into slot 5 of the test tube rack.": 54,
        "Pick up the test tube from slot 1 and insert it into slot 3 of the test tube rack.": 46,
        "Pick up the dropper bottle and place it into the box.": 32,
        "Pick up the cola cup and place it into the box.": 27,
        "Find the small beaker in the current environment, pick it up, and place it into the box.": 26,
    }
)

CANONICAL_SLOT_DISTRIBUTION = Counter(
    {"": 746, "1-2": 154, "1-5": 54, "1-3": 46}
)


@dataclass(frozen=True)
class ExpectedAnnotation:
    """Manifest-derived identity and replacement values for one episode."""

    episode_id: int
    annotation_key: str
    source_file: str
    task_id: str
    slot_pair: str
    language_instruction: str


@dataclass(frozen=True)
class VerificationReport:
    """Machine-readable evidence for one exact source/candidate version pair."""

    status: str
    examples: int
    valid_episode_ids: int
    unique_episode_ids: int
    missing_ids: list[int]
    extra_ids: list[int]
    duplicate_ids: list[int]
    identity_mismatches: list[str]
    unexpected_changed_features: list[str]
    semantic_mismatches: list[str]
    dataset_info_mismatches: list[str]
    source_shard_filenames: list[str]
    candidate_shard_filenames: list[str]
    source_shard_lengths: list[int]
    candidate_shard_lengths: list[int]
    task_counts: dict[str, int]
    instruction_counts: dict[str, int]
    slot_counts: dict[str, int]
    source_immutable_digest: str
    candidate_immutable_digest: str
    immutable_digest_records: int
    immutable_digest_complete: bool

    @property
    def immutable_digest_identity(self) -> bool:
        return (
            self.immutable_digest_complete
            and self.source_immutable_digest == self.candidate_immutable_digest
        )

    @property
    def semantic_mismatch_count(self) -> int:
        return len(self.semantic_mismatches)

    @property
    def dataset_info_mismatch_count(self) -> int:
        return len(self.dataset_info_mismatches)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        counted_details = (
            ("identity_mismatches", "identity_mismatch_details"),
            (
                "unexpected_changed_features",
                "unexpected_changed_feature_details",
            ),
            ("semantic_mismatches", "semantic_mismatch_details"),
            ("dataset_info_mismatches", "dataset_info_mismatch_details"),
        )
        for count_name, details_name in counted_details:
            details = result[count_name]
            result[details_name] = details
            result[count_name] = len(details)
        result["immutable_digest_identity"] = self.immutable_digest_identity
        result["source_shard_count"] = len(self.source_shard_filenames)
        result["candidate_shard_count"] = len(self.candidate_shard_filenames)
        return result


def _value_or(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def _iter_selected_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[int, Mapping[str, Any]]]:
    """Yield strict-real rows using build_annotated_rlds selection rules."""

    for source_index, row in enumerate(rows):
        if _value_or(row.get("data_source"), "unknown") != "real":
            continue
        if not _parse_bool(row.get("usable_for_aligned_training")):
            continue
        if _parse_bool(row.get("needs_review")):
            continue
        if not _value_or(row.get("rel_path")):
            continue
        yield source_index, row


def select_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return rows selected for strict aligned real-data training."""

    return [row for _, row in _iter_selected_manifest_rows(rows)]


def validate_expected_annotation_ids(
    annotations: Iterable[ExpectedAnnotation],
    *,
    expected_count: int = DEFAULT_EXPECTED_EPISODES,
) -> None:
    """Require one annotation for every zero-based episode ID."""

    episode_ids = [annotation.episode_id for annotation in annotations]
    duplicate_ids = sorted(
        episode_id
        for episode_id, count in Counter(episode_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate episode IDs: {duplicate_ids}")

    expected_ids = set(range(expected_count))
    actual_ids = set(episode_ids)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            f"missing episode IDs: {missing_ids}; extra episode IDs: {extra_ids}"
        )


def index_manifest_annotations(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_count: int = DEFAULT_EXPECTED_EPISODES,
) -> dict[int, ExpectedAnnotation]:
    """Build the authoritative annotation map keyed by selected-row ordinal."""

    selected = list(_iter_selected_manifest_rows(rows))
    if len(selected) != expected_count:
        raise ValueError(
            "manifest must contain exactly "
            f"{expected_count} selected rows; found {len(selected)}"
        )

    annotations = [
        ExpectedAnnotation(
            episode_id=episode_id,
            annotation_key=_value_or(
                row.get("annotation_key"), f"row_{source_index:06d}"
            ),
            source_file=_value_or(row.get("rel_path")),
            task_id=_value_or(row.get("task_id"), "unknown"),
            slot_pair=_value_or(row.get("slot_pair")),
            language_instruction=_value_or(
                row.get("language_instruction"), "Pick up the target object."
            ),
        )
        for episode_id, (source_index, row) in enumerate(selected)
    ]
    validate_expected_annotation_ids(annotations, expected_count=expected_count)
    return {annotation.episode_id: annotation for annotation in annotations}


def load_manifest_annotations(
    path: str | Path,
    *,
    expected_count: int = DEFAULT_EXPECTED_EPISODES,
    require_canonical_distributions: bool = True,
) -> dict[int, ExpectedAnnotation]:
    """Load, strictly select, index, and validate the canonical manifest."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        annotations = index_manifest_annotations(
            csv.DictReader(handle), expected_count=expected_count
        )
    if require_canonical_distributions:
        validate_canonical_distributions(annotations.values())
    return annotations


def validate_canonical_distributions(
    annotations: Iterable[ExpectedAnnotation],
) -> dict[str, Counter[str]]:
    """Validate the exact canonical task, instruction, and slot counts."""

    annotation_list = list(annotations)
    actual = {
        "task_id": Counter(annotation.task_id for annotation in annotation_list),
        "language_instruction": Counter(
            annotation.language_instruction for annotation in annotation_list
        ),
        "slot_pair": Counter(annotation.slot_pair for annotation in annotation_list),
    }
    expected = {
        "task_id": CANONICAL_TASK_DISTRIBUTION,
        "language_instruction": CANONICAL_INSTRUCTION_DISTRIBUTION,
        "slot_pair": CANONICAL_SLOT_DISTRIBUTION,
    }
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        raise ValueError(
            "canonical distribution mismatch for: " + ", ".join(mismatches)
        )
    return actual


def _require_int64_scalar(features: Any, name: str) -> int:
    feature = features.get(name)
    if feature is None or feature.WhichOneof("kind") != "int64_list":
        raise ValueError(f"{name} must be an int64 scalar")
    values = feature.int64_list.value
    if len(values) != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return int(values[0])


def _require_bytes_scalar(features: Any, name: str) -> bytes:
    feature = features.get(name)
    if feature is None or feature.WhichOneof("kind") != "bytes_list":
        raise ValueError(f"{name} must be a bytes scalar")
    values = feature.bytes_list.value
    if len(values) != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return bytes(values[0])


def _require_bytes_sequence(features: Any, name: str, n_steps: int) -> list[bytes]:
    feature = features.get(name)
    if feature is None or feature.WhichOneof("kind") != "bytes_list":
        raise ValueError(f"{name} must be a bytes sequence")
    values = [bytes(value) for value in feature.bytes_list.value]
    if len(values) != n_steps:
        raise ValueError(
            f"{name} must contain exactly {n_steps} values; found {len(values)}"
        )
    return values


def _require_step_count(features: Any) -> int:
    feature = features.get(STEP_COUNT_FEATURE)
    if feature is None or feature.WhichOneof("kind") != "int64_list":
        raise ValueError(f"{STEP_COUNT_FEATURE} must be an int64 sequence")
    n_steps = len(feature.int64_list.value)
    if n_steps < 1:
        raise ValueError("example must contain at least one step")
    return n_steps


def _validate_single_line(value: bytes, description: str) -> None:
    if not value or b"\n" in value or b"\r" in value:
        raise ValueError(f"{description} must be non-empty and contain no newline")


def _prompt_suffix(old_instruction: bytes, prompt: bytes, step_index: int) -> bytes:
    _validate_single_line(old_instruction, f"step {step_index} action prompt instruction")
    expected_prefix = ACTION_PROMPT_PREFIX + old_instruction + b"\n"
    if not prompt.startswith(expected_prefix):
        raise ValueError(
            f"step {step_index} action prompt does not exactly match its instruction prefix"
        )
    suffix = prompt[len(expected_prefix) :]
    if (
        not suffix.startswith(CTX_PREFIX)
        or len(suffix) == len(CTX_PREFIX)
        or b"\n" in suffix
        or b"\r" in suffix
    ):
        raise ValueError(
            f"step {step_index} action prompt must have one non-empty CTX suffix"
        )
    return suffix


def _validate_example_identity(
    features: Any, expected: ExpectedAnnotation
) -> None:
    actual_episode_id = _require_int64_scalar(features, EPISODE_ID_FEATURE)
    actual_annotation_key = _require_bytes_scalar(features, ANNOTATION_KEY_FEATURE)
    actual_source_file = _require_bytes_scalar(features, SOURCE_FILE_FEATURE)
    identity_values = (
        ("episode_id", actual_episode_id, expected.episode_id),
        (
            "annotation_key",
            actual_annotation_key,
            expected.annotation_key.encode("utf-8"),
        ),
        (
            "source_file",
            actual_source_file,
            expected.source_file.encode("utf-8"),
        ),
    )
    for name, actual, wanted in identity_values:
        if actual != wanted:
            raise ValueError(
                f"{name} identity mismatch: expected {wanted!r}, found {actual!r}"
            )


def patch_example(example: Any, expected: ExpectedAnnotation) -> Any:
    """Validate identity and patch exactly four semantic Example features.

    Validation is completed before any mutation, so a rejected example is left
    untouched.  The object is returned for convenient functional composition.
    """

    features = example.features.feature
    _validate_example_identity(features, expected)

    new_instruction = expected.language_instruction.encode("utf-8")
    _validate_single_line(new_instruction, "manifest language_instruction")
    n_steps = _require_step_count(features)
    old_instructions = _require_bytes_sequence(
        features, LANGUAGE_INSTRUCTION_FEATURE, n_steps
    )
    old_prompts = _require_bytes_sequence(features, ACTION_PROMPT_FEATURE, n_steps)
    suffixes = [
        _prompt_suffix(instruction, prompt, step_index)
        for step_index, (instruction, prompt) in enumerate(
            zip(old_instructions, old_prompts, strict=True)
        )
    ]

    replacements = {
        TASK_ID_FEATURE: [expected.task_id.encode("utf-8")],
        SLOT_PAIR_FEATURE: [expected.slot_pair.encode("utf-8")],
        LANGUAGE_INSTRUCTION_FEATURE: [new_instruction] * n_steps,
        ACTION_PROMPT_FEATURE: [
            ACTION_PROMPT_PREFIX + new_instruction + b"\n" + suffix
            for suffix in suffixes
        ],
    }
    for name, values in replacements.items():
        target = features[name].bytes_list.value
        del target[:]
        target.extend(values)
    return example


def _tensorflow() -> Any:
    import tensorflow as tf

    return tf


def _shards(version_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in version_dir.iterdir()
        if path.is_file() and ".tfrecord-" in path.name
    )


def _is_stale_statistics(path: Path) -> bool:
    return path.name.startswith("dataset_statistics_")


def _is_tfrecord(path: Path) -> bool:
    return path.is_file() and ".tfrecord-" in path.name


def _reject_version_symlinks(version: Path, label: str) -> None:
    symlinks = [path for path in version.rglob("*") if path.is_symlink()]
    if version.is_symlink():
        symlinks.insert(0, version)
    if symlinks:
        location = (
            version.name
            if symlinks[0] == version
            else str(symlinks[0].relative_to(version))
        )
        raise ValueError(f"{label} metadata symlink is not allowed: {location}")


def _copy_metadata(source_version: Path, candidate_version: Path) -> None:
    for source in sorted(source_version.rglob("*")):
        relative = source.relative_to(source_version)
        if any(_is_stale_statistics(part) for part in relative.parents):
            continue
        if _is_stale_statistics(source) or _is_tfrecord(source):
            continue
        target = candidate_version / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _train_split(dataset_info: dict[str, Any]) -> dict[str, Any]:
    matches = [split for split in dataset_info.get("splits", []) if split.get("name") == "train"]
    if len(matches) != 1:
        raise ValueError("dataset_info.json must contain exactly one train split")
    return matches[0]


def _update_dataset_info(candidate_version: Path, shard_lengths: list[int]) -> None:
    info_path = candidate_version / "dataset_info.json"
    if not info_path.is_file():
        raise ValueError("source version must contain dataset_info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    train = _train_split(info)
    train["numBytes"] = str(sum(path.stat().st_size for path in _shards(candidate_version)))
    train["shardLengths"] = [str(length) for length in shard_lengths]
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def _validate_annotation_map(
    annotations: Mapping[int, ExpectedAnnotation],
    expected_count: int,
    require_canonical_distributions: bool,
) -> None:
    validate_expected_annotation_ids(annotations.values(), expected_count=expected_count)
    if set(annotations) != {annotation.episode_id for annotation in annotations.values()}:
        raise ValueError("annotation mapping keys must equal annotation episode IDs")
    if require_canonical_distributions:
        validate_canonical_distributions(annotations.values())


def generate_candidate_version(
    source_version: str | Path,
    candidate_version: str | Path,
    annotations: Mapping[int, ExpectedAnnotation],
    *,
    expected_count: int = DEFAULT_EXPECTED_EPISODES,
    require_canonical_distributions: bool = True,
    report_path: str | Path | None = None,
) -> VerificationReport:
    """Stream a source TFDS version into a sibling annotation candidate."""

    source = Path(source_version)
    candidate = Path(candidate_version)
    if not source.is_dir():
        raise ValueError(f"source version is not a directory: {source}")
    if candidate.is_symlink():
        raise ValueError(f"candidate version must not be a symlink: {candidate}")
    if (
        candidate.parent.resolve() != source.parent.resolve()
        or candidate.resolve() == source.resolve()
    ):
        raise ValueError("candidate version must be a distinct sibling of the source version")
    if candidate.exists() and (not candidate.is_dir() or any(candidate.iterdir())):
        raise ValueError(f"candidate already exists and is non-empty: {candidate}")
    _validate_annotation_map(annotations, expected_count, require_canonical_distributions)
    source_shards = _shards(source)
    if not source_shards:
        raise ValueError(f"source version contains no TFRecord shards: {source}")
    _reject_version_symlinks(source, "source")

    if candidate.exists():
        candidate.rmdir()
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{candidate.name}.tmp-",
            dir=str(candidate.parent),
        )
    )

    tf = _tensorflow()
    try:
        _copy_metadata(source, temporary)
        seen_ids: list[int] = []
        shard_lengths: list[int] = []
        for source_shard in source_shards:
            count = 0
            with tf.io.TFRecordWriter(str(temporary / source_shard.name)) as writer:
                for raw in tf.compat.v1.io.tf_record_iterator(str(source_shard)):
                    example = tf.train.Example.FromString(raw)
                    episode_id = _require_int64_scalar(
                        example.features.feature, EPISODE_ID_FEATURE
                    )
                    expected = annotations.get(episode_id)
                    if expected is None:
                        raise ValueError(f"extra episode ID: {episode_id}")
                    patch_example(example, expected)
                    writer.write(example.SerializeToString(deterministic=True))
                    seen_ids.append(episode_id)
                    count += 1
            shard_lengths.append(count)

        duplicates = sorted(
            key for key, count in Counter(seen_ids).items() if count > 1
        )
        expected_ids = set(range(expected_count))
        actual_ids = set(seen_ids)
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if duplicates:
            raise ValueError(f"duplicate episode IDs: {duplicates}")
        if missing or extra:
            raise ValueError(
                f"missing episode IDs: {missing}; extra episode IDs: {extra}"
            )
        _update_dataset_info(temporary, shard_lengths)
        report = verify_version_pair(
            source,
            temporary,
            annotations,
            expected_count=expected_count,
            require_canonical_distributions=require_canonical_distributions,
        )
        temporary.rename(candidate)
        if report_path is not None:
            _write_report(report, report_path, source, candidate)
        return report
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _decode_scalar(features: Any, name: str) -> str:
    return _require_bytes_scalar(features, name).decode("utf-8")


def _digest_feature(
    digest: Any, shard_name: str, record_index: int, name: str, serialized: bytes
) -> None:
    for value in (
        shard_name.encode("utf-8"),
        str(record_index).encode("ascii"),
        name.encode("utf-8"),
        serialized,
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)


def _metadata_snapshot(version: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in sorted(version.rglob("*")):
        if not path.is_file() or _is_tfrecord(path) or _is_stale_statistics(path):
            continue
        relative = str(path.relative_to(version))
        if relative == "dataset_info.json":
            info = json.loads(path.read_text(encoding="utf-8"))
            train = _train_split(info)
            train["numBytes"] = "<updated>"
            train["shardLengths"] = ["<updated>"]
            result[relative] = json.dumps(info, sort_keys=True).encode("utf-8")
        else:
            result[relative] = path.read_bytes()
    return result


def _write_report(
    report: VerificationReport,
    report_path: str | Path,
    source_version: str | Path | None = None,
    candidate_version: str | Path | None = None,
) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = report.to_dict()
    if source_version is not None and candidate_version is not None:
        source = Path(source_version)
        candidate = Path(candidate_version)
        result.update(
            source_version=str(source.resolve()),
            candidate_version=str(candidate.resolve()),
            source_identity=_path_identity(source),
            candidate_identity=_path_identity(candidate),
        )
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_report_path(
    report_path: str | Path | None, source: Path, candidate: Path
) -> None:
    if report_path is None:
        return
    path = Path(report_path)
    resolved = path.resolve()
    if resolved.is_relative_to(source.resolve()) or resolved.is_relative_to(
        candidate.resolve()
    ):
        raise ValueError("verification report must be outside source and candidate versions")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"verification report path is not a file: {path}")
        path.unlink()


def verify_version_pair(
    source_version: str | Path,
    candidate_version: str | Path,
    annotations: Mapping[int, ExpectedAnnotation],
    *,
    expected_count: int = DEFAULT_EXPECTED_EPISODES,
    require_canonical_distributions: bool = True,
    report_path: str | Path | None = None,
) -> VerificationReport:
    """Scan an exact source/candidate pair and validate all patch invariants."""

    source = Path(source_version)
    candidate = Path(candidate_version)
    if not source.is_dir() or not candidate.is_dir():
        raise ValueError("source and candidate versions must both be directories")
    _prepare_report_path(report_path, source, candidate)
    _reject_version_symlinks(source, "source")
    _reject_version_symlinks(candidate, "candidate")
    _validate_annotation_map(annotations, expected_count, require_canonical_distributions)
    source_shards = _shards(source)
    candidate_shards = _shards(candidate)
    source_names = [path.name for path in source_shards]
    candidate_names = [path.name for path in candidate_shards]
    if source_names != candidate_names:
        raise ValueError(
            f"shard filenames differ: source={source_names}, candidate={candidate_names}"
        )
    if _metadata_snapshot(source) != _metadata_snapshot(candidate):
        raise ValueError("candidate metadata differs beyond dataset_info train byte metadata")
    if any(path.name.startswith("dataset_statistics_") for path in candidate.rglob("*")):
        raise ValueError("candidate contains a stale dataset statistics cache")
    candidate_info = json.loads(
        (candidate / "dataset_info.json").read_text(encoding="utf-8")
    )
    candidate_train = _train_split(candidate_info)

    tf = _tensorflow()
    source_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    seen_ids: list[int] = []
    scanned_examples = 0
    source_lengths: list[int] = []
    candidate_lengths: list[int] = []
    identity_mismatches: list[str] = []
    unexpected: list[str] = []
    semantic_mismatches: list[str] = []
    task_counts: Counter[str] = Counter()
    instruction_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    immutable_digest_records = 0

    for source_shard, candidate_shard in zip(source_shards, candidate_shards, strict=True):
        source_records = tf.compat.v1.io.tf_record_iterator(str(source_shard))
        candidate_records = tf.compat.v1.io.tf_record_iterator(str(candidate_shard))
        source_count = 0
        candidate_count = 0
        sentinel = object()
        record_index = 0
        while True:
            source_raw = next(source_records, sentinel)
            candidate_raw = next(candidate_records, sentinel)
            if source_raw is sentinel and candidate_raw is sentinel:
                break
            if source_raw is sentinel or candidate_raw is sentinel:
                raise ValueError(f"per-shard example count differs for {source_shard.name}")
            source_count += 1
            candidate_count += 1
            scanned_examples += 1
            source_example = tf.train.Example.FromString(source_raw)
            candidate_example = tf.train.Example.FromString(candidate_raw)
            source_features = source_example.features.feature
            candidate_features = candidate_example.features.feature
            names = sorted(set(source_features) | set(candidate_features))
            for name in names:
                if name in ALLOWED_PATCH_FEATURES:
                    continue
                source_feature = source_features.get(name)
                candidate_feature = candidate_features.get(name)
                source_serialized = (
                    None
                    if source_feature is None
                    else source_feature.SerializeToString(deterministic=True)
                )
                candidate_serialized = (
                    None
                    if candidate_feature is None
                    else candidate_feature.SerializeToString(deterministic=True)
                )
                location = f"{source_shard.name}:{record_index}:{name}"
                if source_serialized is not None:
                    _digest_feature(
                        source_digest,
                        source_shard.name,
                        record_index,
                        name,
                        source_serialized,
                    )
                if candidate_serialized is not None:
                    _digest_feature(
                        candidate_digest,
                        candidate_shard.name,
                        record_index,
                        name,
                        candidate_serialized,
                    )
                if source_serialized != candidate_serialized:
                    kind = (
                        "added"
                        if source_serialized is None
                        else "removed"
                        if candidate_serialized is None
                        else "changed"
                    )
                    unexpected.append(f"{location}:{kind}")
                del source_serialized, candidate_serialized
                del source_feature, candidate_feature
            immutable_digest_records += 1

            expected: ExpectedAnnotation | None = None
            try:
                episode_id = _require_int64_scalar(
                    source_features, EPISODE_ID_FEATURE
                )
            except ValueError as exc:
                identity_mismatches.append(f"{source_shard.name}:{record_index}: {exc}")
            else:
                seen_ids.append(episode_id)
                expected = annotations.get(episode_id)
                if expected is None:
                    identity_mismatches.append(
                        f"{source_shard.name}:{record_index}: "
                        f"unexpected episode_id {episode_id}"
                    )

            source_patched = False
            if expected is not None:
                try:
                    patch_example(source_example, expected)
                    source_patched = True
                except ValueError as exc:
                    identity_mismatches.append(
                        f"{source_shard.name}:{record_index}: {exc}"
                    )
                try:
                    _validate_example_identity(candidate_features, expected)
                except ValueError as exc:
                    identity_mismatches.append(
                        f"{source_shard.name}:{record_index}: candidate {exc}"
                    )
                if source_patched:
                    for name in ALLOWED_PATCH_FEATURES:
                        expected_feature = source_features.get(name)
                        candidate_feature = candidate_features.get(name)
                        expected_serialized = (
                            None
                            if expected_feature is None
                            else expected_feature.SerializeToString(deterministic=True)
                        )
                        candidate_serialized = (
                            None
                            if candidate_feature is None
                            else candidate_feature.SerializeToString(deterministic=True)
                        )
                        if candidate_serialized != expected_serialized:
                            semantic_mismatches.append(
                                f"{source_shard.name}:{record_index}:{name}"
                            )
                        del expected_serialized, candidate_serialized
                        del expected_feature, candidate_feature

            try:
                task_counts[_decode_scalar(candidate_features, TASK_ID_FEATURE)] += 1
                slot_counts[_decode_scalar(candidate_features, SLOT_PAIR_FEATURE)] += 1
                instructions = _require_bytes_sequence(
                    candidate_features,
                    LANGUAGE_INSTRUCTION_FEATURE,
                    _require_step_count(candidate_features),
                )
                instruction_counts[instructions[0].decode("utf-8")] += 1
            except (UnicodeDecodeError, ValueError) as exc:
                semantic_mismatches.append(f"{source_shard.name}:{record_index}: {exc}")
            del source_raw, candidate_raw, source_example, candidate_example
            del source_features, candidate_features, names
            record_index += 1
        source_lengths.append(source_count)
        candidate_lengths.append(candidate_count)

    expected_ids = set(range(expected_count))
    actual_ids = set(seen_ids)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    duplicates = sorted(key for key, count in Counter(seen_ids).items() if count > 1)
    expected_num_bytes = str(sum(path.stat().st_size for path in candidate_shards))
    expected_shard_lengths = [str(length) for length in candidate_lengths]
    dataset_info_mismatches: list[str] = []
    if candidate_train.get("numBytes") != expected_num_bytes:
        dataset_info_mismatches.append(
            "train numBytes: "
            f"expected {expected_num_bytes!r}, found {candidate_train.get('numBytes')!r}"
        )
    if candidate_train.get("shardLengths") != expected_shard_lengths:
        dataset_info_mismatches.append(
            "train shardLengths: "
            f"expected {expected_shard_lengths!r}, "
            f"found {candidate_train.get('shardLengths')!r}"
        )
    canonical_mismatches: list[str] = []
    if require_canonical_distributions:
        actual_distributions = {
            "task_id": task_counts,
            "language_instruction": instruction_counts,
            "slot_pair": slot_counts,
        }
        expected_distributions = {
            "task_id": CANONICAL_TASK_DISTRIBUTION,
            "language_instruction": CANONICAL_INSTRUCTION_DISTRIBUTION,
            "slot_pair": CANONICAL_SLOT_DISTRIBUTION,
        }
        canonical_mismatches = [
            name
            for name in expected_distributions
            if actual_distributions[name] != expected_distributions[name]
        ]
    immutable_digest_complete = immutable_digest_records == scanned_examples
    failed = bool(
        source_lengths != candidate_lengths
        or missing
        or extra
        or duplicates
        or identity_mismatches
        or unexpected
        or not immutable_digest_complete
        or source_digest.digest() != candidate_digest.digest()
        or semantic_mismatches
        or dataset_info_mismatches
        or canonical_mismatches
    )
    report = VerificationReport(
        status="failed" if failed else "success",
        examples=scanned_examples,
        valid_episode_ids=len(seen_ids),
        unique_episode_ids=len(actual_ids),
        missing_ids=missing,
        extra_ids=extra,
        duplicate_ids=duplicates,
        identity_mismatches=identity_mismatches,
        unexpected_changed_features=unexpected,
        semantic_mismatches=semantic_mismatches,
        dataset_info_mismatches=dataset_info_mismatches,
        source_shard_filenames=source_names,
        candidate_shard_filenames=candidate_names,
        source_shard_lengths=source_lengths,
        candidate_shard_lengths=candidate_lengths,
        task_counts=dict(sorted(task_counts.items())),
        instruction_counts=dict(sorted(instruction_counts.items())),
        slot_counts=dict(sorted(slot_counts.items())),
        source_immutable_digest=source_digest.hexdigest(),
        candidate_immutable_digest=candidate_digest.hexdigest(),
        immutable_digest_records=immutable_digest_records,
        immutable_digest_complete=immutable_digest_complete,
    )
    if report_path is not None:
        _write_report(report, report_path, source, candidate)
    if source_lengths != candidate_lengths:
        raise ValueError("source/candidate per-shard example counts differ")
    if missing or extra or duplicates:
        raise ValueError(
            f"episode coverage mismatch: missing={missing}, extra={extra}, duplicates={duplicates}"
        )
    if identity_mismatches:
        raise ValueError("identity mismatches: " + "; ".join(identity_mismatches))
    if unexpected or not report.immutable_digest_identity:
        raise ValueError("immutable feature drift: " + "; ".join(unexpected))
    if semantic_mismatches:
        raise ValueError("semantic feature mismatch: " + "; ".join(semantic_mismatches))
    if dataset_info_mismatches:
        raise ValueError(
            "candidate dataset_info.json does not match shards: "
            + "; ".join(dataset_info_mismatches)
        )
    if canonical_mismatches:
        raise ValueError(
            "canonical distribution mismatch for: "
            + ", ".join(canonical_mismatches)
        )
    return report


@dataclass(frozen=True)
class SwapSpec:
    """The four exact paths participating in one publication swap."""

    source: Path
    candidate: Path
    backup: Path
    report: Path

    def __init__(
        self,
        source: str | Path,
        candidate: str | Path,
        backup: str | Path,
        report: str | Path,
    ) -> None:
        object.__setattr__(self, "source", Path(source))
        object.__setattr__(self, "candidate", Path(candidate))
        object.__setattr__(self, "backup", Path(backup))
        object.__setattr__(self, "report", Path(report))


def _path_identity(path: str | Path) -> dict[str, int | str]:
    """Return the rename-stable identity of a filesystem object."""

    value = Path(path)
    info = value.lstat()
    kind = (
        "directory"
        if stat.S_ISDIR(info.st_mode)
        else "file"
        if stat.S_ISREG(info.st_mode)
        else "other"
    )
    identity: dict[str, int | str] = {
        "device": info.st_dev,
        "inode": info.st_ino,
        "kind": kind,
    }
    if kind == "file":
        identity.update(size=info.st_size, mtime_ns=info.st_mtime_ns)
        identity["content_sha256"] = hashlib.sha256(value.read_bytes()).hexdigest()
    elif kind == "directory":
        digest = hashlib.sha256()
        entries = 0
        for child in sorted(value.rglob("*"), key=lambda item: str(item.relative_to(value))):
            child_info = child.lstat()
            relative = str(child.relative_to(value)).encode("utf-8")
            fields = (
                relative,
                str(stat.S_IFMT(child_info.st_mode)).encode("ascii"),
                str(child_info.st_dev).encode("ascii"),
                str(child_info.st_ino).encode("ascii"),
                str(child_info.st_size).encode("ascii"),
                str(child_info.st_mtime_ns).encode("ascii"),
            )
            for field in fields:
                digest.update(len(field).to_bytes(8, "big"))
                digest.update(field)
            entries += 1
        identity.update(entries=entries, entries_digest=digest.hexdigest())
    return identity


def _identity_at(path: Path) -> dict[str, int | str] | None:
    try:
        return _path_identity(path)
    except FileNotFoundError:
        return None


def _same_identity(path: Path, expected: Mapping[str, Any]) -> bool:
    return _identity_at(path) == dict(expected)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_parents(*paths: Path) -> None:
    parents = sorted({path.parent.resolve() for path in paths}, key=str)
    for parent in parents:
        _fsync_directory(parent)


def _write_journal(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent.resolve())
    finally:
        if temporary.exists():
            temporary.unlink()


def _rename(source: Path, destination: Path) -> None:
    os.rename(source, destination)


def _publication_checkpoint(_label: str) -> None:
    """Crash-injection seam used to exercise durable transaction boundaries."""


def _successful_exact_report(swap: SwapSpec) -> dict[str, Any]:
    if swap.report.is_symlink() or not swap.report.is_file():
        raise ValueError(f"verification report is not a regular file: {swap.report}")
    try:
        report = json.loads(swap.report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid verification report: {swap.report}") from exc
    required_success = {
        "status": "success",
        "examples": DEFAULT_EXPECTED_EPISODES,
        "unique_episode_ids": DEFAULT_EXPECTED_EPISODES,
        "missing_ids": [],
        "extra_ids": [],
        "duplicate_ids": [],
        "identity_mismatches": 0,
        "unexpected_changed_features": 0,
        "semantic_mismatches": 0,
        "dataset_info_mismatches": 0,
        "immutable_digest_identity": True,
        "source_version": str(swap.source.resolve()),
        "candidate_version": str(swap.candidate.resolve()),
        "source_identity": _path_identity(swap.source),
        "candidate_identity": _path_identity(swap.candidate),
    }
    mismatches = [
        name for name, expected in required_success.items() if report.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            f"verification report does not match exact successful pair: {mismatches}"
        )
    return report


def _reject_symlink_ancestors(path: str | Path, label: str) -> None:
    """Reject a symlink in any lexical component of a publication path."""

    absolute = Path(path).absolute()
    for component in reversed((absolute, *absolute.parents)):
        if component.is_symlink():
            raise ValueError(f"{label} path has symlink ancestor: {component}")


def _preflight_publication(swaps: Iterable[SwapSpec], journal: Path) -> list[dict[str, Any]]:
    values = list(swaps)
    if len(values) != 2:
        raise ValueError("publication requires exactly two version swaps")
    _reject_symlink_ancestors(journal, "journal")
    if journal.is_symlink():
        raise ValueError("publication journal may not be a symlink")

    resolved_paths: list[Path] = [journal.resolve()]
    inode_identities: list[tuple[int, int]] = []
    prepared: list[dict[str, Any]] = []
    for swap in values:
        for label, path in (
            ("source", swap.source),
            ("candidate", swap.candidate),
            ("backup", swap.backup),
            ("report", swap.report),
        ):
            _reject_symlink_ancestors(path, label)
        if not swap.source.is_dir() or not swap.candidate.is_dir():
            raise ValueError("source and candidate must both exist as directories")
        if swap.source.is_symlink() or swap.candidate.is_symlink():
            raise ValueError("source and candidate may not be symlinks")
        _reject_version_symlinks(swap.source, "source")
        _reject_version_symlinks(swap.candidate, "candidate")
        if swap.backup.exists() or swap.backup.is_symlink():
            raise ValueError(f"backup path already exists: {swap.backup}")
        for destination in (swap.source.parent, swap.backup.parent):
            if not destination.is_dir() or destination.is_symlink():
                raise ValueError(f"rename parent is not a real directory: {destination}")
        source_identity = _path_identity(swap.source)
        candidate_identity = _path_identity(swap.candidate)
        if source_identity["device"] != swap.backup.parent.stat().st_dev:
            raise ValueError("source and backup are not on the same filesystem")
        if candidate_identity["device"] != swap.source.parent.stat().st_dev:
            raise ValueError("candidate and source are not on the same filesystem")
        _successful_exact_report(swap)
        report_identity = _path_identity(swap.report)
        for path in (swap.source, swap.candidate, swap.backup, swap.report):
            resolved = path.resolve()
            if resolved in resolved_paths:
                raise ValueError(f"publication paths alias: {path}")
            resolved_paths.append(resolved)
        for identity in (source_identity, candidate_identity, report_identity):
            key = (int(identity["device"]), int(identity["inode"]))
            if key in inode_identities:
                raise ValueError("publication paths alias by inode")
            inode_identities.append(key)
        prepared.append(
            {
                "source": str(swap.source.resolve()),
                "candidate": str(swap.candidate.resolve()),
                "backup": str(swap.backup.resolve()),
                "report": str(swap.report.resolve()),
                "source_identity": source_identity,
                "candidate_identity": candidate_identity,
                "report_identity": report_identity,
                "source_rename": "pending",
                "candidate_rename": "pending",
                "post_swap_identity_verified": False,
            }
        )
    return prepared


def _durable_forward_rename(
    journal: Path,
    data: dict[str, Any],
    index: int,
    kind: str,
    source: Path,
    destination: Path,
) -> None:
    item = data["swaps"][index]
    field = f"{kind}_rename"
    item[field] = "before"
    _write_journal(journal, data)
    _publication_checkpoint(f"before_{kind}_rename_{index}")
    _rename(source, destination)
    _publication_checkpoint(f"after_{kind}_rename_{index}")
    _fsync_parents(source, destination)
    item[field] = "complete"
    _write_journal(journal, data)
    _publication_checkpoint(f"{kind}_rename_complete_{index}")


def _locate_identity(item: Mapping[str, Any], identity_name: str) -> Path:
    expected = item[identity_name]
    paths = [Path(item[name]) for name in ("source", "candidate", "backup")]
    matches = [path for path in paths if _same_identity(path, expected)]
    if len(matches) != 1:
        raise RuntimeError(
            f"cannot recover: {identity_name} identity has {len(matches)} locations"
        )
    return matches[0]


def _durable_inverse_rename(
    journal: Path,
    data: dict[str, Any],
    index: int,
    label: str,
    source: Path,
    destination: Path,
) -> None:
    data["state"] = "rolling_back"
    data["swaps"][index]["rollback_phase"] = f"before_{label}"
    _write_journal(journal, data)
    _publication_checkpoint(f"before_inverse_{label}_{index}")
    _rename(source, destination)
    _publication_checkpoint(f"after_inverse_{label}_{index}")
    _fsync_parents(source, destination)
    _publication_checkpoint(f"after_parents_fsync_inverse_{label}_{index}")
    data["swaps"][index]["rollback_phase"] = f"complete_{label}"
    _write_journal(journal, data)
    _publication_checkpoint(f"complete_inverse_{label}_{index}")


def _rollback_publication(journal: Path, data: dict[str, Any]) -> None:
    try:
        for index in reversed(range(len(data["swaps"]))):
            item = data["swaps"][index]
            source = Path(item["source"])
            candidate = Path(item["candidate"])
            backup = Path(item["backup"])
            candidate_location = _locate_identity(item, "candidate_identity")
            source_location = _locate_identity(item, "source_identity")
            if candidate_location == source:
                if candidate.exists() or candidate.is_symlink():
                    raise RuntimeError("candidate destination occupied by unknown identity")
                _durable_inverse_rename(
                    journal, data, index, "candidate", source, candidate
                )
            elif candidate_location != candidate:
                raise RuntimeError("candidate identity is at an invalid path")
            source_location = _locate_identity(item, "source_identity")
            if source_location == backup:
                if source.exists() or source.is_symlink():
                    raise RuntimeError("source destination occupied by unknown identity")
                _durable_inverse_rename(journal, data, index, "source", backup, source)
            elif source_location != source:
                raise RuntimeError("source identity is at an invalid path")
            if index > 0:
                _publication_checkpoint(f"between_rollback_swaps_{index}")
        for item in data["swaps"]:
            if not _same_identity(Path(item["source"]), item["source_identity"]):
                raise RuntimeError("source identity was not restored")
            if not _same_identity(Path(item["candidate"]), item["candidate_identity"]):
                raise RuntimeError("candidate identity was not restored")
            if Path(item["backup"]).exists() or Path(item["backup"]).is_symlink():
                raise RuntimeError("backup still exists after rollback")
        _fsync_parents(
            *(Path(item[name]) for item in data["swaps"] for name in ("source", "candidate", "backup"))
        )
        _publication_checkpoint("before_rolled_back")
        data["state"] = "rolled_back"
        _write_journal(journal, data)
        _publication_checkpoint("after_rolled_back")
    except Exception as exc:
        data["state"] = "rollback_failed"
        data["rollback_error"] = f"{type(exc).__name__}: {exc}"
        _write_journal(journal, data)
        raise RuntimeError(f"publication rollback failed: {exc}") from exc


def _validate_committed(data: Mapping[str, Any]) -> None:
    for item in data["swaps"]:
        if not _same_identity(Path(item["source"]), item["candidate_identity"]):
            raise RuntimeError("committed journal source identity mismatch")
        if not _same_identity(Path(item["backup"]), item["source_identity"]):
            raise RuntimeError("committed journal backup identity mismatch")
        candidate = Path(item["candidate"])
        if candidate.exists() or candidate.is_symlink():
            raise RuntimeError("committed journal candidate path unexpectedly exists")


def recover_publication(journal_path: str | Path) -> bool:
    """Recover an uncommitted transaction; validate committed transactions."""

    journal = Path(journal_path)
    _reject_symlink_ancestors(journal, "journal")
    if not journal.exists():
        return False
    if journal.is_symlink() or not journal.is_file():
        raise RuntimeError(f"publication journal is not a regular file: {journal}")
    try:
        data = json.loads(journal.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"unreadable publication journal: {journal}") from exc
    if data.get("state") == "committed":
        _validate_committed(data)
        return False
    if data.get("state") == "rolled_back":
        for item in data.get("swaps", []):
            if not _same_identity(Path(item["source"]), item["source_identity"]):
                raise RuntimeError("rolled-back journal source identity mismatch")
            if not _same_identity(Path(item["candidate"]), item["candidate_identity"]):
                raise RuntimeError("rolled-back journal candidate identity mismatch")
        return False
    if data.get("state") == "rollback_failed":
        raise RuntimeError(f"publication journal records rollback failure: {journal}")
    _rollback_publication(journal, data)
    return True


def publish_versions(
    swaps: Iterable[SwapSpec],
    journal_path: str | Path,
    *,
    preflight_only: bool = False,
) -> None:
    """Preflight or durably publish exactly two verified candidate versions."""

    journal = Path(journal_path)
    if journal.exists() or journal.is_symlink():
        recover_publication(journal)
    values = list(swaps)
    prepared = _preflight_publication(values, journal)
    if preflight_only:
        print("PREFLIGHT_OK versions=2")
        return
    data: dict[str, Any] = {"format_version": 1, "state": "prepared", "swaps": prepared}
    _write_journal(journal, data)
    _publication_checkpoint("journal_initialized")
    try:
        for index, item in enumerate(prepared):
            # Re-check the exact objects and report immediately before this swap.
            swap = values[index]
            _successful_exact_report(swap)
            if not _same_identity(swap.report, item["report_identity"]):
                raise RuntimeError("verification report identity changed after preflight")
            _durable_forward_rename(
                journal, data, index, "source", swap.source, swap.backup
            )
            _durable_forward_rename(
                journal, data, index, "candidate", swap.candidate, swap.source
            )
            if not _same_identity(swap.source, item["candidate_identity"]):
                raise RuntimeError("post-swap candidate identity mismatch")
            if not _same_identity(swap.backup, item["source_identity"]):
                raise RuntimeError("post-swap backup identity mismatch")
            item["post_swap_identity_verified"] = True
            _write_journal(journal, data)
        _fsync_parents(
            *(Path(item[name]) for item in prepared for name in ("source", "candidate", "backup"))
        )
        _publication_checkpoint("before_commit")
        data["state"] = "committed"
        _write_journal(journal, data)
        _publication_checkpoint("committed")
    except Exception:
        _rollback_publication(journal, data)
        raise
    print("PUBLISH_COMMITTED versions=2")


def _parse_swap(value: str) -> SwapSpec:
    parts = value.split(",")
    if len(parts) != 4 or any(not part for part in parts):
        raise argparse.ArgumentTypeError("--swap requires source,candidate,backup,report")
    return SwapSpec(*parts)


def _add_manifest_pair_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--journal")


def _recover_related_journals(
    explicit: str | None, version_paths: Iterable[str | Path]
) -> bool:
    journals: list[Path] = []
    if explicit is not None:
        journals.append(Path(explicit))
    for version_path in version_paths:
        parent = Path(version_path).absolute().parent
        journals.extend(sorted(parent.glob(".semantic_annotation_patch_transaction*.json")))
    seen: set[Path] = set()
    recovered = False
    for journal in journals:
        resolved = journal.absolute()
        if resolved not in seen:
            recovered = recover_publication(journal) or recovered
            seen.add(resolved)
    return recovered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    patch_parser = commands.add_parser("patch")
    _add_manifest_pair_arguments(patch_parser)
    patch_parser.add_argument("--report", required=True)
    verify_parser = commands.add_parser("verify")
    _add_manifest_pair_arguments(verify_parser)
    verify_parser.add_argument("--report", required=True)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--journal", required=True)
    publish_parser.add_argument("--swap", action="append", type=_parse_swap, required=True)
    publish_parser.add_argument("--preflight-only", action="store_true")
    recover_parser = commands.add_parser("recover")
    recover_parser.add_argument("--journal", required=True)
    args = parser.parse_args(argv)

    if args.command == "recover":
        recovered = _recover_related_journals(args.journal, (args.journal,))
        print(f"RECOVERY_{'COMPLETE' if recovered else 'NOT_NEEDED'}")
        return 0
    if args.command == "publish":
        related_paths = [args.journal]
        related_paths.extend(
            path
            for swap in args.swap
            for path in (swap.source, swap.candidate, swap.backup, swap.report)
        )
        for path in related_paths:
            _reject_symlink_ancestors(path, "publication")
        _recover_related_journals(args.journal, related_paths)
        publish_versions(args.swap, args.journal, preflight_only=args.preflight_only)
        return 0
    for label, path in (
        ("source", args.source_version),
        ("candidate", args.candidate_version),
        ("report", args.report),
    ):
        _reject_symlink_ancestors(path, label)
    if args.journal is not None:
        _reject_symlink_ancestors(args.journal, "journal")
    journal_neighbors = [args.source_version, args.candidate_version, args.report]
    if args.journal is not None:
        journal_neighbors.append(args.journal)
    _recover_related_journals(args.journal, journal_neighbors)
    annotations = load_manifest_annotations(
        args.manifest,
        expected_count=DEFAULT_EXPECTED_EPISODES,
        require_canonical_distributions=True,
    )
    common = {
        "expected_count": DEFAULT_EXPECTED_EPISODES,
        "require_canonical_distributions": True,
    }
    if args.command == "patch":
        generate_candidate_version(
            args.source_version,
            args.candidate_version,
            annotations,
            report_path=args.report,
            **common,
        )
    else:
        verify_version_pair(
            args.source_version,
            args.candidate_version,
            annotations,
            report_path=args.report,
            **common,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
