from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest
import tensorflow as tf
import scripts.patch_rlds_semantic_annotations as patch_module

from scripts.patch_rlds_semantic_annotations import (
    ACTION_PROMPT_FEATURE,
    ALLOWED_PATCH_FEATURES,
    ANNOTATION_KEY_FEATURE,
    CANONICAL_INSTRUCTION_DISTRIBUTION,
    CANONICAL_SLOT_DISTRIBUTION,
    CANONICAL_TASK_DISTRIBUTION,
    EPISODE_ID_FEATURE,
    LANGUAGE_INSTRUCTION_FEATURE,
    SLOT_PAIR_FEATURE,
    SOURCE_FILE_FEATURE,
    TASK_ID_FEATURE,
    ExpectedAnnotation,
    generate_candidate_version,
    index_manifest_annotations,
    load_manifest_annotations,
    patch_example,
    select_manifest_rows,
    validate_canonical_distributions,
    validate_expected_annotation_ids,
    verify_version_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = REPO_ROOT / (
    "dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/"
    "real_1000_train_view_strict/manifest.csv"
)
PROMPT_PREFIX = b"What action should the robot take to execute: "


def manifest_row(**overrides: str) -> dict[str, str]:
    row = {
        "annotation_key": "annotation-0",
        "rel_path": "dataset/sessions/session-0",
        "data_source": "real",
        "usable_for_aligned_training": "yes",
        "needs_review": "false",
        "task_id": "task",
        "slot_pair": "",
        "language_instruction": "Do the task.",
    }
    row.update(overrides)
    return row


def expected_annotation(**overrides: object) -> ExpectedAnnotation:
    values: dict[str, object] = {
        "episode_id": 0,
        "annotation_key": "annotation-0",
        "source_file": "dataset/sessions/session-0",
        "task_id": "new-task",
        "slot_pair": "1-2",
        "language_instruction": "Use the canonical instruction.",
    }
    values.update(overrides)
    return ExpectedAnnotation(**values)  # type: ignore[arg-type]


def make_example(
    *,
    n_steps: int = 2,
    annotation_key: bytes = b"annotation-0",
    source_file: bytes = b"dataset/sessions/session-0",
    episode_id: int = 0,
    instructions: list[bytes] | None = None,
    prompts: list[bytes] | None = None,
) -> tf.train.Example:
    instructions = instructions or [b"Old instruction."] * n_steps
    prompts = prompts or [
        PROMPT_PREFIX + instruction + b"\nCTX:OBJ=cube|PHASE=contact"
        for instruction in instructions
    ]
    example = tf.train.Example()
    features = example.features.feature
    features[EPISODE_ID_FEATURE].int64_list.value.append(episode_id)
    features[ANNOTATION_KEY_FEATURE].bytes_list.value.append(annotation_key)
    features[SOURCE_FILE_FEATURE].bytes_list.value.append(source_file)
    features[TASK_ID_FEATURE].bytes_list.value.append(b"old-task")
    features[SLOT_PAIR_FEATURE].bytes_list.value.append(b"")
    features[LANGUAGE_INSTRUCTION_FEATURE].bytes_list.value.extend(instructions)
    features[ACTION_PROMPT_FEATURE].bytes_list.value.extend(prompts)
    features["steps/is_first"].int64_list.value.extend([1] + [0] * (n_steps - 1))
    features["steps/action"].float_list.value.extend([9.5] * (n_steps * 7))
    features["version_specific/contact_memory"].bytes_list.value.extend([b"opaque"] * n_steps)
    return example


def feature_snapshot(example: tf.train.Example) -> dict[str, bytes]:
    return {
        name: feature.SerializeToString(deterministic=True)
        for name, feature in example.features.feature.items()
    }


def write_shard(path: Path, examples: list[tf.train.Example]) -> None:
    with tf.io.TFRecordWriter(str(path)) as writer:
        for example in examples:
            writer.write(example.SerializeToString(deterministic=True))


def read_shard(path: Path) -> list[tf.train.Example]:
    return [
        tf.train.Example.FromString(raw)
        for raw in tf.compat.v1.io.tf_record_iterator(str(path))
    ]


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()


def make_version(
    root: Path,
    shard_examples: list[list[tf.train.Example]],
) -> list[str]:
    root.mkdir()
    shard_names = [
        f"transvtla_annotated_rlds-train.tfrecord-{index:05d}-of-{len(shard_examples):05d}"
        for index in range(len(shard_examples))
    ]
    for name, examples in zip(shard_names, shard_examples, strict=True):
        write_shard(root / name, examples)
    info = {
        "description": "preserve me",
        "splits": [
            {
                "name": "train",
                "numBytes": "stale",
                "shardLengths": ["stale"],
                "statistics": {"preserve": True},
            }
        ],
    }
    (root / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "features.json").write_text('{"version_specific": true}\n', encoding="utf-8")
    (root / "LICENSE").write_text("metadata\n", encoding="utf-8")
    (root / "dataset_statistics_train.json").write_text("stale\n", encoding="utf-8")
    return shard_names


def annotations_for(count: int) -> dict[int, ExpectedAnnotation]:
    return {
        episode_id: expected_annotation(
            episode_id=episode_id,
            annotation_key=f"annotation-{episode_id}",
            source_file=f"dataset/sessions/session-{episode_id}",
            task_id=f"task-{episode_id % 2}",
            slot_pair="1-2" if episode_id % 2 else "",
            language_instruction=f"Canonical instruction {episode_id}.",
        )
        for episode_id in range(count)
    }


def canonical_annotations() -> dict[int, ExpectedAnnotation]:
    def expanded(counter: Counter[str]) -> list[str]:
        return [value for value, count in counter.items() for _ in range(count)]

    tasks = expanded(CANONICAL_TASK_DISTRIBUTION)
    instructions = expanded(CANONICAL_INSTRUCTION_DISTRIBUTION)
    slots = expanded(CANONICAL_SLOT_DISTRIBUTION)
    return {
        episode_id: expected_annotation(
            episode_id=episode_id,
            annotation_key=f"annotation-{episode_id}",
            source_file=f"dataset/sessions/session-{episode_id}",
            task_id=tasks[episode_id],
            slot_pair=slots[episode_id],
            language_instruction=instructions[episode_id],
        )
        for episode_id in range(1000)
    }


def sync_candidate_dataset_info(candidate: Path) -> None:
    shards = sorted(candidate.glob("*.tfrecord-*"))
    lengths = [len(read_shard(shard)) for shard in shards]
    info_path = candidate / "dataset_info.json"
    info = json.loads(info_path.read_text())
    train = info["splits"][0]
    train["numBytes"] = str(sum(shard.stat().st_size for shard in shards))
    train["shardLengths"] = [str(length) for length in lengths]
    info_path.write_text(json.dumps(info), encoding="utf-8")


def test_manifest_selection_matches_strict_real_builder_rules() -> None:
    rows = [
        manifest_row(annotation_key="keep-yes", usable_for_aligned_training="YES"),
        manifest_row(annotation_key="keep-one", usable_for_aligned_training="1"),
        manifest_row(annotation_key="drop-sim", data_source="sim"),
        manifest_row(annotation_key="drop-unusable", usable_for_aligned_training="0"),
        manifest_row(annotation_key="drop-review", needs_review="true"),
        manifest_row(annotation_key="drop-empty-path", rel_path="  "),
    ]

    selected = select_manifest_rows(rows)

    assert [row["annotation_key"] for row in selected] == ["keep-yes", "keep-one"]


def test_manifest_index_requires_exactly_1000_selected_rows() -> None:
    with pytest.raises(ValueError, match="exactly 1000 selected rows"):
        index_manifest_annotations([manifest_row()], expected_count=1000)


@pytest.mark.parametrize(
    "ids, message",
    [
        ([0, 0], "duplicate"),
        ([0, 2], "missing"),
        ([0, 1, 3], "missing"),
    ],
)
def test_expected_ids_reject_duplicates_and_gaps(
    ids: list[int], message: str
) -> None:
    annotations = [expected_annotation(episode_id=value) for value in ids]

    with pytest.raises(ValueError, match=message):
        validate_expected_annotation_ids(annotations, expected_count=len(ids))


def test_manifest_index_is_keyed_by_zero_based_selected_ordinal() -> None:
    rows = [manifest_row(annotation_key=f"key-{index}") for index in range(3)]

    indexed = index_manifest_annotations(rows, expected_count=3)

    assert list(indexed) == [0, 1, 2]
    assert indexed[2].episode_id == 2
    assert indexed[2].annotation_key == "key-2"
    assert indexed[2].source_file == "dataset/sessions/session-0"


def test_canonical_manifest_has_required_distributions() -> None:
    indexed = load_manifest_annotations(CANONICAL_MANIFEST)
    actual = validate_canonical_distributions(indexed.values())

    assert actual["task_id"] == CANONICAL_TASK_DISTRIBUTION
    assert actual["language_instruction"] == CANONICAL_INSTRUCTION_DISTRIBUTION
    assert actual["slot_pair"] == CANONICAL_SLOT_DISTRIBUTION
    assert sum(actual["task_id"].values()) == 1000


def test_distribution_validation_rejects_semantic_drift() -> None:
    annotations = [
        expected_annotation(task_id="unexpected", language_instruction="Unexpected.")
    ]

    with pytest.raises(ValueError, match="canonical distribution mismatch"):
        validate_canonical_distributions(annotations)


@pytest.mark.parametrize(
    "example_kwargs, mismatch_name",
    [
        ({"episode_id": 9}, "episode_id"),
        ({"annotation_key": b"wrong"}, "annotation_key"),
        ({"source_file": b"wrong"}, "source_file"),
    ],
)
def test_patch_rejects_each_part_of_triple_identity_mismatch(
    example_kwargs: dict[str, object], mismatch_name: str
) -> None:
    example = make_example(**example_kwargs)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=mismatch_name):
        patch_example(example, expected_annotation())


@pytest.mark.parametrize(
    "instructions,prompts,field_name",
    [
        ([b"Old instruction."], None, LANGUAGE_INSTRUCTION_FEATURE),
        (None, [PROMPT_PREFIX + b"Old instruction.\nCTX:x"], ACTION_PROMPT_FEATURE),
    ],
)
def test_patch_requires_one_instruction_and_prompt_per_step(
    instructions: list[bytes] | None,
    prompts: list[bytes] | None,
    field_name: str,
) -> None:
    example = make_example(n_steps=2)
    if instructions is not None:
        del example.features.feature[LANGUAGE_INSTRUCTION_FEATURE].bytes_list.value[:]
        example.features.feature[LANGUAGE_INSTRUCTION_FEATURE].bytes_list.value.extend(instructions)
    if prompts is not None:
        del example.features.feature[ACTION_PROMPT_FEATURE].bytes_list.value[:]
        example.features.feature[ACTION_PROMPT_FEATURE].bytes_list.value.extend(prompts)

    with pytest.raises(ValueError, match=field_name):
        patch_example(example, expected_annotation())


def test_patch_changes_only_allowlisted_features_and_preserves_ctx_bytes() -> None:
    suffixes = [
        b"CTX:OBJ=cube|PHASE=visual_pre_contact",
        b"CTX:OBJ=cube|PHASE=contact|RAW=\xff",
    ]
    old_instructions = [b"Old first.", b"Old second."]
    prompts = [
        PROMPT_PREFIX + instruction + b"\n" + suffix
        for instruction, suffix in zip(old_instructions, suffixes, strict=True)
    ]
    example = make_example(instructions=old_instructions, prompts=prompts)
    before = feature_snapshot(example)

    patch_example(example, expected_annotation())

    after = feature_snapshot(example)
    changed = {name for name in before if before[name] != after[name]}
    assert changed == ALLOWED_PATCH_FEATURES
    features = example.features.feature
    new_instruction = b"Use the canonical instruction."
    assert list(features[LANGUAGE_INSTRUCTION_FEATURE].bytes_list.value) == [
        new_instruction,
        new_instruction,
    ]
    assert list(features[ACTION_PROMPT_FEATURE].bytes_list.value) == [
        PROMPT_PREFIX + new_instruction + b"\n" + suffix for suffix in suffixes
    ]
    assert list(features[TASK_ID_FEATURE].bytes_list.value) == [b"new-task"]
    assert list(features[SLOT_PAIR_FEATURE].bytes_list.value) == [b"1-2"]


@pytest.mark.parametrize(
    "instruction,prompt",
    [
        (b"Old.", b"Wrong prefix: Old.\nCTX:x"),
        (b"Old.", PROMPT_PREFIX + b"Different.\nCTX:x"),
        (b"Old.", PROMPT_PREFIX + b"Old.\n"),
        (b"Old.", PROMPT_PREFIX + b"Old.\nCTX:"),
        (b"Old.\nInjected", PROMPT_PREFIX + b"Old.\nInjected\nCTX:x"),
        (b"Old.", PROMPT_PREFIX + b"Old.\nCTX:x\nInjected"),
        (b"Old.\rInjected", PROMPT_PREFIX + b"Old.\rInjected\nCTX:x"),
    ],
)
def test_patch_rejects_malformed_prompts(instruction: bytes, prompt: bytes) -> None:
    example = make_example(n_steps=1, instructions=[instruction], prompts=[prompt])

    with pytest.raises(ValueError, match="action prompt"):
        patch_example(example, expected_annotation())


def test_patch_rejects_new_manifest_instruction_with_newline() -> None:
    example = make_example(n_steps=1)

    with pytest.raises(ValueError, match="manifest language_instruction"):
        patch_example(
            example,
            expected_annotation(language_instruction="Unsafe\nInstruction"),
        )


def test_expected_distribution_constants_are_exact_counters() -> None:
    assert CANONICAL_TASK_DISTRIBUTION == Counter(
        {
            "task1_simple_put_object_in_box": 230,
            "task2_complex_put_object_or_beaker_tripod": 291,
            "task3_simple_beaker_tripod": 225,
            "task4_test_tube_rack_insertion": 254,
        }
    )
    assert CANONICAL_SLOT_DISTRIBUTION == Counter(
        {"": 746, "1-2": 154, "1-5": 54, "1-3": 46}
    )


def test_generate_candidate_streams_shards_and_copies_metadata(tmp_path: Path) -> None:
    source = tmp_path / "3.4.0"
    candidate = tmp_path / "3.4.0.annotation_candidate"
    examples = [
        make_example(
            episode_id=index,
            annotation_key=f"annotation-{index}".encode(),
            source_file=f"dataset/sessions/session-{index}".encode(),
        )
        for index in range(3)
    ]
    shard_names = make_version(source, [[examples[0], examples[1]], [examples[2]]])

    report = generate_candidate_version(
        source,
        candidate,
        annotations_for(3),
        expected_count=3,
        require_canonical_distributions=False,
    )

    assert report.examples == 3
    assert report.source_shard_filenames == shard_names
    assert report.candidate_shard_filenames == shard_names
    assert report.source_shard_lengths == [2, 1]
    assert report.candidate_shard_lengths == [2, 1]
    assert (candidate / "features.json").read_bytes() == (source / "features.json").read_bytes()
    assert (candidate / "LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    assert not (candidate / "dataset_statistics_train.json").exists()
    source_info = json.loads((source / "dataset_info.json").read_text())
    candidate_info = json.loads((candidate / "dataset_info.json").read_text())
    candidate_train = candidate_info["splits"][0]
    assert candidate_train["shardLengths"] == ["2", "1"]
    assert candidate_train["numBytes"] == str(
        sum((candidate / name).stat().st_size for name in shard_names)
    )
    candidate_train["shardLengths"] = source_info["splits"][0]["shardLengths"]
    candidate_train["numBytes"] = source_info["splits"][0]["numBytes"]
    assert candidate_info == source_info


def test_generate_refuses_nonempty_candidate_without_touching_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    candidate.mkdir()
    (candidate / "keep").write_text("occupied", encoding="utf-8")
    source_before = tree_digest(source)

    with pytest.raises(ValueError, match="non-empty"):
        generate_candidate_version(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert tree_digest(source) == source_before
    assert (candidate / "keep").read_text() == "occupied"


def test_generate_requires_a_distinct_sibling_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested_candidate = source / "candidate"
    make_version(source, [[make_example()]])
    source_before = tree_digest(source)

    with pytest.raises(ValueError, match="sibling"):
        generate_candidate_version(
            source,
            nested_candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert tree_digest(source) == source_before


@pytest.mark.parametrize(
    "examples,annotations,message",
    [
        ([make_example(annotation_key=b"wrong")], {0: expected_annotation()}, "annotation_key"),
        (
            [make_example(), make_example()],
            {0: expected_annotation()},
            "duplicate episode IDs",
        ),
    ],
)
def test_generation_failures_never_modify_source(
    tmp_path: Path,
    examples: list[tf.train.Example],
    annotations: dict[int, ExpectedAnnotation],
    message: str,
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [examples])
    source_before = tree_digest(source)

    with pytest.raises(ValueError, match=message):
        generate_candidate_version(
            source,
            candidate,
            annotations,
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert tree_digest(source) == source_before


def test_verify_reports_coverage_distributions_and_immutable_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    report_path = tmp_path / "report.json"
    first = make_example()
    first.features.feature["steps/pointcloud/xyz"].float_list.value.extend([1.0, 2.0, 3.0])
    second = make_example(
        episode_id=1,
        annotation_key=b"annotation-1",
        source_file=b"dataset/sessions/session-1",
    )
    make_version(source, [[first], [second]])
    annotations = annotations_for(2)

    generate_candidate_version(
        source,
        candidate,
        annotations,
        expected_count=2,
        require_canonical_distributions=False,
    )
    report = verify_version_pair(
        source,
        candidate,
        annotations,
        expected_count=2,
        require_canonical_distributions=False,
        report_path=report_path,
    )

    assert report.examples == 2
    assert report.valid_episode_ids == 2
    assert report.status == "success"
    assert report.unique_episode_ids == 2
    assert report.missing_ids == []
    assert report.extra_ids == []
    assert report.duplicate_ids == []
    assert report.identity_mismatches == []
    assert report.unexpected_changed_features == []
    assert report.semantic_mismatch_count == 0
    assert report.dataset_info_mismatch_count == 0
    assert report.task_counts == {"task-0": 1, "task-1": 1}
    assert report.instruction_counts == {
        "Canonical instruction 0.": 1,
        "Canonical instruction 1.": 1,
    }
    assert report.slot_counts == {"": 1, "1-2": 1}
    assert report.source_immutable_digest == report.candidate_immutable_digest
    json_report = json.loads(report_path.read_text())
    assert json_report["immutable_digest_identity"] is True
    assert json_report["identity_mismatches"] == 0
    assert json_report["identity_mismatch_details"] == []
    assert json_report["unexpected_changed_features"] == 0
    assert json_report["unexpected_changed_feature_details"] == []
    assert json_report["semantic_mismatches"] == 0
    assert json_report["semantic_mismatch_details"] == []
    assert json_report["dataset_info_mismatches"] == 0
    assert json_report["dataset_info_mismatch_details"] == []


def test_early_verification_failure_removes_stale_success_report(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    report_path = tmp_path / "report.json"
    make_version(source, [[make_example()]])
    generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )
    report_path.write_text('{"status": "success"}', encoding="utf-8")
    next(candidate.glob("*.tfrecord-*")).rename(
        candidate / "wrong.tfrecord-00000-of-00001"
    )

    with pytest.raises(ValueError, match="shard filenames"):
        verify_version_pair(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
            report_path=report_path,
        )

    assert not report_path.exists()


@pytest.mark.parametrize("field,bad_value", [("numBytes", "1"), ("shardLengths", ["9"])])
def test_verify_rejects_inaccurate_candidate_dataset_info(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )
    info_path = candidate / "dataset_info.json"
    info = json.loads(info_path.read_text())
    info["splits"][0][field] = bad_value
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_info"):
        verify_version_pair(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )


@pytest.mark.parametrize("mutation", ["changed", "added", "removed"])
def test_verify_detects_every_kind_of_immutable_feature_drift(
    tmp_path: Path, mutation: str
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    example = make_example()
    example.features.feature["steps/pointcloud/xyz"].float_list.value.extend([1.0, 2.0])
    make_version(source, [[example]])
    generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )
    shard = next(candidate.glob("*.tfrecord-*"))
    changed = read_shard(shard)[0]
    if mutation == "changed":
        changed.features.feature["version_specific/contact_memory"].bytes_list.value[0] = b"drift"
    elif mutation == "added":
        changed.features.feature["steps/pointcloud/new_metadata"].bytes_list.value.append(b"drift")
    else:
        del changed.features.feature["steps/pointcloud/xyz"]
    write_shard(shard, [changed])

    with pytest.raises(ValueError, match="immutable feature drift") as error:
        verify_version_pair(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert "pointcloud" in str(error.value) or "contact_memory" in str(error.value)


def test_failed_reports_expose_semantic_and_dataset_info_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    report_path = tmp_path / "report.json"
    make_version(source, [[make_example()]])
    generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )
    shard = next(candidate.glob("*.tfrecord-*"))
    changed = read_shard(shard)[0]
    changed.features.feature[TASK_ID_FEATURE].bytes_list.value[0] = b"wrong-task"
    write_shard(shard, [changed])

    with pytest.raises(ValueError, match="semantic feature mismatch"):
        verify_version_pair(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
            report_path=report_path,
        )

    failed = json.loads(report_path.read_text())
    assert failed["status"] == "failed"
    assert failed["semantic_mismatches"] > 0
    assert failed["semantic_mismatch_details"]
    assert failed["dataset_info_mismatches"] > 0
    assert failed["dataset_info_mismatch_details"]


def test_report_counts_scanned_examples_separately_from_valid_ids(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    report_path = tmp_path / "report.json"
    malformed = make_example()
    del malformed.features.feature[EPISODE_ID_FEATURE]
    make_version(source, [[malformed]])
    make_version(candidate, [[malformed]])
    (candidate / "dataset_statistics_train.json").unlink()
    sync_candidate_dataset_info(candidate)

    with pytest.raises(ValueError, match="coverage mismatch"):
        verify_version_pair(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
            report_path=report_path,
        )

    failed = json.loads(report_path.read_text())
    assert failed["examples"] == 1
    assert failed["valid_episode_ids"] == 0
    assert failed["unique_episode_ids"] == 0
    assert failed["status"] == "failed"


def test_end_to_end_generation_enforces_canonical_distributions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    annotations = canonical_annotations()
    examples = [
        make_example(
            episode_id=episode_id,
            annotation_key=f"annotation-{episode_id}".encode(),
            source_file=f"dataset/sessions/session-{episode_id}".encode(),
            n_steps=1,
        )
        for episode_id in range(1000)
    ]
    make_version(source, [examples])

    report = generate_candidate_version(source, candidate, annotations)

    assert report.examples == 1000
    assert report.unique_episode_ids == 1000
    assert Counter(report.task_counts) == CANONICAL_TASK_DISTRIBUTION
    assert Counter(report.instruction_counts) == CANONICAL_INSTRUCTION_DISTRIBUTION
    assert Counter(report.slot_counts) == CANONICAL_SLOT_DISTRIBUTION


@pytest.mark.parametrize(
    "feature,kind",
    [
        (EPISODE_ID_FEATURE, "int64"),
        (ANNOTATION_KEY_FEATURE, "bytes"),
        (SOURCE_FILE_FEATURE, "bytes"),
    ],
)
def test_verifier_rejects_candidate_triple_identity_mismatch(
    tmp_path: Path, feature: str, kind: str
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )
    source_before = tree_digest(source)
    shard = next(candidate.glob("*.tfrecord-*"))
    changed = read_shard(shard)[0]
    target = changed.features.feature[feature]
    if kind == "int64":
        target.int64_list.value[0] = 9
    else:
        target.bytes_list.value[0] = b"wrong"
    write_shard(shard, [changed])
    sync_candidate_dataset_info(candidate)

    with pytest.raises(ValueError, match="identity mismatches"):
        verify_version_pair(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert tree_digest(source) == source_before


@pytest.mark.parametrize("coverage", ["missing", "extra"])
def test_generation_rejects_missing_and_extra_coverage(
    tmp_path: Path, coverage: str
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    if coverage == "missing":
        make_version(source, [[make_example()]])
        annotations = annotations_for(2)
        expected_count = 2
    else:
        make_version(
            source,
            [[make_example(episode_id=1, annotation_key=b"annotation-1")]],
        )
        annotations = {0: expected_annotation()}
        expected_count = 1
    source_before = tree_digest(source)

    with pytest.raises(ValueError, match=coverage):
        generate_candidate_version(
            source,
            candidate,
            annotations,
            expected_count=expected_count,
            require_canonical_distributions=False,
        )

    assert tree_digest(source) == source_before


@pytest.mark.parametrize("coverage", ["missing", "extra"])
def test_verifier_rejects_missing_and_extra_coverage(
    tmp_path: Path, coverage: str
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    if coverage == "missing":
        make_version(source, [[make_example()]])
        make_version(candidate, [[make_example()]])
        annotations = annotations_for(2)
        expected_count = 2
    else:
        extra = make_example(
            episode_id=1,
            annotation_key=b"annotation-1",
            source_file=b"dataset/sessions/session-1",
        )
        make_version(source, [[extra]])
        make_version(candidate, [[extra]])
        annotations = {0: expected_annotation()}
        expected_count = 1
    (candidate / "dataset_statistics_train.json").unlink()
    sync_candidate_dataset_info(candidate)

    with pytest.raises(ValueError, match="coverage mismatch"):
        verify_version_pair(
            source,
            candidate,
            annotations,
            expected_count=expected_count,
            require_canonical_distributions=False,
        )


@pytest.mark.parametrize("drift", ["filename", "count", "per_shard_examples"])
def test_verifier_rejects_shard_structure_drift(tmp_path: Path, drift: str) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    second = make_example(
        episode_id=1,
        annotation_key=b"annotation-1",
        source_file=b"dataset/sessions/session-1",
    )
    make_version(source, [[make_example(), second]])
    generate_candidate_version(
        source,
        candidate,
        annotations_for(2),
        expected_count=2,
        require_canonical_distributions=False,
    )
    shard = next(candidate.glob("*.tfrecord-*"))
    if drift == "filename":
        shard.rename(candidate / "renamed-train.tfrecord-00000-of-00001")
    elif drift == "count":
        write_shard(candidate / "extra.tfrecord-00001-of-00002", [])
    else:
        write_shard(shard, read_shard(shard)[:1])
        sync_candidate_dataset_info(candidate)

    message = "shard filenames" if drift != "per_shard_examples" else "per-shard example count"
    with pytest.raises(ValueError, match=message):
        verify_version_pair(
            source,
            candidate,
            annotations_for(2),
            expected_count=2,
            require_canonical_distributions=False,
        )


def test_generation_copies_recursive_metadata_and_omits_nested_statistics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    nested = source / "version_specific" / "schema"
    nested.mkdir(parents=True)
    (nested / "features.json").write_text("nested features", encoding="utf-8")
    (nested / "dataset_statistics_train.json").write_text("stale", encoding="utf-8")

    generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )

    assert (candidate / "version_specific/schema/features.json").read_text() == "nested features"
    assert not (candidate / "version_specific/schema/dataset_statistics_train.json").exists()


def test_metadata_copy_failure_leaves_source_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    source_before = tree_digest(source)

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(
        "scripts.patch_rlds_semantic_annotations.shutil.copy2", fail_copy
    )
    with pytest.raises(OSError, match="copy failed"):
        generate_candidate_version(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert tree_digest(source) == source_before


def test_dataset_info_failure_leaves_source_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    (source / "dataset_info.json").write_text("not json", encoding="utf-8")
    source_before = tree_digest(source)

    with pytest.raises(json.JSONDecodeError):
        generate_candidate_version(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert tree_digest(source) == source_before


def test_candidate_is_atomically_published_only_after_temp_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    original_verify = patch_module.verify_version_pair
    observed_temp: list[Path] = []

    def observe_verify(
        source_version: str | Path,
        candidate_version: str | Path,
        annotations: dict[int, ExpectedAnnotation],
        **kwargs: object,
    ) -> object:
        temporary = Path(candidate_version)
        observed_temp.append(temporary)
        assert temporary != candidate
        assert ".candidate.tmp-" in temporary.name
        assert not candidate.exists()
        return original_verify(source_version, candidate_version, annotations, **kwargs)

    monkeypatch.setattr(patch_module, "verify_version_pair", observe_verify)

    patch_module.generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )

    assert candidate.is_dir()
    assert observed_temp and not observed_temp[0].exists()
    assert list(tmp_path.glob(".candidate.tmp-*")) == []


def test_failed_temp_verification_is_clean_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])

    def fail_verify(*_args: object, **_kwargs: object) -> object:
        raise ValueError("verification failed")

    monkeypatch.setattr(patch_module, "verify_version_pair", fail_verify)

    with pytest.raises(ValueError, match="verification failed"):
        patch_module.generate_candidate_version(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert not candidate.exists()
    assert list(tmp_path.glob(".candidate.tmp-*")) == []


@pytest.mark.parametrize("dangling", [False, True])
def test_generation_rejects_symlink_candidate(
    tmp_path: Path, dangling: bool
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    target = tmp_path / "missing" if dangling else tmp_path / "empty-target"
    if not dangling:
        target.mkdir()
    candidate.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        generate_candidate_version(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert candidate.is_symlink()
    assert list(tmp_path.glob(".candidate.tmp-*")) == []


def test_generation_rejects_source_metadata_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    external = tmp_path / "external-features.json"
    external.write_text("external", encoding="utf-8")
    (source / "linked_features.json").symlink_to(external)
    source_before = tree_digest(source)

    with pytest.raises(ValueError, match="source metadata symlink"):
        generate_candidate_version(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )

    assert tree_digest(source) == source_before
    assert not candidate.exists()


@pytest.mark.parametrize("version_name", ["source", "candidate"])
def test_verifier_rejects_metadata_symlinks(
    tmp_path: Path, version_name: str
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    make_version(source, [[make_example()]])
    generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )
    external = tmp_path / f"external-{version_name}.json"
    external.write_text("external", encoding="utf-8")
    ((source if version_name == "source" else candidate) / "linked.json").symlink_to(
        external
    )

    with pytest.raises(ValueError, match=f"{version_name} metadata symlink"):
        verify_version_pair(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
        )


def test_verifier_patches_only_one_example_object_per_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    example = make_example(n_steps=1)
    example.features.feature["version_specific/large_opaque"].bytes_list.value.append(
        b"x" * (4 * 1024 * 1024)
    )
    make_version(source, [[example]])
    generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )
    calls = 0
    original_patch = patch_module.patch_example

    def count_patch(example_value: object, expected: ExpectedAnnotation) -> object:
        nonlocal calls
        calls += 1
        return original_patch(example_value, expected)

    monkeypatch.setattr(patch_module, "patch_example", count_patch)

    verify_version_pair(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
    )

    assert calls == 1


def test_failed_report_records_complete_immutable_digest_coverage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    report_path = tmp_path / "report.json"
    malformed = make_example()
    del malformed.features.feature[EPISODE_ID_FEATURE]
    make_version(source, [[malformed]])
    make_version(candidate, [[malformed]])
    (candidate / "dataset_statistics_train.json").unlink()
    sync_candidate_dataset_info(candidate)

    with pytest.raises(ValueError, match="coverage mismatch"):
        verify_version_pair(
            source,
            candidate,
            {0: expected_annotation()},
            expected_count=1,
            require_canonical_distributions=False,
            report_path=report_path,
        )

    failed = json.loads(report_path.read_text())
    assert failed["examples"] == 1
    assert failed["immutable_digest_records"] == 1
    assert failed["immutable_digest_complete"] is True
    assert failed["immutable_digest_identity"] is True


def test_load_manifest_can_disable_canonical_validation_for_generic_counts(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    row = manifest_row()
    manifest.write_text(
        ",".join(row) + "\n" + ",".join(row.values()) + "\n",
        encoding="utf-8",
    )

    loaded = load_manifest_annotations(
        manifest,
        expected_count=1,
        require_canonical_distributions=False,
    )

    assert list(loaded) == [0]


def make_publish_swap(root: Path, name: str) -> object:
    source = root / name
    candidate = root / f"{name}.candidate"
    backup = root / f"{name}.backup"
    report = root / f"{name}.report.json"
    source.mkdir()
    candidate.mkdir()
    (source / "payload").write_text(f"old-{name}", encoding="utf-8")
    (candidate / "payload").write_text(f"new-{name}", encoding="utf-8")
    report.write_text(
        json.dumps(
            {
                "status": "success",
                "examples": 1000,
                "unique_episode_ids": 1000,
                "missing_ids": [],
                "extra_ids": [],
                "duplicate_ids": [],
                "identity_mismatches": 0,
                "unexpected_changed_features": 0,
                "semantic_mismatches": 0,
                "dataset_info_mismatches": 0,
                "immutable_digest_identity": True,
                "source_version": str(source.resolve()),
                "candidate_version": str(candidate.resolve()),
                "source_identity": patch_module._path_identity(source),
                "candidate_identity": patch_module._path_identity(candidate),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return patch_module.SwapSpec(source, candidate, backup, report)


def make_publish_set(tmp_path: Path) -> tuple[list[object], Path]:
    return (
        [make_publish_swap(tmp_path, "3.0.0"), make_publish_swap(tmp_path, "3.4.0")],
        tmp_path / ".publication.json",
    )


def test_publish_preflight_requires_exactly_two_verified_versions(
    tmp_path: Path,
) -> None:
    swap = make_publish_swap(tmp_path, "3.0.0")

    with pytest.raises(ValueError, match="exactly two"):
        patch_module.publish_versions([swap], tmp_path / "journal.json", preflight_only=True)


def test_publish_preflight_is_read_only_and_reports_exact_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    swaps, journal = make_publish_set(tmp_path)

    patch_module.publish_versions(swaps, journal, preflight_only=True)

    assert capsys.readouterr().out == "PREFLIGHT_OK versions=2\n"
    assert all(s.source.is_dir() and s.candidate.is_dir() for s in swaps)
    assert all(not s.backup.exists() for s in swaps)
    assert not journal.exists()


@pytest.mark.parametrize(
    "problem", ["backup", "symlink", "alias", "report_identity", "candidate_changed"]
)
def test_publish_preflight_rejects_unsafe_paths_and_stale_reports(
    tmp_path: Path, problem: str
) -> None:
    swaps, journal = make_publish_set(tmp_path)
    swap = swaps[0]
    if problem == "backup":
        swap.backup.mkdir()
    elif problem == "symlink":
        (swap.candidate / "linked").symlink_to(swap.source / "payload")
    elif problem == "alias":
        swaps[1] = patch_module.SwapSpec(
            swap.source, swaps[1].candidate, swaps[1].backup, swaps[1].report
        )
    else:
        if problem == "report_identity":
            data = json.loads(swap.report.read_text(encoding="utf-8"))
            data["candidate_identity"]["inode"] += 1
            swap.report.write_text(json.dumps(data), encoding="utf-8")
        else:
            (swap.candidate / "payload").write_text("changed after verify", encoding="utf-8")

    with pytest.raises(ValueError):
        patch_module.publish_versions(swaps, journal, preflight_only=True)


def test_publish_commits_both_versions_and_durable_identity_journal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    swaps, journal = make_publish_set(tmp_path)
    old = [patch_module._path_identity(s.source) for s in swaps]
    new = [patch_module._path_identity(s.candidate) for s in swaps]

    patch_module.publish_versions(swaps, journal)

    assert capsys.readouterr().out == "PUBLISH_COMMITTED versions=2\n"
    data = json.loads(journal.read_text(encoding="utf-8"))
    assert data["state"] == "committed"
    assert [patch_module._path_identity(s.source) for s in swaps] == new
    assert [patch_module._path_identity(s.backup) for s in swaps] == old
    assert all(item["post_swap_identity_verified"] for item in data["swaps"])


def test_caught_publication_failure_rolls_back_in_reverse_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swaps, journal = make_publish_set(tmp_path)
    original_rename = patch_module._rename
    calls: list[tuple[Path, Path]] = []

    def fail_fourth(source: Path, destination: Path) -> None:
        calls.append((source, destination))
        if len(calls) == 4:
            raise OSError("rename failed")
        original_rename(source, destination)

    monkeypatch.setattr(patch_module, "_rename", fail_fourth)
    with pytest.raises(OSError, match="rename failed"):
        patch_module.publish_versions(swaps, journal)

    assert all(s.source.is_dir() and s.candidate.is_dir() for s in swaps)
    assert all(not s.backup.exists() for s in swaps)
    assert calls[4:] == [
        (swaps[1].backup, swaps[1].source),
        (swaps[0].source, swaps[0].candidate),
        (swaps[0].backup, swaps[0].source),
    ]
    assert json.loads(journal.read_text())["state"] == "rolled_back"


CRASH_CHECKPOINTS = [
    "journal_initialized",
    "before_source_rename_0",
    "after_source_rename_0",
    "source_rename_complete_0",
    "before_candidate_rename_0",
    "after_candidate_rename_0",
    "candidate_rename_complete_0",
    "before_source_rename_1",
    "after_source_rename_1",
    "source_rename_complete_1",
    "before_candidate_rename_1",
    "after_candidate_rename_1",
    "candidate_rename_complete_1",
    "before_commit",
]


@pytest.mark.parametrize("checkpoint", CRASH_CHECKPOINTS)
def test_recovery_restores_preflight_state_after_every_crash_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    swaps, journal = make_publish_set(tmp_path)

    def crash(label: str) -> None:
        if label == checkpoint:
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(patch_module, "_publication_checkpoint", crash)
    with pytest.raises(KeyboardInterrupt, match=checkpoint):
        patch_module.publish_versions(swaps, journal)
    monkeypatch.setattr(patch_module, "_publication_checkpoint", lambda _label: None)

    assert patch_module.recover_publication(journal) is True
    assert all(s.source.is_dir() and s.candidate.is_dir() for s in swaps)
    assert all(not s.backup.exists() for s in swaps)
    assert json.loads(journal.read_text())["state"] == "rolled_back"


def test_recovery_uses_recorded_identity_not_path_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swaps, journal = make_publish_set(tmp_path)

    def crash(label: str) -> None:
        if label == "source_rename_complete_0":
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(patch_module, "_publication_checkpoint", crash)
    with pytest.raises(KeyboardInterrupt):
        patch_module.publish_versions(swaps, journal)
    swaps[0].source.mkdir()
    (swaps[0].source / "intruder").write_text("wrong", encoding="utf-8")
    monkeypatch.setattr(patch_module, "_publication_checkpoint", lambda _label: None)

    with pytest.raises(RuntimeError, match="identity"):
        patch_module.recover_publication(journal)


def test_committed_journal_is_idempotent_but_rejects_postswap_tampering(
    tmp_path: Path
) -> None:
    swaps, journal = make_publish_set(tmp_path)
    patch_module.publish_versions(swaps, journal)

    assert patch_module.recover_publication(journal) is False
    moved = tmp_path / "moved"
    os.rename(swaps[0].source, moved)
    swaps[0].source.mkdir()

    with pytest.raises(RuntimeError, match="committed.*identity"):
        patch_module.recover_publication(journal)


def test_crash_after_committed_journal_preserves_and_validates_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swaps, journal = make_publish_set(tmp_path)

    def crash(label: str) -> None:
        if label == "committed":
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(patch_module, "_publication_checkpoint", crash)
    with pytest.raises(KeyboardInterrupt, match="committed"):
        patch_module.publish_versions(swaps, journal)

    assert patch_module.recover_publication(journal) is False
    assert all(s.source.is_dir() and s.backup.is_dir() for s in swaps)
    assert all(not s.candidate.exists() for s in swaps)


def test_cli_patch_and_verify_hard_require_canonical_1000(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("ignored", encoding="utf-8")
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(patch_module, "load_manifest_annotations", lambda *_a, **_k: {})
    monkeypatch.setattr(
        patch_module,
        "generate_candidate_version",
        lambda *_a, **kwargs: calls.append(("patch", kwargs)),
    )
    monkeypatch.setattr(
        patch_module,
        "verify_version_pair",
        lambda *_a, **kwargs: calls.append(("verify", kwargs)),
    )

    assert patch_module.main(["patch", "--manifest", str(manifest), "--source-version", "s", "--candidate-version", "c", "--report", "r"]) == 0
    assert patch_module.main(["verify", "--manifest", str(manifest), "--source-version", "s", "--candidate-version", "c", "--report", "r"]) == 0
    assert all(call[1]["expected_count"] == 1000 for call in calls)
    assert all(call[1]["require_canonical_distributions"] is True for call in calls)


def test_generation_report_names_final_candidate_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    report = tmp_path / "report.json"
    make_version(source, [[make_example()]])

    generate_candidate_version(
        source,
        candidate,
        {0: expected_annotation()},
        expected_count=1,
        require_canonical_distributions=False,
        report_path=report,
    )

    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert evidence["candidate_version"] == str(candidate.resolve())
    assert evidence["candidate_identity"] == patch_module._path_identity(candidate)


def test_publication_fsyncs_journal_file_and_all_rename_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swaps, journal = make_publish_set(tmp_path)
    fsynced_directories: list[Path] = []
    file_fsyncs = 0
    original_fsync = os.fsync
    original_directory_fsync = patch_module._fsync_directory

    def observe_fsync(descriptor: int) -> None:
        nonlocal file_fsyncs
        file_fsyncs += 1
        original_fsync(descriptor)

    def observe_directory(path: Path) -> None:
        fsynced_directories.append(path.resolve())
        original_directory_fsync(path)

    monkeypatch.setattr(patch_module.os, "fsync", observe_fsync)
    monkeypatch.setattr(patch_module, "_fsync_directory", observe_directory)

    patch_module.publish_versions(swaps, journal)

    assert file_fsyncs >= 11
    assert journal.parent.resolve() in fsynced_directories
    assert all(s.source.parent.resolve() in fsynced_directories for s in swaps)


def test_rollback_failure_is_a_hard_error_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swaps, journal = make_publish_set(tmp_path)
    original_rename = patch_module._rename
    calls = 0

    def fail_forward_and_rollback(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls in (4, 5):
            raise OSError(f"rename failure {calls}")
        original_rename(source, destination)

    monkeypatch.setattr(patch_module, "_rename", fail_forward_and_rollback)

    with pytest.raises(RuntimeError, match="rollback failed"):
        patch_module.publish_versions(swaps, journal)
    evidence = json.loads(journal.read_text(encoding="utf-8"))
    assert evidence["state"] == "rollback_failed"
    assert "rename failure 5" in evidence["rollback_error"]


def test_process_termination_during_rollback_remains_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swaps, journal = make_publish_set(tmp_path)
    original_rename = patch_module._rename
    calls = 0

    def fail_then_crash(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("trigger rollback")
        if calls == 5:
            raise KeyboardInterrupt("power loss during rollback")
        original_rename(source, destination)

    monkeypatch.setattr(patch_module, "_rename", fail_then_crash)
    with pytest.raises(KeyboardInterrupt, match="power loss"):
        patch_module.publish_versions(swaps, journal)
    monkeypatch.setattr(patch_module, "_rename", original_rename)

    assert patch_module.recover_publication(journal) is True
    assert all(s.source.is_dir() and s.candidate.is_dir() for s in swaps)
    assert all(not s.backup.exists() for s in swaps)


def test_verify_discovers_and_recovers_adjacent_uncommitted_journal_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swaps, original_journal = make_publish_set(tmp_path)
    journal = tmp_path / ".semantic_annotation_patch_transaction_test.json"

    def crash(label: str) -> None:
        if label == "source_rename_complete_0":
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(patch_module, "_publication_checkpoint", crash)
    with pytest.raises(KeyboardInterrupt):
        patch_module.publish_versions(swaps, journal)
    assert not original_journal.exists()
    monkeypatch.setattr(patch_module, "_publication_checkpoint", lambda _label: None)

    def load_after_recovery(*_args: object, **_kwargs: object) -> dict[int, object]:
        assert swaps[0].source.is_dir() and swaps[0].candidate.is_dir()
        assert not swaps[0].backup.exists()
        return {}

    monkeypatch.setattr(patch_module, "load_manifest_annotations", load_after_recovery)
    monkeypatch.setattr(patch_module, "verify_version_pair", lambda *_a, **_k: None)

    assert patch_module.main(
        [
            "verify",
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--source-version",
            str(swaps[0].source),
            "--candidate-version",
            str(swaps[0].candidate),
            "--report",
            str(tmp_path / "new-report.json"),
        ]
    ) == 0


ROLLBACK_CRASH_CHECKPOINTS = [
    f"{phase}_inverse_{kind}_{index}"
    for index in (1, 0)
    for kind in ("candidate", "source")
    for phase in ("before", "after", "after_parents_fsync", "complete")
] + ["between_rollback_swaps_1", "before_rolled_back", "after_rolled_back"]


def leave_fully_swapped_uncommitted(
    swaps: list[object], journal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(label: str) -> None:
        if label == "before_commit":
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(patch_module, "_publication_checkpoint", crash)
    with pytest.raises(KeyboardInterrupt, match="before_commit"):
        patch_module.publish_versions(swaps, journal)
    monkeypatch.setattr(patch_module, "_publication_checkpoint", lambda _label: None)


@pytest.mark.parametrize("checkpoint", ROLLBACK_CRASH_CHECKPOINTS)
def test_recovery_is_idempotent_at_every_rollback_crash_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    swaps, journal = make_publish_set(tmp_path)
    leave_fully_swapped_uncommitted(swaps, journal, monkeypatch)

    def crash(label: str) -> None:
        if label == checkpoint:
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(patch_module, "_publication_checkpoint", crash)
    with pytest.raises(KeyboardInterrupt, match=checkpoint):
        patch_module.recover_publication(journal)
    monkeypatch.setattr(patch_module, "_publication_checkpoint", lambda _label: None)

    patch_module.recover_publication(journal)
    assert all(s.source.is_dir() and s.candidate.is_dir() for s in swaps)
    assert all(not s.backup.exists() for s in swaps)
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "rolled_back"


def swap_cli_value(swap: object) -> str:
    return ",".join(str(path) for path in (swap.source, swap.candidate, swap.backup, swap.report))


def test_publish_cli_parses_repeatable_swaps_and_exact_commit_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    swaps, journal = make_publish_set(tmp_path)

    result = patch_module.main(
        [
            "publish",
            "--journal",
            str(journal),
            "--swap",
            swap_cli_value(swaps[0]),
            "--swap",
            swap_cli_value(swaps[1]),
        ]
    )

    assert result == 0
    assert capsys.readouterr().out == "PUBLISH_COMMITTED versions=2\n"
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "committed"


def test_publish_cli_preflight_only_has_exact_output_and_no_journal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    swaps, journal = make_publish_set(tmp_path)

    result = patch_module.main(
        [
            "publish",
            "--preflight-only",
            "--journal",
            str(journal),
            "--swap",
            swap_cli_value(swaps[0]),
            "--swap",
            swap_cli_value(swaps[1]),
        ]
    )

    assert result == 0
    assert capsys.readouterr().out == "PREFLIGHT_OK versions=2\n"
    assert not journal.exists()


def test_publish_cli_requires_journal_and_two_complete_swaps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    swaps, journal = make_publish_set(tmp_path)
    with pytest.raises(SystemExit) as missing_journal:
        patch_module.main(["publish", "--swap", swap_cli_value(swaps[0])])
    assert missing_journal.value.code == 2
    assert "--journal" in capsys.readouterr().err

    with pytest.raises(ValueError, match="exactly two"):
        patch_module.main(
            [
                "publish",
                "--journal",
                str(journal),
                "--swap",
                swap_cli_value(swaps[0]),
            ]
        )

    with pytest.raises(SystemExit) as malformed_swap:
        patch_module.main(
            ["publish", "--journal", str(journal), "--swap", "only,two"]
        )
    assert malformed_swap.value.code == 2
    assert "--swap requires source,candidate,backup,report" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["patch", "verify"])
@pytest.mark.parametrize("journal_location", ["source", "report"])
def test_patch_and_verify_discover_adjacent_journal_before_manifest_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    journal_location: str,
) -> None:
    swaps, _unused = make_publish_set(tmp_path)
    report_parent = tmp_path / "logs"
    report_parent.mkdir()
    journal_parent = tmp_path if journal_location == "source" else report_parent
    journal = journal_parent / ".semantic_annotation_patch_transaction_old.json"

    def crash(label: str) -> None:
        if label == "source_rename_complete_0":
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(patch_module, "_publication_checkpoint", crash)
    with pytest.raises(KeyboardInterrupt):
        patch_module.publish_versions(swaps, journal)
    monkeypatch.setattr(patch_module, "_publication_checkpoint", lambda _label: None)

    def load_after_recovery(*_args: object, **_kwargs: object) -> dict[int, object]:
        assert swaps[0].source.is_dir() and swaps[0].candidate.is_dir()
        return {}

    monkeypatch.setattr(patch_module, "load_manifest_annotations", load_after_recovery)
    monkeypatch.setattr(patch_module, "generate_candidate_version", lambda *_a, **_k: None)
    monkeypatch.setattr(patch_module, "verify_version_pair", lambda *_a, **_k: None)
    argv = [
        command,
        "--manifest",
        str(tmp_path / "manifest.csv"),
        "--source-version",
        str(swaps[0].source),
        "--candidate-version",
        str(swaps[0].candidate),
        "--report",
        str(report_parent / "new-report.json"),
    ]

    assert patch_module.main(argv) == 0


def test_publish_cli_recovers_older_adjacent_journal_before_new_explicit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    swaps, _unused = make_publish_set(tmp_path)
    old_journal = tmp_path / ".semantic_annotation_patch_transaction_old.json"
    new_journal = tmp_path / "new-explicit-publication.json"

    def crash(label: str) -> None:
        if label == "source_rename_complete_0":
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(patch_module, "_publication_checkpoint", crash)
    with pytest.raises(KeyboardInterrupt):
        patch_module.publish_versions(swaps, old_journal)
    monkeypatch.setattr(patch_module, "_publication_checkpoint", lambda _label: None)

    assert patch_module.main(
        [
            "publish",
            "--journal",
            str(new_journal),
            "--swap",
            swap_cli_value(swaps[0]),
            "--swap",
            swap_cli_value(swaps[1]),
        ]
    ) == 0
    assert capsys.readouterr().out == "PUBLISH_COMMITTED versions=2\n"
    assert json.loads(old_journal.read_text(encoding="utf-8"))["state"] == "rolled_back"
    assert json.loads(new_journal.read_text(encoding="utf-8"))["state"] == "committed"


def test_recover_cli_discovers_adjacent_standard_journal_and_reports_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    swaps, _unused = make_publish_set(tmp_path)
    standard = tmp_path / ".semantic_annotation_patch_transaction_old.json"
    explicit = tmp_path / "requested-but-absent.json"

    def crash(label: str) -> None:
        if label == "source_rename_complete_0":
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(patch_module, "_publication_checkpoint", crash)
    with pytest.raises(KeyboardInterrupt):
        patch_module.publish_versions(swaps, standard)
    monkeypatch.setattr(patch_module, "_publication_checkpoint", lambda _label: None)

    assert patch_module.main(["recover", "--journal", str(explicit)]) == 0
    assert capsys.readouterr().out == "RECOVERY_COMPLETE\n"
    assert json.loads(standard.read_text(encoding="utf-8"))["state"] == "rolled_back"
    assert not explicit.exists()


@pytest.mark.parametrize("path_kind", ["candidate", "report"])
def test_preflight_rejects_symlinked_path_ancestors(
    tmp_path: Path, path_kind: str
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    swaps = [make_publish_swap(tmp_path, "3.0.0"), make_publish_swap(tmp_path, "3.4.0")]
    original = swaps[0]
    if path_kind == "candidate":
        target = real / "nested" / "candidate"
        target.parent.mkdir()
        original.candidate.rename(target)
        candidate = linked / "nested" / "candidate"
        data = json.loads(original.report.read_text(encoding="utf-8"))
        data["candidate_version"] = str(candidate.resolve())
        data["candidate_identity"] = patch_module._path_identity(candidate)
        original.report.write_text(json.dumps(data), encoding="utf-8")
        swaps[0] = patch_module.SwapSpec(
            original.source, candidate, original.backup, original.report
        )
    else:
        target = real / "nested" / "report.json"
        target.parent.mkdir()
        original.report.rename(target)
        swaps[0] = patch_module.SwapSpec(
            original.source,
            original.candidate,
            original.backup,
            linked / "nested" / "report.json",
        )

    with pytest.raises(ValueError, match="symlink.*ancestor"):
        patch_module.publish_versions(
            swaps, tmp_path / ".semantic_annotation_patch_transaction_new.json", preflight_only=True
        )
