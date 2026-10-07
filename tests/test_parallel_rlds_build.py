"""Tests for the independent-shard RLDS build launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_annotated_rlds_parallel.py"
SPEC = importlib.util.spec_from_file_location("build_annotated_rlds_parallel", SCRIPT)
assert SPEC and SPEC.loader
parallel_build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel_build)


def test_partition_rows_is_balanced_and_preserves_input_order() -> None:
    rows = [{"annotation_key": f"row-{index}"} for index in range(10)]

    partitions = parallel_build.partition_rows(rows, workers=3)

    assert [[row["annotation_key"] for row in part] for part in partitions] == [
        ["row-0", "row-1", "row-2", "row-3"],
        ["row-4", "row-5", "row-6"],
        ["row-7", "row-8", "row-9"],
    ]


def test_partition_rows_rejects_invalid_worker_count() -> None:
    try:
        parallel_build.partition_rows([{"annotation_key": "one"}], workers=0)
    except ValueError as exc:
        assert "workers" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_limit_rows_returns_prefix_without_reordering() -> None:
    rows = [{"annotation_key": f"row-{index}"} for index in range(4)]

    assert parallel_build.limit_rows(rows, max_episodes=2) == rows[:2]
    assert parallel_build.limit_rows(rows, max_episodes=None) == rows
