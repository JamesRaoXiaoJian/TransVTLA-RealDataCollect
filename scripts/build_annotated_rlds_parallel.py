#!/usr/bin/env python3
"""Build one RLDS/TFDS dataset through independent worker builders.

Each worker owns a disjoint manifest partition and its own TFDS directory.
After all workers finish, this launcher rewrites only episode_id while merging
their TFRecord shards into one conventional TFDS version directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "transvtla_annotated_rlds"


def partition_rows(rows: list[dict[str, str]], workers: int) -> list[list[dict[str, str]]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    active_workers = min(workers, len(rows)) if rows else 1
    base, remainder = divmod(len(rows), active_workers)
    partitions: list[list[dict[str, str]]] = []
    start = 0
    for index in range(active_workers):
        size = base + (1 if index < remainder else 0)
        partitions.append(rows[start : start + size])
        start += size
    return partitions


def limit_rows(rows: list[dict[str, str]], max_episodes: int | None) -> list[dict[str, str]]:
    return rows if max_episodes is None else rows[:max_episodes]


def selected_rows(manifest: Path, data_source: str) -> tuple[list[str], list[dict[str, str]]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {manifest}")
        rows = [
            row for row in reader
            if row.get("data_source", "").strip() == data_source
            and row.get("usable_for_aligned_training", "").strip().lower() in {"1", "true", "yes", "y"}
            and row.get("needs_review", "").strip().lower() not in {"1", "true", "yes", "y"}
        ]
        return list(reader.fieldnames), rows


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_workers(worker_roots: list[Path], destination: Path, version: str) -> None:
    import tensorflow as tf

    destination_version = destination / DATASET_NAME / version
    destination_version.mkdir(parents=True, exist_ok=False)
    worker_versions = [root / DATASET_NAME / version for root in worker_roots]
    template = worker_versions[0]
    for source in template.iterdir():
        if source.is_file() and not source.name.startswith(f"{DATASET_NAME}-train.tfrecord") and not source.name.startswith("dataset_statistics_"):
            shutil.copy2(source, destination_version / source.name)

    source_shards = [
        shard for worker_version in worker_versions
        for shard in sorted(worker_version.glob(f"{DATASET_NAME}-train.tfrecord-*-of-*"))
    ]
    total_shards = len(source_shards)
    shard_lengths: list[str] = []
    total_bytes = 0
    episode_id = 0
    for shard_index, source in enumerate(source_shards):
        target = destination_version / f"{DATASET_NAME}-train.tfrecord-{shard_index:05d}-of-{total_shards:05d}"
        examples = 0
        with tf.io.TFRecordWriter(str(target)) as writer:
            for raw in tf.compat.v1.io.tf_record_iterator(str(source)):
                example = tf.train.Example.FromString(raw)
                example.features.feature["episode_metadata/episode_id"].int64_list.value[:] = [episode_id]
                writer.write(example.SerializeToString())
                examples += 1
                episode_id += 1
        shard_lengths.append(str(examples))
        total_bytes += target.stat().st_size

    info_path = destination_version / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    train_split = next(split for split in info["splits"] if split["name"] == "train")
    train_split["numBytes"] = str(total_bytes)
    train_split["shardLengths"] = shard_lengths
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None, help="Small integration-test subset.")
    parser.add_argument("--data-source", default="real", choices=["real", "sim"])
    parser.add_argument("--version", default="3.0.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory already exists and is non-empty: {output_dir}")
    fields, rows = selected_rows(manifest, args.data_source)
    if not rows:
        raise SystemExit("no usable non-review rows selected")
    rows = limit_rows(rows, args.max_episodes)
    partitions = partition_rows(rows, args.workers)
    workspace = output_dir.with_name(f"{output_dir.name}_parallel_workspace")
    if workspace.exists():
        raise SystemExit(f"parallel workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    worker_roots: list[Path] = []
    processes: list[subprocess.Popen[str]] = []
    try:
        for index, part in enumerate(partitions):
            part_manifest = workspace / f"manifest_{index:02d}.csv"
            write_manifest(part_manifest, fields, part)
            worker_root = workspace / f"worker_{index:02d}"
            worker_roots.append(worker_root)
            command = [
                sys.executable, str(REPO_ROOT / "scripts" / "build_annotated_rlds.py"),
                "--manifest", str(part_manifest), "--output-dir", str(worker_root),
                "--data-source", args.data_source, "--build", "--version", args.version,
            ]
            print(f"[worker {index + 1}/{len(partitions)}] episodes={len(part)}", flush=True)
            processes.append(subprocess.Popen(command, cwd=REPO_ROOT, text=True))
        failed = [process.wait() for process in processes]
        if any(code != 0 for code in failed):
            raise SystemExit(f"worker failures: {failed}")
        output_dir.mkdir(parents=True, exist_ok=False)
        merge_workers(worker_roots, output_dir, args.version)
        print(f"done: {output_dir / DATASET_NAME / args.version}", flush=True)
        return 0
    finally:
        if any(process.poll() is None for process in processes):
            for process in processes:
                if process.poll() is None:
                    process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
