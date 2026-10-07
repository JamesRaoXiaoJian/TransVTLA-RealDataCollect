#!/usr/bin/env python3
"""Fast in-place-style TFRecord rewrite for TransVTLA gripper normalization.

This script does not modify the source directory. It creates a new TFDS builder
directory with the same dataset name/version and rewrites only:

  steps/observation/state[6] -> clip(raw_gripper_pos / 1000, 0, 1)
  steps/action[6]            -> next normalized gripper position

All image/depth/tactile/text/metadata fields are copied through unchanged.
The output intentionally omits old dataset_statistics_*.json cache files so
OpenVLA recomputes statistics for the corrected dataset on first use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


DATASET_SUBDIR = Path("transvtla_annotated_rlds/3.0.0")
ACTION_KEY = "steps/action"
STATE_KEY = "steps/observation/state"


def normalize_gripper(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(values.astype(np.float32) / float(scale), 0.0, 1.0)


def rewrite_example(raw: bytes, scale: float) -> tuple[bytes, int, np.ndarray, np.ndarray]:
    example = tf.train.Example.FromString(raw)
    features = example.features.feature
    if ACTION_KEY not in features or STATE_KEY not in features:
        raise ValueError(f"Missing required keys: {ACTION_KEY}, {STATE_KEY}")

    state_values = features[STATE_KEY].float_list.value
    action_values = features[ACTION_KEY].float_list.value
    if len(state_values) % 7 != 0 or len(action_values) % 7 != 0:
        raise ValueError(f"State/action length is not divisible by 7: {len(state_values)}, {len(action_values)}")
    n_state = len(state_values) // 7
    n_action = len(action_values) // 7
    if n_state != n_action:
        raise ValueError(f"State/action step count mismatch: {n_state}, {n_action}")

    raw_gripper = np.asarray([state_values[i * 7 + 6] for i in range(n_state)], dtype=np.float32)
    normalized = normalize_gripper(raw_gripper, scale)

    for i in range(n_state):
        state_values[i * 7 + 6] = float(normalized[i])
        target_i = min(i + 1, n_state - 1)
        action_values[i * 7 + 6] = float(normalized[target_i])

    return example.SerializeToString(), n_state, raw_gripper, normalized


def copy_metadata(source_version_dir: Path, output_version_dir: Path) -> None:
    output_version_dir.mkdir(parents=True, exist_ok=True)
    for src in source_version_dir.iterdir():
        if not src.is_file():
            continue
        name = src.name
        if name.startswith("transvtla_annotated_rlds-train.tfrecord"):
            continue
        if name.startswith("dataset_statistics_"):
            continue
        shutil.copy2(src, output_version_dir / name)


def write_summary(path: Path, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def unique_sibling(path: Path, suffix: str) -> Path:
    stamp = f"{int(time.time())}_{os.getpid()}"
    candidate = path.with_name(f"{path.name}.{suffix}_{stamp}")
    idx = 0
    while candidate.exists():
        idx += 1
        candidate = path.with_name(f"{path.name}.{suffix}_{stamp}_{idx}")
    return candidate


def prepare_work_root(output_root: Path, overwrite: bool) -> tuple[Path, list[Path]]:
    """Return a fresh temporary output root and any old dirs to clean later."""
    trash_roots: list[Path] = []
    if output_root.exists():
        has_contents = output_root.is_file() or any(output_root.iterdir())
        if has_contents and not overwrite:
            raise SystemExit(f"Output root already exists and is non-empty: {output_root}")
        if not overwrite:
            work_root = unique_sibling(output_root, "tmp")
            return work_root, trash_roots

        trash_root = unique_sibling(output_root, "delete")
        output_root.rename(trash_root)
        trash_roots.append(trash_root)

    work_root = unique_sibling(output_root, "tmp")
    if work_root.exists():
        raise SystemExit(f"Temporary output root unexpectedly exists: {work_root}")
    return work_root, trash_roots


def publish_work_root(work_root: Path, output_root: Path, trash_roots: list[Path]) -> None:
    if output_root.exists():
        late_trash = unique_sibling(output_root, "delete")
        output_root.rename(late_trash)
        trash_roots.append(late_trash)
    work_root.rename(output_root)
    for trash_root in trash_roots:
        try:
            shutil.rmtree(trash_root)
        except OSError as exc:
            print(f"warning: could not fully remove old output directory {trash_root}: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_strict"),
        help="Source TFDS data root containing transvtla_annotated_rlds/3.0.0.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_gripper01_strict"),
        help="New TFDS data root to create.",
    )
    parser.add_argument("--gripper-scale", type=float, default=1000.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None, help="Debug option.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_version_dir = args.source_root / DATASET_SUBDIR
    if not source_version_dir.is_dir():
        raise SystemExit(f"Source TFDS version directory not found: {source_version_dir}")

    shards = sorted(source_version_dir.glob("transvtla_annotated_rlds-train.tfrecord-*"))
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    if not shards:
        raise SystemExit(f"No TFRecord shards found in {source_version_dir}")

    work_root, trash_roots = prepare_work_root(args.output_root, args.overwrite)
    output_version_dir = work_root / DATASET_SUBDIR
    final_version_dir = args.output_root / DATASET_SUBDIR
    copy_metadata(source_version_dir, output_version_dir)

    total_examples = 0
    total_steps = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    norm_min = float("inf")
    norm_max = float("-inf")
    abs_gt_1_before = 0

    print(f"source={source_version_dir}")
    print(f"work_output={output_version_dir}")
    print(f"final_output={final_version_dir}")
    print(f"shards={len(shards)} gripper_scale={args.gripper_scale}")

    for shard_idx, src in enumerate(shards, start=1):
        dst = output_version_dir / src.name
        shard_examples = 0
        with tf.io.TFRecordWriter(str(dst)) as writer:
            for raw in tf.compat.v1.io.tf_record_iterator(str(src)):
                rewritten, n_steps, raw_gripper, normalized = rewrite_example(raw, args.gripper_scale)
                writer.write(rewritten)
                shard_examples += 1
                total_examples += 1
                total_steps += n_steps
                raw_min = min(raw_min, float(raw_gripper.min()))
                raw_max = max(raw_max, float(raw_gripper.max()))
                norm_min = min(norm_min, float(normalized.min()))
                norm_max = max(norm_max, float(normalized.max()))
                abs_gt_1_before += int(np.count_nonzero(np.abs(raw_gripper) > 1.0))
        if shard_idx == 1 or shard_idx % 25 == 0 or shard_idx == len(shards):
            print(
                f"[{shard_idx:04d}/{len(shards):04d}] {src.name} "
                f"examples={shard_examples} total_examples={total_examples} total_steps={total_steps}",
                flush=True,
            )

    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "version_dir": str(final_version_dir),
        "gripper_scale": args.gripper_scale,
        "transform": {
            "observation.state[6]": "clip(raw_gripper_pos / 1000, 0, 1)",
            "action[6]": "next normalized gripper position, terminal step repeats current normalized gripper",
        },
        "num_shards": len(shards),
        "num_examples": total_examples,
        "num_steps": total_steps,
        "raw_gripper_min": raw_min,
        "raw_gripper_max": raw_max,
        "normalized_gripper_min": norm_min,
        "normalized_gripper_max": norm_max,
        "raw_state6_abs_gt_1_count_before": abs_gt_1_before,
        "note": "Old dataset_statistics_*.json files were intentionally not copied.",
    }
    write_summary(output_version_dir / "gripper_normalization_summary.json", summary)
    publish_work_root(work_root, args.output_root, trash_roots)
    print(f"done: {final_version_dir}")
    print(f"summary: {final_version_dir / 'gripper_normalization_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
