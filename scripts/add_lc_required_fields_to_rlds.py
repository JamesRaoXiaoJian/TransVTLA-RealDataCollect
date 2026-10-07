#!/usr/bin/env python3
"""Add Latent-Correction required per-step fields to a TransVTLA RLDS version.

The source version is left untouched. This script rewrites TFRecord shards into
a new version directory and appends:

  steps/observation/gripper_xyz       float32 [3]
  steps/observation/current_timestep  int64   scalar

gripper_xyz follows the current TransVTLA/MLA convention used by existing
transforms: state[:3]. In this real dataset, state is built from robot
pose_1..pose_6 plus gripper position, so state[:3] is the robot base/world
frame end-effector pose position in meters.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


DATASET_NAME = "transvtla_annotated_rlds"
STATE_KEY = "steps/observation/state"
GRIPPER_XYZ_KEY = "steps/observation/gripper_xyz"
CURRENT_TIMESTEP_KEY = "steps/observation/current_timestep"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def unique_sibling(path: Path, suffix: str) -> Path:
    stamp = f"{int(time.time())}_{os.getpid()}"
    candidate = path.with_name(f"{path.name}.{suffix}_{stamp}")
    idx = 0
    while candidate.exists():
        idx += 1
        candidate = path.with_name(f"{path.name}.{suffix}_{stamp}_{idx}")
    return candidate


def prepare_output_dir(output_version_dir: Path, overwrite: bool) -> tuple[Path, list[Path]]:
    trash_dirs: list[Path] = []
    if output_version_dir.exists():
        has_contents = output_version_dir.is_file() or any(output_version_dir.iterdir())
        if has_contents and not overwrite:
            raise SystemExit(f"Output version directory already exists: {output_version_dir}")
        trash = unique_sibling(output_version_dir, "delete")
        output_version_dir.rename(trash)
        trash_dirs.append(trash)
    work_dir = unique_sibling(output_version_dir, "tmp")
    work_dir.mkdir(parents=True)
    return work_dir, trash_dirs


def publish_output_dir(work_dir: Path, output_version_dir: Path, trash_dirs: list[Path]) -> None:
    if output_version_dir.exists():
        trash = unique_sibling(output_version_dir, "delete")
        output_version_dir.rename(trash)
        trash_dirs.append(trash)
    work_dir.rename(output_version_dir)
    for trash in trash_dirs:
        try:
            shutil.rmtree(trash)
        except OSError as exc:
            print(f"warning: could not remove old directory {trash}: {exc}", flush=True)


def copy_metadata(source_version_dir: Path, work_dir: Path) -> None:
    for src in source_version_dir.iterdir():
        if not src.is_file():
            continue
        if src.name.startswith(f"{DATASET_NAME}-train.tfrecord"):
            continue
        if src.name.startswith("dataset_statistics_"):
            continue
        shutil.copy2(src, work_dir / src.name)


def tensor_feature(dtype: str, dims: list[str]) -> dict[str, Any]:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
        "tensor": {
            "dtype": dtype,
            "encoding": "none",
            "shape": {"dimensions": dims},
        },
    }


def update_features_json(work_dir: Path) -> None:
    path = work_dir / "features.json"
    features = read_json(path)
    obs_features = (
        features["featuresDict"]["features"]["steps"]["sequence"]["feature"]["featuresDict"]
        ["features"]["observation"]["featuresDict"]["features"]
    )
    obs_features["gripper_xyz"] = tensor_feature("float32", ["3"])
    obs_features["current_timestep"] = tensor_feature("int64", [])
    write_json(path, features)


def update_dataset_info(work_dir: Path, version: str, shard_lengths: list[int], num_bytes: int) -> None:
    path = work_dir / "dataset_info.json"
    info = read_json(path)
    info["version"] = version
    if info.get("splits"):
        info["splits"][0]["numBytes"] = str(num_bytes)
        info["splits"][0]["shardLengths"] = [str(v) for v in shard_lengths]
    write_json(path, info)


def output_shard_name(index: int, total: int) -> str:
    return f"{DATASET_NAME}-train.tfrecord-{index:05d}-of-{total:05d}"


def clear_and_extend_float(feature: Any, values: np.ndarray) -> None:
    del feature.float_list.value[:]
    feature.float_list.value.extend(np.asarray(values, dtype=np.float32).reshape(-1).tolist())


def clear_and_extend_int64(feature: Any, values: np.ndarray) -> None:
    del feature.int64_list.value[:]
    feature.int64_list.value.extend(np.asarray(values, dtype=np.int64).reshape(-1).tolist())


def add_lc_fields_to_example(raw: bytes, tf: Any) -> tuple[bytes, int, int]:
    example = tf.train.Example.FromString(raw)
    features = example.features.feature
    if STATE_KEY not in features:
        raise ValueError(f"missing required field: {STATE_KEY}")
    state = np.asarray(features[STATE_KEY].float_list.value, dtype=np.float32)
    if state.size % 7 != 0:
        raise ValueError(f"state length must be divisible by 7, got {state.size}")
    n_steps = state.size // 7
    state = state.reshape(n_steps, 7)
    gripper_xyz = state[:, :3].astype(np.float32)
    current_timestep = np.arange(n_steps, dtype=np.int64)
    clear_and_extend_float(features[GRIPPER_XYZ_KEY], gripper_xyz)
    clear_and_extend_int64(features[CURRENT_TIMESTEP_KEY], current_timestep)
    return example.SerializeToString(), n_steps, int(np.count_nonzero(np.isfinite(gripper_xyz).all(axis=1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-version-dir",
        type=Path,
        default=Path(
            "/media/files/TransVTLA-RealDataCollect/dataset/"
            "rlds_real1000_strict/transvtla_annotated_rlds/3.2.0"
        ),
    )
    parser.add_argument("--output-version", default="3.3.0")
    parser.add_argument(
        "--output-version-dir",
        type=Path,
        default=Path(
            "/media/files/TransVTLA-RealDataCollect/dataset/"
            "rlds_real1000_strict/transvtla_annotated_rlds/3.3.0_candidate"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--summary-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import tensorflow as tf

    if not args.source_version_dir.is_dir():
        raise SystemExit(f"Source version directory not found: {args.source_version_dir}")
    shards = sorted(args.source_version_dir.glob(f"{DATASET_NAME}-train.tfrecord-*"))
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    if not shards:
        raise SystemExit(f"No TFRecord shards found in {args.source_version_dir}")

    work_dir, trash_dirs = prepare_output_dir(args.output_version_dir, args.overwrite)
    copy_metadata(args.source_version_dir, work_dir)

    shard_lengths: list[int] = []
    total_steps = 0
    total_examples = 0
    total_output_bytes = 0
    finite_gripper_steps = 0
    started = time.time()

    print(f"source={args.source_version_dir}")
    print(f"work_dir={work_dir}")
    print(f"output={args.output_version_dir}")
    print(f"shards={len(shards)} output_version={args.output_version}")

    for shard_idx, src in enumerate(shards):
        dst = work_dir / output_shard_name(shard_idx, len(shards))
        shard_examples = 0
        shard_steps = 0
        with tf.io.TFRecordWriter(str(dst)) as writer:
            for raw in tf.compat.v1.io.tf_record_iterator(str(src)):
                rewritten, n_steps, finite_steps = add_lc_fields_to_example(raw, tf)
                writer.write(rewritten)
                shard_examples += 1
                shard_steps += n_steps
                total_examples += 1
                total_steps += n_steps
                finite_gripper_steps += finite_steps
        shard_lengths.append(shard_examples)
        total_output_bytes += dst.stat().st_size
        if shard_idx == 0 or (shard_idx + 1) % args.summary_every == 0 or shard_idx == len(shards) - 1:
            print(
                f"[{shard_idx + 1:04d}/{len(shards):04d}] {src.name} "
                f"examples={shard_examples} steps={shard_steps} "
                f"total_examples={total_examples} total_steps={total_steps} "
                f"elapsed_min={(time.time() - started) / 60.0:.1f}",
                flush=True,
            )

    update_features_json(work_dir)
    update_dataset_info(work_dir, args.output_version, shard_lengths, total_output_bytes)
    summary = {
        "source_version_dir": str(args.source_version_dir),
        "output_version_dir": str(args.output_version_dir),
        "output_version": args.output_version,
        "num_shards": len(shards),
        "num_examples": total_examples,
        "num_steps": total_steps,
        "finite_gripper_xyz_steps": finite_gripper_steps,
        "fields_added": {
            "gripper_xyz": ["float32", [3]],
            "current_timestep": ["int64", []],
        },
        "gripper_xyz_rule": "observation/state[:3]",
        "gripper_xyz_frame": "robot_base / TransVTLA world-base",
        "gripper_xyz_unit": "meter",
        "current_timestep_rule": "zero-based visual observation step index inside each episode",
        "notes": [
            "Original source version directory is not modified.",
            "state is built from robot pose_1..pose_6 plus gripper position by build_annotated_rlds.py.",
            "state[:3] follows the existing model-side gripper_xyz convention; it is the robot pose position, not an extra T_flange_gripper offset center.",
        ],
    }
    write_json(work_dir / "lc_required_fields_generation_summary.json", summary)
    publish_output_dir(work_dir, args.output_version_dir, trash_dirs)
    print(f"done: {args.output_version_dir}")
    print(f"summary: {args.output_version_dir / 'lc_required_fields_generation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
