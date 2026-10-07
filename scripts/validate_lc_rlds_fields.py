#!/usr/bin/env python3
"""Validate TransVTLA RLDS fields required by Latent Correction."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DATASET_NAME = "transvtla_annotated_rlds"
OBS_PREFIX = "steps/observation"
REQUIRED_FIELDS = {
    "state": ("float", 7),
    "point_cloud": ("float", None),
    "next_point_cloud": ("float", None),
    "contact_memory": ("float", 18),
    "valid_contact_mask": ("int", 6),
    "contact_type": ("int", 6),
    "contact_reliability": ("float", 6),
    "contact_time": ("int", 6),
    "gripper_xyz": ("float", 3),
    "current_timestep": ("int", 1),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def feature_values(feature: Any, kind: str) -> np.ndarray:
    if kind == "float":
        return np.asarray(feature.float_list.value, dtype=np.float32)
    if kind == "int":
        return np.asarray(feature.int64_list.value, dtype=np.int64)
    raise ValueError(kind)


def stats(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def schema_fields(version_dir: Path) -> set[str]:
    features = read_json(version_dir / "features.json")
    obs = (
        features["featuresDict"]["features"]["steps"]["sequence"]["feature"]["featuresDict"]
        ["features"]["observation"]["featuresDict"]["features"]
    )
    return set(obs.keys())


def validate_example(features: Any, episode_ref: str, summary: dict[str, Any]) -> None:
    missing = []
    for field in REQUIRED_FIELDS:
        key = f"{OBS_PREFIX}/{field}"
        if key not in features:
            missing.append(field)
    if missing:
        summary["errors"].append({"episode": episode_ref, "error": "missing_fields", "fields": missing})
        return

    state = feature_values(features[f"{OBS_PREFIX}/state"], "float")
    if state.size % 7 != 0:
        summary["errors"].append({"episode": episode_ref, "error": "state_length", "value": int(state.size)})
        return
    n = state.size // 7
    state = state.reshape(n, 7)

    gripper = feature_values(features[f"{OBS_PREFIX}/gripper_xyz"], "float")
    current = feature_values(features[f"{OBS_PREFIX}/current_timestep"], "int")
    contact_memory = feature_values(features[f"{OBS_PREFIX}/contact_memory"], "float")
    valid = feature_values(features[f"{OBS_PREFIX}/valid_contact_mask"], "int")
    contact_type = feature_values(features[f"{OBS_PREFIX}/contact_type"], "int")
    reliability = feature_values(features[f"{OBS_PREFIX}/contact_reliability"], "float")
    contact_time = feature_values(features[f"{OBS_PREFIX}/contact_time"], "int")
    pc = feature_values(features[f"{OBS_PREFIX}/point_cloud"], "float")
    next_pc = feature_values(features[f"{OBS_PREFIX}/next_point_cloud"], "float")

    expected_lengths = {
        "gripper_xyz": n * 3,
        "current_timestep": n,
        "contact_memory": n * 18,
        "valid_contact_mask": n * 6,
        "contact_type": n * 6,
        "contact_reliability": n * 6,
        "contact_time": n * 6,
    }
    actual_lengths = {
        "gripper_xyz": gripper.size,
        "current_timestep": current.size,
        "contact_memory": contact_memory.size,
        "valid_contact_mask": valid.size,
        "contact_type": contact_type.size,
        "contact_reliability": reliability.size,
        "contact_time": contact_time.size,
    }
    bad_lengths = {
        key: [int(actual_lengths[key]), int(expected)]
        for key, expected in expected_lengths.items()
        if actual_lengths[key] != expected
    }
    if bad_lengths:
        summary["errors"].append({"episode": episode_ref, "error": "length_mismatch", "lengths": bad_lengths})
        return

    gripper = gripper.reshape(n, 3)
    current = current.reshape(n)
    contact_memory = contact_memory.reshape(n, 6, 3)
    valid = valid.reshape(n, 6).astype(bool)
    contact_type = contact_type.reshape(n, 6)
    reliability = reliability.reshape(n, 6)
    contact_time = contact_time.reshape(n, 6)

    if not np.array_equal(current, np.arange(n, dtype=np.int64)):
        summary["errors"].append({"episode": episode_ref, "error": "bad_current_timestep"})
    if not np.allclose(gripper, state[:, :3], atol=1e-6):
        summary["errors"].append({"episode": episode_ref, "error": "gripper_xyz_not_state_prefix"})
    if not np.isfinite(gripper).all():
        summary["errors"].append({"episode": episode_ref, "error": "gripper_xyz_nonfinite"})
    if not np.isfinite(contact_memory).all():
        summary["errors"].append({"episode": episode_ref, "error": "contact_memory_nonfinite"})
    if pc.size == 0 or pc.size % (n * 3) != 0:
        summary["errors"].append({"episode": episode_ref, "error": "point_cloud_length", "value": int(pc.size)})
    if next_pc.size != pc.size:
        summary["errors"].append({"episode": episode_ref, "error": "next_point_cloud_length", "point_cloud": int(pc.size), "next": int(next_pc.size)})
    if pc.size and (not np.isfinite(pc).all()):
        summary["errors"].append({"episode": episode_ref, "error": "point_cloud_nonfinite"})
    if not set(np.unique(contact_type).tolist()).issubset({0, 1}):
        summary["errors"].append({"episode": episode_ref, "error": "contact_type_values", "values": np.unique(contact_type).tolist()})
    if ((reliability < -1e-6) | (reliability > 1 + 1e-6)).any():
        summary["errors"].append({"episode": episode_ref, "error": "contact_reliability_range"})
    frame_ids = np.arange(n, dtype=np.int64)[:, None]
    if (valid & (contact_time < 0)).any():
        summary["errors"].append({"episode": episode_ref, "error": "valid_contact_time_negative"})
    if (valid & (contact_time > frame_ids)).any():
        summary["errors"].append({"episode": episode_ref, "error": "future_contact_time"})
    if ((~valid) & (contact_time != -1)).any():
        summary["errors"].append({"episode": episode_ref, "error": "invalid_slot_contact_time_not_minus_one"})
    if (valid[:, :4] & (contact_type[:, :4] != 0)).any():
        summary["errors"].append({"episode": episode_ref, "error": "active_slot_type_mismatch"})
    if (valid[:, 4:6] & (contact_type[:, 4:6] != 1)).any():
        summary["errors"].append({"episode": episode_ref, "error": "grasp_slot_type_mismatch"})

    workspace_gripper = (
        (gripper[:, 0] > -0.2) & (gripper[:, 0] < 0.8)
        & (gripper[:, 1] > -0.5) & (gripper[:, 1] < 0.5)
        & (gripper[:, 2] > -0.1) & (gripper[:, 2] < 0.8)
    )
    if not workspace_gripper.all():
        summary["risk_episodes"].append({"episode": episode_ref, "risk": "gripper_xyz_workspace", "bad_steps": int(np.count_nonzero(~workspace_gripper))})
    if valid.any():
        cm_valid = contact_memory[valid]
        workspace_contact = (
            (cm_valid[:, 0] > -0.2) & (cm_valid[:, 0] < 0.8)
            & (cm_valid[:, 1] > -0.5) & (cm_valid[:, 1] < 0.5)
            & (cm_valid[:, 2] > -0.1) & (cm_valid[:, 2] < 0.8)
        )
        if not workspace_contact.all():
            summary["risk_episodes"].append({"episode": episode_ref, "risk": "contact_memory_workspace", "bad_slots": int(np.count_nonzero(~workspace_contact))})

    zero_pc_ratio = 0.0
    if pc.size:
        pc_arr = pc.reshape(n, -1, 3)
        zero_pc_ratio = float(np.count_nonzero(np.linalg.norm(pc_arr, axis=-1) < 1e-9) / pc_arr.shape[0] / pc_arr.shape[1])
        if zero_pc_ratio > 0.1:
            summary["risk_episodes"].append({"episode": episode_ref, "risk": "point_cloud_many_zero_points", "zero_ratio": zero_pc_ratio})

    summary["num_examples"] += 1
    summary["num_steps"] += n
    summary["contact_valid_slots"] += int(valid.sum())
    summary["frames_with_contact"] += int(valid.any(axis=1).sum())
    summary["episodes_with_contact"] += int(valid.any())
    summary["zero_point_ratios"].append(zero_pc_ratio)
    summary["gripper_xyz_min"] = np.minimum(summary["gripper_xyz_min"], gripper.min(axis=0)).tolist()
    summary["gripper_xyz_max"] = np.maximum(summary["gripper_xyz_max"], gripper.max(axis=0)).tolist()
    if valid.any():
        cm_valid = contact_memory[valid]
        summary["contact_memory_min"] = np.minimum(summary["contact_memory_min"], cm_valid.min(axis=0)).tolist()
        summary["contact_memory_max"] = np.maximum(summary["contact_memory_max"], cm_valid.max(axis=0)).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version-dir",
        type=Path,
        default=Path("/media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.3.0_candidate"),
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--report", type=Path, default=Path("/media/files/TransVTLA-RealDataCollect/reports/lc_rlds_fields_validation.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import tensorflow as tf

    if not args.version_dir.is_dir():
        raise SystemExit(f"Version directory not found: {args.version_dir}")
    schema = schema_fields(args.version_dir)
    required_schema = set(REQUIRED_FIELDS.keys())
    missing_schema = sorted(required_schema - schema)
    shards = sorted(args.version_dir.glob(f"{DATASET_NAME}-train.tfrecord-*"))
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    summary: dict[str, Any] = {
        "version_dir": str(args.version_dir),
        "num_shards": len(shards),
        "schema_missing_fields": missing_schema,
        "num_examples": 0,
        "num_steps": 0,
        "episodes_with_contact": 0,
        "contact_valid_slots": 0,
        "frames_with_contact": 0,
        "errors": [],
        "risk_episodes": [],
        "zero_point_ratios": [],
        "gripper_xyz_min": [math.inf, math.inf, math.inf],
        "gripper_xyz_max": [-math.inf, -math.inf, -math.inf],
        "contact_memory_min": [math.inf, math.inf, math.inf],
        "contact_memory_max": [-math.inf, -math.inf, -math.inf],
    }

    for shard_idx, shard in enumerate(shards):
        if shard.stat().st_size == 0:
            summary["errors"].append({"episode": str(shard), "error": "empty_shard"})
            continue
        for rec_idx, raw in enumerate(tf.compat.v1.io.tf_record_iterator(str(shard))):
            ex = tf.train.Example.FromString(raw)
            validate_example(ex.features.feature, f"{shard.name}:{rec_idx}", summary)
        if shard_idx < 3 or (shard_idx + 1) % 64 == 0 or shard_idx == len(shards) - 1:
            print(f"[{shard_idx + 1:04d}/{len(shards):04d}] examples={summary['num_examples']} errors={len(summary['errors'])}")

    summary["zero_point_ratio_stats"] = stats(summary.pop("zero_point_ratios"))
    summary["contact_frame_ratio"] = summary["frames_with_contact"] / summary["num_steps"] if summary["num_steps"] else 0.0
    summary["valid_slot_density"] = summary["contact_valid_slots"] / (summary["num_steps"] * 6) if summary["num_steps"] else 0.0
    summary["status"] = "PASS" if not summary["schema_missing_fields"] and not summary["errors"] else "FAIL"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={summary['status']}")
    print(f"examples={summary['num_examples']} steps={summary['num_steps']} errors={len(summary['errors'])} risks={len(summary['risk_episodes'])}")
    print(f"report={args.report}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
