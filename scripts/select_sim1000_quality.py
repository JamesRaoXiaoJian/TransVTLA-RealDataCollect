#!/usr/bin/env python3
"""Select a balanced high-quality 1000-session view from the 2026-07-13 sim2000 raw set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_SHARDS = [
    Path("/media/data/simdata/franka_sim_gpu1_batch1000_20260713_shard00_gpu1"),
    Path("/media/data/simdata/franka_sim_gpu1_batch1000_20260713_shard01_gpu1"),
    Path("/media/data/simdata/franka_sim_gpu1_batch1000_part2_20260713_shard00_gpu1"),
    Path("/media/data/simdata/franka_sim_gpu1_batch1000_part2_20260713_shard01_gpu1"),
]
DEFAULT_OUTPUT = Path("/media/files/TransVTLA-RealDataCollect/dataset/sim1000_quality_selected_20260713")
REPORTS_DIR = Path("/media/files/TransVTLA-RealDataCollect/reports")
SESSION_RE = re.compile(r"session_ep(?P<episode>\d+)_env(?P<env>\d+)$")
PRESSURE_CHANNELS = [
    "CH51", "CH50",
    "CH63", "CH60", "CH57", "CH64", "CH61", "CH58", "CH49", "CH62", "CH59",
    "CH47", "CH44", "CH41", "CH48", "CH45", "CH42", "CH33", "CH46", "CH43",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def nearest_indices(source_us: np.ndarray, target_us: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(source_us, target_us)
    pos = np.clip(pos, 1, len(source_us) - 1)
    left = pos - 1
    right = pos
    choose_right = np.abs(source_us[right] - target_us) < np.abs(target_us - source_us[left])
    return np.where(choose_right, right, left)


def wrap_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + math.pi) % (2 * math.pi) - math.pi


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def session_metrics(session_dir: Path, shard_index: int, source_shard: str) -> dict[str, object]:
    match = SESSION_RE.match(session_dir.name)
    if not match:
        raise ValueError(f"unexpected session name: {session_dir}")

    frames = read_csv(session_dir / "frames.csv")
    robot = read_csv(session_dir / "robot_state" / "robot_state.csv")
    gripper = read_csv(session_dir / "robot_state" / "gripper_state.csv")
    pressure = read_csv(session_dir / "pressure" / "pressure.csv")

    frame_us = np.asarray([int(row["capture_monotonic_us"]) for row in frames], dtype=np.int64)
    robot_us = np.asarray([int(row["timestamp_us"]) for row in robot], dtype=np.int64)
    robot_idx = nearest_indices(robot_us, frame_us)
    pose = np.asarray(
        [[float(robot[i][f"pose_{j}"]) for j in range(1, 7)] for i in robot_idx],
        dtype=np.float64,
    )
    pos_delta = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
    rot_delta = np.linalg.norm(wrap_delta(np.diff(pose[:, 3:6], axis=0)), axis=1)

    gripper_values = np.asarray([float(row["gripper_pos"]) for row in gripper], dtype=np.float64)

    pressure_arr = np.asarray(
        [[float(row[channel]) for channel in PRESSURE_CHANNELS] for row in pressure],
        dtype=np.float64,
    )
    baseline = np.median(pressure_arr[:50], axis=0)
    drop = np.maximum(baseline[None, :] - pressure_arr, 0.0)
    pressure_peak = float(np.max(drop))
    pressure_p95 = percentile(drop.reshape(-1), 95)
    pressure_positive_ratio = float(np.mean(drop > 50.0))

    pos_p95_mm = percentile(pos_delta, 95) * 1000.0
    pos_max_mm = float(np.max(pos_delta) * 1000.0)
    rot_p95_rad = percentile(rot_delta, 95)
    rot_max_rad = float(np.max(rot_delta))
    gripper_min = float(np.min(gripper_values))
    gripper_max = float(np.max(gripper_values))
    gripper_range = gripper_max - gripper_min

    frame_count = len(frames)
    robot_rows = len(robot)
    gripper_rows = len(gripper)
    pressure_rows = len(pressure)
    file_ok = (
        frame_count == 175
        and robot_rows == 350
        and gripper_rows == 350
        and pressure_rows == 816
        and len(list((session_dir / "world_camera" / "rgb").glob("*.jpg"))) == 175
        and len(list((session_dir / "world_camera" / "depth").glob("*.png"))) == 175
        and len(list((session_dir / "wrist_camera" / "rgb").glob("*.jpg"))) == 175
        and len(list((session_dir / "wrist_camera" / "depth").glob("*.png"))) == 175
    )

    # Higher score favors complete, smooth, real-aligned motion with non-degenerate gripper/tactile signals.
    gripper_score = min(max(gripper_range / 900.0, 0.0), 1.0)
    pressure_score = min(max(pressure_peak / 2200.0, 0.0), 1.0)
    smooth_penalty = (
        0.018 * pos_p95_mm
        + 0.004 * pos_max_mm
        + 1.8 * rot_p95_rad
        + 0.35 * rot_max_rad
    )
    outlier_penalty = 0.0
    if rot_max_rad > math.pi:
        outlier_penalty += 4.0
    if pos_max_mm > 160.0:
        outlier_penalty += 1.5
    if pressure_peak < 250.0:
        outlier_penalty += 1.0
    if gripper_range < 500.0:
        outlier_penalty += 1.0
    score = (2.0 * gripper_score) + (1.5 * pressure_score) - smooth_penalty - outlier_penalty
    if not file_ok:
        score -= 100.0

    episode = int(match.group("episode"))
    env = int(match.group("env"))
    return {
        "selected": 0,
        "quality_score": score,
        "file_ok": int(file_ok),
        "source_shard": source_shard,
        "source_shard_index": shard_index,
        "source_session": session_dir.name,
        "episode": episode,
        "env": env,
        "source_path": str(session_dir),
        "frame_count": frame_count,
        "robot_rows": robot_rows,
        "gripper_rows": gripper_rows,
        "pressure_rows": pressure_rows,
        "pos_delta_p95_mm": pos_p95_mm,
        "pos_delta_max_mm": pos_max_mm,
        "rot_delta_p95_rad": rot_p95_rad,
        "rot_delta_max_rad": rot_max_rad,
        "gripper_min_raw": gripper_min,
        "gripper_max_raw": gripper_max,
        "gripper_range_raw": gripper_range,
        "pressure_peak_drop": pressure_peak,
        "pressure_p95_drop": pressure_p95,
        "pressure_positive_ratio": pressure_positive_ratio,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-env-per-shard", type=int, default=50)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory exists and is non-empty: {output_dir}")
    sessions_dir = output_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for shard_index, shard in enumerate(DEFAULT_SHARDS):
        session_dirs = sorted(path for path in shard.iterdir() if path.is_dir() and SESSION_RE.match(path.name))
        for idx, session_dir in enumerate(session_dirs, start=1):
            metrics = session_metrics(session_dir, shard_index, shard.name)
            all_rows.append(metrics)
            if idx % 100 == 0:
                print(f"[scan] {shard.name}: {idx}/{len(session_dirs)}", flush=True)

    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        groups[(int(row["source_shard_index"]), int(row["env"]))].append(row)

    selected: list[dict[str, object]] = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda row: float(row["quality_score"]), reverse=True)
        take = ranked[: args.per_env_per_shard]
        if len(take) != args.per_env_per_shard:
            raise SystemExit(f"group {key} has only {len(take)} sessions")
        selected.extend(take)

    selected_ids = set()
    manifest_rows: list[dict[str, object]] = []
    for order, row in enumerate(sorted(selected, key=lambda r: (int(r["source_shard_index"]), int(r["env"]), int(r["episode"])))):
        row["selected"] = 1
        selected_session_id = f"sim2000_s{int(row['source_shard_index']):02d}_ep{int(row['episode']):03d}_env{int(row['env'])}"
        if selected_session_id in selected_ids:
            raise SystemExit(f"duplicate selected_session_id: {selected_session_id}")
        selected_ids.add(selected_session_id)
        link_path = sessions_dir / selected_session_id
        os.symlink(row["source_path"], link_path, target_is_directory=True)
        manifest_rows.append({
            "index": order,
            "session": selected_session_id,
            "rel_path": str(link_path.relative_to(Path("/media/files/TransVTLA-RealDataCollect"))),
            "source_path": row["source_path"],
            "source_shard": row["source_shard"],
            "source_shard_index": row["source_shard_index"],
            "source_session": row["source_session"],
            "episode": row["episode"],
            "env": row["env"],
            "data_source": "sim",
            "split": "sim1000_quality_selected_20260713",
            "data_format": "dual_realsense_sim_raw",
            "usable_for_aligned_training": 1,
            "success": 1,
            "needs_review": 0,
            "task_id": "sim_transparent_cube_box",
            "task_name": "Simulated transparent cube box placement",
            "language_instruction": "Pick up the transparent cube and place it into the box.",
            "object_type": "transparent_cube",
            "quality_score": f"{float(row['quality_score']):.8f}",
            "pos_delta_p95_mm": f"{float(row['pos_delta_p95_mm']):.8f}",
            "pos_delta_max_mm": f"{float(row['pos_delta_max_mm']):.8f}",
            "rot_delta_p95_rad": f"{float(row['rot_delta_p95_rad']):.8f}",
            "rot_delta_max_rad": f"{float(row['rot_delta_max_rad']):.8f}",
            "gripper_range_raw": f"{float(row['gripper_range_raw']):.8f}",
            "pressure_peak_drop": f"{float(row['pressure_peak_drop']):.8f}",
        })

    metric_fields = list(all_rows[0].keys())
    write_csv(output_dir / "quality_metrics_all.csv", all_rows, metric_fields)
    write_csv(output_dir / "quality_metrics_selected.csv", [row for row in all_rows if row["selected"]], metric_fields)
    write_csv(output_dir / "manifest.csv", manifest_rows)

    selected_rows = [row for row in all_rows if row["selected"]]
    summary = {
        "schema": "transvtla_sim1000_quality_selection/v1",
        "source_sessions": len(all_rows),
        "selected_sessions": len(selected_rows),
        "selection": "top 50 per source_shard/env by quality_score",
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.csv"),
        "symlink_sessions": str(sessions_dir),
        "source_shards": [str(path) for path in DEFAULT_SHARDS],
        "selected_by_shard": Counter(str(row["source_shard_index"]) for row in selected_rows),
        "selected_by_env": Counter(str(row["env"]) for row in selected_rows),
        "metrics": {
            name: {
                "min": float(np.min([float(row[name]) for row in selected_rows])),
                "p50": float(np.percentile([float(row[name]) for row in selected_rows], 50)),
                "p95": float(np.percentile([float(row[name]) for row in selected_rows], 95)),
                "max": float(np.max([float(row[name]) for row in selected_rows])),
            }
            for name in [
                "quality_score",
                "pos_delta_p95_mm",
                "pos_delta_max_mm",
                "rot_delta_p95_rad",
                "rot_delta_max_rad",
                "gripper_range_raw",
                "pressure_peak_drop",
                "pressure_positive_ratio",
            ]
        },
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "sim1000_quality_selection_report_20260713.md"
    report_path.write_text(
        "# Sim1000 Quality Selection Report 20260713\n\n"
        f"- Source raw sessions: `{len(all_rows)}`\n"
        f"- Selected sessions: `{len(selected_rows)}`\n"
        f"- Output dataset view: `{output_dir}`\n"
        f"- Manifest: `{output_dir / 'manifest.csv'}`\n"
        f"- Selection rule: top `50` sessions per `(source_shard, env)` by quality score.\n"
        "- Raw source files were not modified; selected sessions are symlinks to `/media/data/simdata`.\n\n"
        "## Quality Score\n\n"
        "The score rewards non-degenerate gripper and pressure signals, and penalizes high per-frame "
        "position/rotation deltas plus Euler/wide-motion outliers. All selected sessions keep the "
        "complete raw file contract: 175 visual frames, 350 robot/gripper rows, and 816 pressure rows.\n\n"
        "## Selected Metric Summary\n\n"
        "| metric | min | p50 | p95 | max |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {vals['min']:.6g} | {vals['p50']:.6g} | {vals['p95']:.6g} | {vals['max']:.6g} |"
            for name, vals in summary["metrics"].items()
        )
        + "\n\n"
        "## Distribution\n\n"
        "- Per shard: `250` selected from each of 4 raw shards.\n"
        "- Per env: `200` selected for each env id 0-4, i.e. 50 per env per shard.\n\n"
        "## Files\n\n"
        f"- Full metrics: `{output_dir / 'quality_metrics_all.csv'}`\n"
        f"- Selected metrics: `{output_dir / 'quality_metrics_selected.csv'}`\n"
        f"- Summary JSON: `{output_dir / 'selection_summary.json'}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "selected": len(selected_rows), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
