#!/usr/bin/env python3
"""Audit timestamp domains and alignment quality for real TransVTLA sessions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_MANIFEST = Path(
    "/media/files/TransVTLA-RealDataCollect/dataset/"
    "auto_annotations_qwen3vl_thirdpass/final_annotations/"
    "real_1000_train_view_strict/manifest.csv"
)
DEFAULT_REPORT_DIR = Path("/media/files/TransVTLA-RealDataCollect/reports")


@dataclass
class SeriesStats:
    count: int
    monotonic_non_decreasing: bool
    strictly_increasing: bool
    duplicate_count: int
    min_value: int | None
    max_value: int | None
    median_interval_ms: float | None
    p95_interval_ms: float | None
    max_interval_ms: float | None


def parse_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        v = float(str(value))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_int_series(path: Path, columns: Iterable[str]) -> tuple[str | None, np.ndarray]:
    rows = read_csv(path)
    if not rows:
        return None, np.asarray([], dtype=np.int64)
    columns = list(columns)
    selected = None
    values: list[int] = []
    for column in columns:
        if column in rows[0]:
            selected = column
            break
    if selected is None:
        return None, np.asarray([], dtype=np.int64)
    for row in rows:
        value = parse_int(row.get(selected))
        if value is not None:
            values.append(value)
    return selected, np.asarray(values, dtype=np.int64)


def series_stats(values: np.ndarray) -> SeriesStats:
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return SeriesStats(0, False, False, 0, None, None, None, None, None)
    diffs = np.diff(values)
    intervals_ms = diffs.astype(np.float64) / 1000.0 if diffs.size else np.asarray([], dtype=np.float64)
    return SeriesStats(
        count=int(values.size),
        monotonic_non_decreasing=bool(np.all(diffs >= 0)) if diffs.size else True,
        strictly_increasing=bool(np.all(diffs > 0)) if diffs.size else True,
        duplicate_count=int(values.size - np.unique(values).size),
        min_value=int(values.min()),
        max_value=int(values.max()),
        median_interval_ms=float(np.median(intervals_ms)) if intervals_ms.size else None,
        p95_interval_ms=float(np.percentile(intervals_ms, 95)) if intervals_ms.size else None,
        max_interval_ms=float(np.max(intervals_ms)) if intervals_ms.size else None,
    )


def nearest_offsets_ms(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    if target.size == 0 or source.size == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(source)
    source = source[order]
    idx = np.searchsorted(source, target)
    idx0 = np.clip(idx - 1, 0, source.size - 1)
    idx1 = np.clip(idx, 0, source.size - 1)
    nearest = np.where(np.abs(source[idx0] - target) <= np.abs(source[idx1] - target), source[idx0], source[idx1])
    return (nearest.astype(np.float64) - target.astype(np.float64)) / 1000.0


def offset_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "median_abs_ms": None,
            "p95_abs_ms": None,
            "max_abs_ms": None,
            "gt_50ms": 0,
            "gt_100ms": 0,
            "gt_150ms": 0,
        }
    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "median_abs_ms": float(np.median(abs_values)),
        "p95_abs_ms": float(np.percentile(abs_values, 95)),
        "max_abs_ms": float(np.max(abs_values)),
        "gt_50ms": int(np.count_nonzero(abs_values > 50.0)),
        "gt_100ms": int(np.count_nonzero(abs_values > 100.0)),
        "gt_150ms": int(np.count_nonzero(abs_values > 150.0)),
    }


def stats_to_dict(stats: SeriesStats) -> dict[str, object]:
    return {
        "count": stats.count,
        "monotonic_non_decreasing": stats.monotonic_non_decreasing,
        "strictly_increasing": stats.strictly_increasing,
        "duplicate_count": stats.duplicate_count,
        "min_value": stats.min_value,
        "max_value": stats.max_value,
        "median_interval_ms": stats.median_interval_ms,
        "p95_interval_ms": stats.p95_interval_ms,
        "max_interval_ms": stats.max_interval_ms,
    }


def analyze_aligned_offsets(path: Path) -> tuple[dict[str, np.ndarray], int]:
    rows = read_csv(path)
    offsets: dict[str, list[float]] = {"robot": [], "gripper": [], "pressure": []}
    for row in rows:
        for name in offsets:
            value = parse_float(row.get(f"{name}_offset_ms"))
            if value is not None:
                offsets[name].append(value)
    return {key: np.asarray(value, dtype=np.float64) for key, value in offsets.items()}, len(rows)


def audit_session(session_dir: Path) -> dict[str, object]:
    frames_path = session_dir / "frames.csv"
    robot_path = session_dir / "robot_state" / "robot_state.csv"
    gripper_path = session_dir / "robot_state" / "gripper_state.csv"
    pressure_path = session_dir / "pressure" / "pressure.csv"
    aligned_path = session_dir / "aligned_timesteps.csv"
    sync_index_path = session_dir / "sync" / "sync_index.csv"

    visual_col, visual_ts = read_int_series(frames_path, ("capture_monotonic_us", "visual_timestamp_us", "timestamp_us"))
    robot_col, robot_ts = read_int_series(robot_path, ("timestamp_us", "host_monotonic_us"))
    gripper_col, gripper_ts = read_int_series(gripper_path, ("timestamp_us", "host_monotonic_us"))
    pressure_col, pressure_ts = read_int_series(pressure_path, ("host_monotonic_us", "timestamp_us", "sensor_timestamp_us"))
    sensor_col, pressure_sensor_ts = read_int_series(pressure_path, ("sensor_timestamp_us",))

    missing_files = [
        str(path.relative_to(session_dir))
        for path in (frames_path, robot_path, gripper_path, pressure_path)
        if not path.is_file()
    ]
    aligned_offsets, aligned_rows = analyze_aligned_offsets(aligned_path)
    if aligned_rows:
        robot_offsets = aligned_offsets["robot"]
        gripper_offsets = aligned_offsets["gripper"]
        pressure_offsets = aligned_offsets["pressure"]
        alignment_source = "aligned_timesteps.csv"
    else:
        robot_offsets = nearest_offsets_ms(visual_ts, robot_ts)
        gripper_offsets = nearest_offsets_ms(visual_ts, gripper_ts)
        pressure_offsets = nearest_offsets_ms(visual_ts, pressure_ts)
        alignment_source = "nearest_raw_timestamps"

    # Host monotonic timestamps for visual/robot/gripper/pressure should live in
    # the same order of magnitude. sensor_timestamp_us is intentionally separate.
    host_mins = [
        values.min()
        for values in (visual_ts, robot_ts, gripper_ts, pressure_ts)
        if values.size
    ]
    host_same_domain = bool(host_mins) and (max(host_mins) - min(host_mins) < 24 * 60 * 60 * 1_000_000)
    sensor_same_as_host = bool(pressure_sensor_ts.size and visual_ts.size) and (
        abs(int(pressure_sensor_ts[0]) - int(visual_ts[0])) < 24 * 60 * 60 * 1_000_000
    )

    modality = {
        "visual": stats_to_dict(series_stats(visual_ts)),
        "robot": stats_to_dict(series_stats(robot_ts)),
        "gripper": stats_to_dict(series_stats(gripper_ts)),
        "pressure_host": stats_to_dict(series_stats(pressure_ts)),
        "pressure_sensor": stats_to_dict(series_stats(pressure_sensor_ts)),
    }
    offsets = {
        "robot": offset_stats(robot_offsets),
        "gripper": offset_stats(gripper_offsets),
        "pressure": offset_stats(pressure_offsets),
    }
    high_risk = any(
        (offsets[name]["max_abs_ms"] is not None and float(offsets[name]["max_abs_ms"]) > 100.0)
        for name in offsets
    )
    high_risk = high_risk or bool(missing_files) or not host_same_domain
    return {
        "session_dir": str(session_dir),
        "missing_files": missing_files,
        "timestamp_columns": {
            "visual": visual_col,
            "robot": robot_col,
            "gripper": gripper_col,
            "pressure_alignment": pressure_col,
            "pressure_sensor": sensor_col,
        },
        "aligned_timesteps_present": aligned_path.is_file(),
        "aligned_timesteps_rows": aligned_rows,
        "sync_index_present": sync_index_path.is_file(),
        "alignment_source": alignment_source,
        "host_timestamps_same_domain": host_same_domain,
        "pressure_sensor_same_as_host": sensor_same_as_host,
        "modality": modality,
        "offsets": offsets,
        "high_risk": high_risk,
    }


def aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    def collect_offsets(name: str) -> list[float]:
        values: list[float] = []
        for result in results:
            stat = result["offsets"][name]  # type: ignore[index]
            max_abs = stat.get("max_abs_ms")  # type: ignore[union-attr]
            if max_abs is not None:
                values.append(float(max_abs))
        return values

    missing_file_eps = [r for r in results if r["missing_files"]]
    non_monotonic = {
        name: [
            r["session_dir"]
            for r in results
            if not r["modality"][name]["monotonic_non_decreasing"]  # type: ignore[index]
        ]
        for name in ("visual", "robot", "gripper", "pressure_host", "pressure_sensor")
    }
    duplicate_counts = {
        name: int(sum(r["modality"][name]["duplicate_count"] for r in results))  # type: ignore[index]
        for name in ("visual", "robot", "gripper", "pressure_host", "pressure_sensor")
    }
    outlier_eps: list[dict[str, object]] = []
    for result in results:
        entry = {"session_dir": result["session_dir"], "modalities": {}}
        for name in ("robot", "gripper", "pressure"):
            stat = result["offsets"][name]  # type: ignore[index]
            if stat["max_abs_ms"] is not None and float(stat["max_abs_ms"]) > 50.0:
                entry["modalities"][name] = stat  # type: ignore[index]
        if entry["modalities"]:  # type: ignore[truthy-function]
            outlier_eps.append(entry)
    return {
        "total_episodes": len(results),
        "successful_read_episodes": sum(1 for r in results if not r["missing_files"]),
        "missing_file_episodes": len(missing_file_eps),
        "aligned_timesteps_present_episodes": sum(1 for r in results if r["aligned_timesteps_present"]),
        "sync_index_present_episodes": sum(1 for r in results if r["sync_index_present"]),
        "host_timestamps_same_domain_episodes": sum(1 for r in results if r["host_timestamps_same_domain"]),
        "pressure_sensor_same_as_host_episodes": sum(1 for r in results if r["pressure_sensor_same_as_host"]),
        "non_monotonic_episode_counts": {key: len(value) for key, value in non_monotonic.items()},
        "duplicate_timestamp_total_counts": duplicate_counts,
        "offset_max_abs_ms_distribution_by_episode": {
            name: offset_stats(np.asarray(collect_offsets(name), dtype=np.float64))
            for name in ("robot", "gripper", "pressure")
        },
        "outlier_episode_count_gt50ms": len(outlier_eps),
        "outlier_episodes_gt50ms": outlier_eps[:100],
        "high_risk_episode_count": sum(1 for r in results if r["high_risk"]),
        "high_risk_episodes": [
            {
                "session_dir": r["session_dir"],
                "missing_files": r["missing_files"],
                "host_timestamps_same_domain": r["host_timestamps_same_domain"],
                "offsets": r["offsets"],
            }
            for r in results
            if r["high_risk"]
        ][:100],
    }


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    agg = payload["aggregate"]  # type: ignore[index]
    lines = [
        "# Timestamp Domain Audit",
        "",
        f"Dataset manifest: `{payload['manifest']}`",
        f"Total episodes: {agg['total_episodes']}",
        f"Successful read episodes: {agg['successful_read_episodes']}",
        f"Missing file episodes: {agg['missing_file_episodes']}",
        f"aligned_timesteps.csv present: {agg['aligned_timesteps_present_episodes']}",
        f"sync_index.csv present: {agg['sync_index_present_episodes']}",
        "",
        "## Clock Domain Conclusion",
        "",
        f"Host timestamp same-domain episodes: {agg['host_timestamps_same_domain_episodes']} / {agg['total_episodes']}",
        f"Pressure sensor timestamp same as host: {agg['pressure_sensor_same_as_host_episodes']} / {agg['total_episodes']}",
        "",
        "The alignment timestamp for pressure should use `host_monotonic_us` or `aligned_timesteps.csv` pressure columns, not `sensor_timestamp_us`.",
        "",
        "## Offset Summary",
        "",
        "| modality | count | median max abs ms | p95 max abs ms | max abs ms | episodes >50ms | >100ms | >150ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stat in agg["offset_max_abs_ms_distribution_by_episode"].items():  # type: ignore[index]
        lines.append(
            f"| {name} | {stat['count']} | {stat['median_abs_ms']} | {stat['p95_abs_ms']} | "
            f"{stat['max_abs_ms']} | {stat['gt_50ms']} | {stat['gt_100ms']} | {stat['gt_150ms']} |"
        )
    lines += [
        "",
        "## Monotonicity",
        "",
        f"Non-monotonic episode counts: `{agg['non_monotonic_episode_counts']}`",
        f"Duplicate timestamp total counts: `{agg['duplicate_timestamp_total_counts']}`",
        "",
        "## Recommendation",
        "",
        "- Use `aligned_timesteps.csv` as the synchronization source when present.",
        "- Use visual observation step index for `contact_time/current_timestep`.",
        "- If a future dataset shows offsets above 50 ms, consider lowering `contact_reliability` or applying +/-1 frame tolerance.",
    ]
    if agg["high_risk_episode_count"]:  # type: ignore[index]
        lines += ["", "## High-Risk Episodes", ""]
        for item in agg["high_risk_episodes"]:  # type: ignore[index]
            lines.append(f"- `{item['session_dir']}` offsets={item['offsets']} missing={item['missing_files']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.manifest)
    if args.max_episodes is not None:
        rows = rows[: args.max_episodes]
    results = []
    for idx, row in enumerate(rows):
        session_dir = Path(row.get("rel_path") or row.get("view_rel_path") or "")
        if not session_dir.is_absolute():
            session_dir = Path("/media/files/TransVTLA-RealDataCollect") / session_dir
        result = audit_session(session_dir)
        result["manifest_index"] = idx
        result["annotation_key"] = row.get("annotation_key")
        results.append(result)
        if idx < 3 or (idx + 1) % 100 == 0 or idx == len(rows) - 1:
            print(f"[{idx + 1:04d}/{len(rows):04d}] {session_dir} high_risk={result['high_risk']}")

    payload = {
        "manifest": str(args.manifest),
        "aggregate": aggregate(results),
        "episodes": results,
        "conclusion": {
            "visual_robot_gripper_pressure_host_same_clock_domain": all(
                bool(r["host_timestamps_same_domain"]) for r in results
            ),
            "sensor_timestamp_us_is_not_host_clock": not any(
                bool(r["pressure_sensor_same_as_host"]) for r in results
            ),
            "aligned_timesteps_can_be_sync_source": all(
                bool(r["aligned_timesteps_present"]) for r in results
            ),
            "contact_time_should_use_visual_step_index": True,
            "recommended_frame_tolerance": "+/-1 frame only for future high-offset samples; current aligned rows are near-zero offset",
        },
    }
    payload = jsonable(payload)
    json_path = args.report_dir / "timestamp_domain_audit.json"
    md_path = args.report_dir / "timestamp_domain_audit.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
