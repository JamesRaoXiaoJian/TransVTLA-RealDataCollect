#!/usr/bin/env python3
"""Read-only structural and temporal audit for raw Franka simulation sessions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("dataset/franka_sim_10k_70f_848x480")


def wrapped_euler_delta(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Return the shortest per-axis Euler delta in [-pi, pi]."""
    return (current - previous + np.pi) % (2.0 * np.pi) - np.pi


def summarize_timestamps(timestamps: np.ndarray) -> dict[str, float | int | None]:
    if len(timestamps) < 2:
        return {"valid_intervals": 0, "nonpositive_intervals": 0, "median_hz": None}
    intervals = np.diff(timestamps)
    valid = intervals[intervals > 0]
    return {
        "valid_intervals": int(len(valid)),
        "nonpositive_intervals": int(np.count_nonzero(intervals <= 0)),
        "median_hz": None if len(valid) == 0 else float(1_000_000 / np.median(valid)),
    }


def _safe_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in {None, ""} else None
    except ValueError:
        return None


def _safe_int(value: str | None) -> int | None:
    try:
        return int(float(value)) if value not in {None, ""} else None
    except ValueError:
        return None


def _quantiles(values: list[float], scale: float = 1.0) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64) * scale
    return {
        "count": int(len(array)),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def summarize_values(values: list[float]) -> dict[str, float | int | bool | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "constant": False}
    array = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return {
        "count": int(len(array)),
        "min": minimum,
        "max": maximum,
        "constant": bool(np.isclose(minimum, maximum)),
    }


def _read_csv(path: Path) -> tuple[list[dict[str, str]], str | None]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle)), None
    except Exception as exc:  # Filesystem errors must be reported, not hidden.
        return [], f"{type(exc).__name__}: {exc}"


def _sample_image(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
            return {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "min": int(array.min()),
                "max": int(array.max()),
            }, None
    except Exception as exc:  # Corrupt image files are audit findings.
        return None, f"{type(exc).__name__}: {exc}"


def _nearest_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    upper = np.searchsorted(source_ts, target_ts, side="left")
    lower = np.clip(upper - 1, 0, len(source_ts) - 1)
    upper = np.clip(upper, 0, len(source_ts) - 1)
    use_upper = np.abs(source_ts[upper] - target_ts) < np.abs(source_ts[lower] - target_ts)
    indices = np.where(use_upper, upper, lower)
    return indices, np.abs(source_ts[indices] - target_ts)


@dataclass
class SessionAudit:
    session: str
    rel_path: str
    status: str = "ok"
    frame_count: int = 0
    robot_rows: int = 0
    gripper_rows: int = 0
    pressure_rows: int = 0
    pressure_channels: int = 0
    rgb_world: int = 0
    depth_world: int = 0
    rgb_wrist: int = 0
    depth_wrist: int = 0
    visual_hz: float | None = None
    robot_hz: float | None = None
    robot_offset_p95_ms: float | None = None
    gripper_constant: bool = False
    gripper_min: float | None = None
    gripper_max: float | None = None
    pressure_constant: bool = False
    pressure_min: float | None = None
    pressure_max: float | None = None
    image_decode_failures: int = 0
    issues: list[dict[str, str]] = field(default_factory=list)
    action_position_norm_m: list[float] = field(default_factory=list)
    action_rotation_norm_rad: list[float] = field(default_factory=list)
    euler_wrap_events: int = 0
    frame_intervals_us: list[float] = field(default_factory=list)
    robot_intervals_us: list[float] = field(default_factory=list)
    robot_offsets_us: list[float] = field(default_factory=list)

    def issue(self, severity: str, code: str, detail: str) -> None:
        if severity == "error":
            self.status = "error"
        elif self.status == "ok":
            self.status = "warning"
        self.issues.append({"severity": severity, "code": code, "detail": detail})


def _modality_files(session_dir: Path, camera: str, modality: str) -> list[Path]:
    try:
        return sorted(path for path in (session_dir / camera / modality).iterdir() if path.is_file())
    except Exception:
        return []


def discover_session_dirs(root: Path) -> list[Path]:
    """Discover both direct run sessions and legacy sharded sessions."""
    direct = root.glob("session_*")
    sharded = root.glob("shards/*/session_*")
    return sorted({path for path in (*direct, *sharded) if path.is_dir()})


def audit_session(session_dir: Path, root: Path, image_samples: int) -> SessionAudit:
    audit = SessionAudit(session=session_dir.name, rel_path=str(session_dir.relative_to(root)))
    try:
        if not session_dir.is_dir():
            audit.issue("error", "session_unreadable", "session directory cannot be accessed")
            return audit
    except OSError as exc:
        audit.issue("error", "session_io_error", str(exc))
        return audit

    modalities = {
        "rgb_world": _modality_files(session_dir, "world_camera", "rgb"),
        "depth_world": _modality_files(session_dir, "world_camera", "depth"),
        "rgb_wrist": _modality_files(session_dir, "wrist_camera", "rgb"),
        "depth_wrist": _modality_files(session_dir, "wrist_camera", "depth"),
    }
    for name, paths in modalities.items():
        setattr(audit, name, len(paths))
        if not paths:
            audit.issue("error", "missing_modality", name)

    frame_rows, frame_error = _read_csv(session_dir / "frames.csv")
    if frame_error:
        audit.issue("error", "frames_csv_unreadable", frame_error)
    else:
        audit.frame_count = len(frame_rows)
        frame_ts = np.asarray(
            [value for row in frame_rows if (value := _safe_int(row.get("capture_monotonic_us"))) is not None],
            dtype=np.int64,
        )
        frame_summary = summarize_timestamps(frame_ts)
        audit.visual_hz = frame_summary["median_hz"] if isinstance(frame_summary["median_hz"], float) else None
        if len(frame_ts) > 1:
            intervals = np.diff(frame_ts)
            audit.frame_intervals_us.extend(intervals[intervals > 0].astype(float).tolist())
        if frame_summary["nonpositive_intervals"]:
            audit.issue("error", "nonmonotonic_frame_timestamps", str(frame_summary["nonpositive_intervals"]))
        for name, paths in modalities.items():
            if paths and len(paths) != audit.frame_count:
                audit.issue("warning", "modality_frame_count_mismatch", f"{name}={len(paths)} frames={audit.frame_count}")
    
    robot_rows, robot_error = _read_csv(session_dir / "robot_state" / "robot_state.csv")
    if robot_error:
        audit.issue("error", "robot_csv_unreadable", robot_error)
    else:
        audit.robot_rows = len(robot_rows)
        robot_ts: list[int] = []
        robot_pose: list[list[float]] = []
        for row in robot_rows:
            timestamp = _safe_int(row.get("timestamp_us"))
            pose = [_safe_float(row.get(f"pose_{index}")) for index in range(1, 7)]
            if timestamp is None or any(value is None for value in pose):
                audit.issue("warning", "invalid_robot_row", "missing timestamp or pose component")
                continue
            robot_ts.append(timestamp)
            robot_pose.append([float(value) for value in pose if value is not None])
        robot_ts_array = np.asarray(robot_ts, dtype=np.int64)
        robot_summary = summarize_timestamps(robot_ts_array)
        audit.robot_hz = robot_summary["median_hz"] if isinstance(robot_summary["median_hz"], float) else None
        if len(robot_ts_array) > 1:
            intervals = np.diff(robot_ts_array)
            audit.robot_intervals_us.extend(intervals[intervals > 0].astype(float).tolist())
        if robot_summary["nonpositive_intervals"]:
            audit.issue("error", "nonmonotonic_robot_timestamps", str(robot_summary["nonpositive_intervals"]))
        if frame_rows and len(robot_ts_array):
            frame_ts = np.asarray(
                [value for row in frame_rows if (value := _safe_int(row.get("capture_monotonic_us"))) is not None],
                dtype=np.int64,
            )
            indices, offsets = _nearest_indices(robot_ts_array, frame_ts)
            audit.robot_offsets_us.extend(offsets.astype(float).tolist())
            if len(offsets):
                audit.robot_offset_p95_ms = float(np.quantile(offsets, 0.95) / 1000)
            aligned_pose = np.asarray(robot_pose, dtype=np.float32)[indices]
            if len(aligned_pose) > 1:
                raw_delta = np.diff(aligned_pose, axis=0)
                wrapped_delta = raw_delta.copy()
                wrapped_delta[:, 3:6] = wrapped_euler_delta(aligned_pose[1:, 3:6], aligned_pose[:-1, 3:6])
                audit.action_position_norm_m.extend(np.linalg.norm(wrapped_delta[:, :3], axis=1).tolist())
                audit.action_rotation_norm_rad.extend(np.linalg.norm(wrapped_delta[:, 3:6], axis=1).tolist())
                audit.euler_wrap_events += int(np.count_nonzero(np.abs(raw_delta[:, 3:6] - wrapped_delta[:, 3:6]) > 1e-4))

    gripper_rows, gripper_error = _read_csv(session_dir / "robot_state" / "gripper_state.csv")
    if gripper_error:
        audit.issue("error", "gripper_csv_unreadable", gripper_error)
    else:
        audit.gripper_rows = len(gripper_rows)
        values = [value for row in gripper_rows if (value := _safe_float(row.get("gripper_pos"))) is not None]
        gripper_summary = summarize_values(values)
        audit.gripper_constant = bool(gripper_summary["constant"])
        audit.gripper_min = gripper_summary["min"] if isinstance(gripper_summary["min"], float) else None
        audit.gripper_max = gripper_summary["max"] if isinstance(gripper_summary["max"], float) else None

    pressure_rows, pressure_error = _read_csv(session_dir / "pressure" / "pressure.csv")
    if pressure_error:
        audit.issue("error", "pressure_csv_unreadable", pressure_error)
    else:
        audit.pressure_rows = len(pressure_rows)
        headers = set(pressure_rows[0]) if pressure_rows else set()
        channels = sorted(header for header in headers if header.startswith("CH"))
        audit.pressure_channels = len(channels)
        if audit.pressure_rows and audit.pressure_channels != 20:
            audit.issue("warning", "pressure_channel_count", str(audit.pressure_channels))
        pressure_min = math.inf
        pressure_max = -math.inf
        for row in pressure_rows:
            for channel in channels:
                value = _safe_float(row.get(channel))
                if value is not None:
                    pressure_min = min(pressure_min, value)
                    pressure_max = max(pressure_max, value)
        if pressure_min < math.inf:
            audit.pressure_min = pressure_min
            audit.pressure_max = pressure_max
            audit.pressure_constant = bool(np.isclose(pressure_min, pressure_max))

    for name, paths in modalities.items():
        if not paths or image_samples <= 0:
            continue
        sample_indices = sorted({0, len(paths) // 2, len(paths) - 1})[:image_samples]
        for index in sample_indices:
            _image, image_error = _sample_image(paths[index])
            if image_error:
                audit.image_decode_failures += 1
                audit.issue("error", "image_decode_failure", f"{name}/{paths[index].name}: {image_error}")

    return audit


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    issue_rows = summary["issue_counts"]
    lines = [
        "# Raw Franka Simulation Dataset Audit",
        "",
        f"- Root: `{summary['root']}`",
        f"- Sessions discovered: {summary['sessions_discovered']}",
        f"- Sessions scanned: {summary['sessions_scanned']}",
        f"- Errors: {summary['sessions_error']}",
        f"- Warnings: {summary['sessions_warning']}",
        f"- Clean: {summary['sessions_ok']}",
        "",
        "## Temporal Summary",
        "",
        f"- Visual inter-frame rate: median {summary['visual_hz']['p50']:.3f} Hz; P95 {summary['visual_hz']['p95']:.3f} Hz.",
        f"- Robot state rate: median {summary['robot_hz']['p50']:.3f} Hz; P95 {summary['robot_hz']['p95']:.3f} Hz.",
        f"- Robot-to-visual nearest offset: P95 {summary['robot_offset_ms']['p95']:.3f} ms; max {summary['robot_offset_ms']['max']:.3f} ms.",
        f"- Sessions with constant gripper position: {summary['constant_gripper_sessions']}.",
        f"- Sessions with constant raw pressure: {summary['constant_pressure_sessions']}.",
        f"- Raw gripper range: {summary['gripper_value_range']['min']:.6f} to {summary['gripper_value_range']['max']:.6f}.",
        f"- Raw pressure range: {summary['pressure_value_range']['min']:.3f} to {summary['pressure_value_range']['max']:.3f}.",
        "",
        "## Motion Summary",
        "",
        f"- Position delta norm: P50 {summary['position_delta_m']['p50'] * 1000:.3f} mm; P95 {summary['position_delta_m']['p95'] * 1000:.3f} mm; P99 {summary['position_delta_m']['p99'] * 1000:.3f} mm.",
        f"- Wrapped rotation delta norm: P50 {summary['rotation_delta_rad']['p50']:.5f} rad; P95 {summary['rotation_delta_rad']['p95']:.5f} rad; P99 {summary['rotation_delta_rad']['p99']:.5f} rad.",
        f"- Euler wrap events: {summary['euler_wrap_events']}",
        "",
        "## Issue Counts",
        "",
        "| code | count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{code}` | {count} |" for code, count in sorted(issue_rows.items()))
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This report is read-only. It distinguishes raw-file corruption from known model-training risks. "
        "In particular, Euler wrap events, world-frame coordinates in multi-environment simulation, "
        "constant gripper values and the distinction between raw pressure values and the all-zero tactile field in the current RLDS should be handled by derived training views, not by modifying raw sessions.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-samples", type=int, default=3)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / "audit_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dirs = discover_session_dirs(root)
    audits = [audit_session(path, root, args.image_samples) for path in session_dirs]

    issue_counts: Counter[str] = Counter()
    frame_hz: list[float] = []
    robot_hz: list[float] = []
    frame_intervals: list[float] = []
    robot_intervals: list[float] = []
    robot_offsets: list[float] = []
    position_delta: list[float] = []
    rotation_delta: list[float] = []
    gripper_mins: list[float] = []
    gripper_maxs: list[float] = []
    pressure_mins: list[float] = []
    pressure_maxs: list[float] = []
    for audit in audits:
        issue_counts.update(issue["code"] for issue in audit.issues)
        if audit.visual_hz is not None:
            frame_hz.append(audit.visual_hz)
        if audit.robot_hz is not None:
            robot_hz.append(audit.robot_hz)
        frame_intervals.extend(audit.frame_intervals_us)
        robot_intervals.extend(audit.robot_intervals_us)
        robot_offsets.extend(audit.robot_offsets_us)
        position_delta.extend(audit.action_position_norm_m)
        rotation_delta.extend(audit.action_rotation_norm_rad)
        if audit.gripper_min is not None:
            gripper_mins.append(audit.gripper_min)
        if audit.gripper_max is not None:
            gripper_maxs.append(audit.gripper_max)
        if audit.pressure_min is not None:
            pressure_mins.append(audit.pressure_min)
        if audit.pressure_max is not None:
            pressure_maxs.append(audit.pressure_max)

    summary = {
        "schema": "transvtla_raw_sim_audit/v1",
        "root": str(root),
        "sessions_discovered": len(session_dirs),
        "sessions_scanned": len(audits),
        "sessions_ok": sum(a.status == "ok" for a in audits),
        "sessions_warning": sum(a.status == "warning" for a in audits),
        "sessions_error": sum(a.status == "error" for a in audits),
        "issue_counts": dict(sorted(issue_counts.items())),
        "visual_hz": _quantiles(frame_hz),
        "robot_hz": _quantiles(robot_hz),
        "visual_interval_ms": _quantiles(frame_intervals, scale=1 / 1000),
        "robot_interval_ms": _quantiles(robot_intervals, scale=1 / 1000),
        "robot_offset_ms": _quantiles(robot_offsets, scale=1 / 1000),
        "position_delta_m": _quantiles(position_delta),
        "rotation_delta_rad": _quantiles(rotation_delta),
        "euler_wrap_events": sum(a.euler_wrap_events for a in audits),
        "image_decode_failures": sum(a.image_decode_failures for a in audits),
        "constant_gripper_sessions": sum(a.gripper_constant for a in audits),
        "constant_pressure_sessions": sum(a.pressure_constant for a in audits),
        "pressure_channel_counts": dict(sorted(Counter(a.pressure_channels for a in audits).items())),
        "gripper_value_range": {
            "min": None if not gripper_mins else float(min(gripper_mins)),
            "max": None if not gripper_maxs else float(max(gripper_maxs)),
        },
        "pressure_value_range": {
            "min": None if not pressure_mins else float(min(pressure_mins)),
            "max": None if not pressure_maxs else float(max(pressure_maxs)),
        },
    }
    per_session_rows = [
        {
            "session": audit.session,
            "rel_path": audit.rel_path,
            "status": audit.status,
            "frame_count": audit.frame_count,
            "robot_rows": audit.robot_rows,
            "gripper_rows": audit.gripper_rows,
            "pressure_rows": audit.pressure_rows,
            "pressure_channels": audit.pressure_channels,
            "rgb_world": audit.rgb_world,
            "depth_world": audit.depth_world,
            "rgb_wrist": audit.rgb_wrist,
            "depth_wrist": audit.depth_wrist,
            "visual_hz": audit.visual_hz,
            "robot_hz": audit.robot_hz,
            "robot_offset_p95_ms": audit.robot_offset_p95_ms,
            "gripper_constant": audit.gripper_constant,
            "gripper_min": audit.gripper_min,
            "gripper_max": audit.gripper_max,
            "pressure_constant": audit.pressure_constant,
            "pressure_min": audit.pressure_min,
            "pressure_max": audit.pressure_max,
            "euler_wrap_events": audit.euler_wrap_events,
            "image_decode_failures": audit.image_decode_failures,
            "issues": "|".join(issue["code"] for issue in audit.issues),
        }
        for audit in audits
    ]
    issue_rows = [
        {"session": audit.session, "rel_path": audit.rel_path, **issue}
        for audit in audits
        for issue in audit.issues
    ]
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "per_session.csv", per_session_rows, list(per_session_rows[0].keys()) if per_session_rows else [])
    _write_csv(output_dir / "issues.csv", issue_rows, ["session", "rel_path", "severity", "code", "detail"])
    (output_dir / "REPORT.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
