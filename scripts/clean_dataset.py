#!/usr/bin/env python3
"""Clean and summarize TransVTLA real-data sessions.

This script performs the repair pass requested after the sessions.zip merge:

* Sort pressure rows by their usable timestamp column.
* Replace bad pressure channel CH58 with the mean of CH61, CH57, and CH59.
* Trim legacy 64-channel pressure CSVs to the configured 20 tactile channels.
* Deduplicate gripper rows by timestamp and fill missing sys_state values.
* Rebuild dual-RealSense aligned_timesteps.csv files from cleaned CSV sources.
* Write a clean split manifest for downstream dataset construction.

Default mode is dry-run. Use --apply to rewrite files after one-time backups.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from channel_config import INTERPOLATE_CHANNELS, VALID_CHANNELS  # noqa: E402


DEFAULT_ROOTS = [
    Path("dataset/phase2_realdata_sessions/sessions"),
    Path("dataset/sessions"),
]

BEGIN_MARKER = "session_20260615_193300_begin"
END_MARKER = "session_20260615_201528_end"
DATE_RE = re.compile(r"session_(\d{8})_(\d{6})")
BACKUP_SUFFIX = ".bak_before_clean"

PRESSURE_TS_COLUMNS = ("host_monotonic_us", "timestamp_us", "sensor_timestamp_us")
ROBOT_TS_COLUMNS = ("timestamp_us",)
GRIPPER_TS_COLUMNS = ("timestamp_us",)
FRAME_TS_COLUMN = "capture_monotonic_us"

DISCRETE_GRIPPER_COLUMNS = {
    "rm_plus_read_code",
    "sys_state",
    "gripper_dof_state",
    "gripper_dof_err",
}


@dataclass
class FileChange:
    path: str
    status: str
    rows: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleaningSummary:
    apply: bool
    sessions_seen: int = 0
    pressure_files_seen: int = 0
    pressure_files_changed: int = 0
    pressure_files_sorted: int = 0
    pressure_rows_reordered: int = 0
    pressure_ch58_files_changed: int = 0
    pressure_ch58_rows_replaced: int = 0
    pressure_ch58_rows_changed: int = 0
    pressure_ch58_rows_failed: int = 0
    pressure_64ch_files_trimmed: int = 0
    gripper_files_seen: int = 0
    gripper_files_changed: int = 0
    gripper_duplicate_rows_removed: int = 0
    gripper_sys_state_filled: int = 0
    aligned_files_seen: int = 0
    aligned_files_written: int = 0
    aligned_files_created: int = 0
    aligned_rows_written: int = 0
    manifest_csv: str = ""
    manifest_json: str = ""
    detail_files: list[FileChange] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _safe_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(str(value))
    except (TypeError, ValueError):
        return None
    if math.isnan(value_f) or math.isinf(value_f):
        return None
    return value_f


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    text = f"{float(value):.6f}"
    return text.rstrip("0").rstrip(".")


def _format_int_mean(values: list[float]) -> str:
    return str(int(sum(values) / len(values)))


def _format_float6(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _format_offset_ms(source_ts: int | None, target_ts: int) -> str:
    if source_ts is None:
        return ""
    return f"{(source_ts - target_ts) / 1000.0:.3f}"


def _parse_session_datetime(name: str) -> tuple[str, str]:
    match = DATE_RE.search(name)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def _looks_like_session(path: Path) -> bool:
    camera_like = any(
        (path / rel).is_dir()
        for rel in (
            "world_camera/rgb",
            "wrist_camera/rgb",
            "dji",
            "realsense_rgb",
        )
    )
    data_like = any(
        (path / rel).is_file()
        for rel in (
            "frames.csv",
            "pressure/pressure.csv",
            "robot_state/robot_state.csv",
        )
    )
    return camera_like or data_like


def discover_sessions(roots: Iterable[Path]) -> list[Path]:
    sessions: set[Path] = set()
    for raw_root in roots:
        root = raw_root if raw_root.is_absolute() else REPO_ROOT / raw_root
        if not root.exists():
            continue
        candidates = [root] if root.name.startswith("session_") else root.rglob("session_*")
        for candidate in candidates:
            if candidate.is_dir() and candidate.name.startswith("session_") and _looks_like_session(candidate):
                sessions.add(candidate.resolve())
    return sorted(sessions, key=lambda p: str(p))


def build_begin_end_set(sessions: Iterable[Path]) -> set[str]:
    by_parent: dict[Path, list[Path]] = defaultdict(list)
    for session in sessions:
        by_parent[session.parent].append(session)

    names: set[str] = set()
    for group in by_parent.values():
        ordered_names = [p.name for p in sorted(group, key=lambda p: p.name)]
        if BEGIN_MARKER not in ordered_names or END_MARKER not in ordered_names:
            continue
        start = ordered_names.index(BEGIN_MARKER)
        end = ordered_names.index(END_MARKER)
        if start <= end:
            names.update(ordered_names[start : end + 1])
    return names


def detect_format(session: Path) -> str:
    if (session / "world_camera/rgb").is_dir() and (session / "wrist_camera/rgb").is_dir():
        return "dual_realsense"
    if (
        (session / "dji").is_dir()
        and (session / "realsense_rgb").is_dir()
        and (session / "realsense_depth").is_dir()
    ):
        return "legacy_dji_realsense"
    if (session / "dji").is_dir() and (session / "realsense_rgb").is_dir():
        return "legacy_dji_realsense_rgb_only"
    return "unknown"


def classify_session(session: Path, begin_end_names: set[str]) -> tuple[str, str, str, str, str]:
    date, _time = _parse_session_datetime(session.name)
    data_format = detect_format(session)

    if date and date < "20260615":
        phase = "pre_0615_dji_world"
    elif data_format in {"legacy_dji_realsense", "legacy_dji_realsense_rgb_only"}:
        phase = "pre_0615_dji_world"
    elif date and date >= "20260615":
        phase = "post_0615_dual_realsense"
    else:
        phase = "unknown_phase"

    if session.name in begin_end_names:
        view_tag = "0615_initial_realsense_position_incomplete_arm_view"
    elif phase == "pre_0615_dji_world":
        view_tag = "legacy_dji_world_position"
    elif phase == "post_0615_dual_realsense":
        view_tag = "realsense_repositioned_after_begin_end"
    else:
        view_tag = "unknown_view"

    if data_format == "legacy_dji_realsense_rgb_only":
        split = "legacy_dji_realsense_rgb_only"
    elif data_format == "legacy_dji_realsense":
        split = "legacy_dji_realsense_rgbd"
    elif view_tag == "0615_initial_realsense_position_incomplete_arm_view":
        split = "dual_realsense_initial_position_begin_end"
    elif data_format == "dual_realsense":
        split = "dual_realsense_repositioned"
    else:
        split = "unknown"

    return data_format, phase, view_tag, split, date


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({name: row.get(name, "") for name in header})
    return header, rows


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]], apply: bool) -> None:
    if not apply:
        return
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists() and path.exists():
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".clean.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header})
    tmp.replace(path)


def _is_channel_col(name: str) -> bool:
    return name.startswith("CH") and name[2:].isdigit()


def _timestamp_column(header: list[str], preferred: Iterable[str]) -> str:
    for col in preferred:
        if col in header:
            return col
    return header[0] if header else ""


def _count_reordered(original: list[dict[str, str]], ordered: list[dict[str, str]], ts_col: str) -> int:
    if original == ordered:
        return 0
    return sum(1 for before, after in zip(original, ordered) if before.get(ts_col) != after.get(ts_col))


def clean_pressure_file(path: Path, apply: bool) -> FileChange:
    header, rows = _read_csv(path)
    original_header = list(header)
    original_rows = [dict(row) for row in rows]
    channel_cols = [c for c in header if _is_channel_col(c)]
    original_channel_count = len(channel_cols)
    ts_col = _timestamp_column(header, PRESSURE_TS_COLUMNS)

    sorted_rows = rows
    timestamp_reversals = 0
    rows_reordered = 0
    if ts_col:
        indexed_ts: list[tuple[int, int, dict[str, str]]] = []
        all_valid = True
        prev_ts: int | None = None
        for idx, row in enumerate(rows):
            ts = _safe_int(row.get(ts_col))
            if ts is None:
                all_valid = False
                break
            if prev_ts is not None and ts < prev_ts:
                timestamp_reversals += 1
            prev_ts = ts
            indexed_ts.append((ts, idx, row))
        if all_valid:
            sorted_rows = [row for _ts, _idx, row in sorted(indexed_ts, key=lambda item: (item[0], item[1]))]
            rows_reordered = _count_reordered(rows, sorted_rows, ts_col)
            rows = sorted_rows

    replaced_rows = 0
    changed_rows = 0
    failed_rows = 0
    for bad_ch, adjacent_chs in INTERPOLATE_CHANNELS.items():
        bad_col = f"CH{bad_ch}"
        adjacent_cols = [f"CH{ch}" for ch in adjacent_chs]
        if bad_col not in header or any(col not in header for col in adjacent_cols):
            continue
        for row in rows:
            values = [_safe_float(row.get(col)) for col in adjacent_cols]
            if any(value is None for value in values):
                failed_rows += 1
                continue
            new_value = _format_int_mean([v for v in values if v is not None])
            if row.get(bad_col, "") != new_value:
                changed_rows += 1
            row[bad_col] = new_value
            replaced_rows += 1

    non_channel_cols = [c for c in header if not _is_channel_col(c)]
    valid_channel_cols = [f"CH{ch}" for ch in VALID_CHANNELS]
    output_header = non_channel_cols + [c for c in valid_channel_cols if c in header]
    missing_valid_cols = [c for c in valid_channel_cols if c not in header]
    trimmed = original_channel_count > len([c for c in output_header if _is_channel_col(c)])

    output_rows = [{name: row.get(name, "") for name in output_header} for row in rows]
    changed = original_header != output_header or original_rows != output_rows
    if changed:
        _write_csv(path, output_header, output_rows, apply)

    status = "changed" if changed else "unchanged"
    return FileChange(
        path=_rel(path),
        status=status,
        rows=len(rows),
        details={
            "timestamp_column": ts_col,
            "timestamp_reversals": timestamp_reversals,
            "rows_reordered": rows_reordered,
            "original_channel_count": original_channel_count,
            "output_channel_count": len([c for c in output_header if _is_channel_col(c)]),
            "trimmed_64_to_20": trimmed and original_channel_count == 64,
            "ch58_rows_replaced": replaced_rows,
            "ch58_rows_changed": changed_rows,
            "ch58_rows_failed": failed_rows,
            "missing_valid_channels": missing_valid_cols,
        },
    )


def clean_gripper_file(path: Path, apply: bool) -> FileChange:
    header, rows = _read_csv(path)
    original_rows = [dict(row) for row in rows]
    ts_col = _timestamp_column(header, GRIPPER_TS_COLUMNS)

    duplicate_rows = 0
    if ts_col:
        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for row in rows:
            ts = row.get(ts_col, "")
            if ts in seen:
                duplicate_rows += 1
                continue
            seen.add(ts)
            deduped.append(row)
        rows = deduped

    filled = 0
    if "sys_state" in header:
        last_valid = ""
        for row in rows:
            if row.get("sys_state", "") == "":
                if last_valid:
                    row["sys_state"] = last_valid
                    filled += 1
            else:
                last_valid = row["sys_state"]

        next_valid = ""
        for row in reversed(rows):
            if row.get("sys_state", "") == "":
                if next_valid:
                    row["sys_state"] = next_valid
                    filled += 1
            else:
                next_valid = row["sys_state"]

    changed = original_rows != rows
    if changed:
        _write_csv(path, header, rows, apply)

    return FileChange(
        path=_rel(path),
        status="changed" if changed else "unchanged",
        rows=len(rows),
        details={
            "timestamp_column": ts_col,
            "duplicate_rows_removed": duplicate_rows,
            "sys_state_filled": filled,
        },
    )


def _read_sorted_series(path: Path, preferred_ts: Iterable[str]) -> tuple[list[str], list[dict[str, str]], str, list[int]]:
    if not path.is_file():
        return [], [], "", []
    header, rows = _read_csv(path)
    ts_col = _timestamp_column(header, preferred_ts)
    if not ts_col:
        return header, [], "", []

    indexed: list[tuple[int, int, dict[str, str]]] = []
    seen: set[int] = set()
    for idx, row in enumerate(rows):
        ts = _safe_int(row.get(ts_col))
        if ts is None or ts in seen:
            continue
        seen.add(ts)
        indexed.append((ts, idx, row))
    indexed.sort(key=lambda item: (item[0], item[1]))
    sorted_rows = [row for _ts, _idx, row in indexed]
    timestamps = [ts for ts, _idx, _row in indexed]
    return header, sorted_rows, ts_col, timestamps


def _interpolate_row(
    target_ts: int,
    header: list[str],
    rows: list[dict[str, str]],
    timestamps: list[int],
    value_columns: list[str],
    discrete_columns: set[str] | None = None,
) -> tuple[int | None, dict[str, str]]:
    if not rows or not timestamps:
        return None, {col: "" for col in value_columns}

    discrete_columns = discrete_columns or set()
    if target_ts <= timestamps[0]:
        nearest_idx = 0
        lower_idx = upper_idx = 0
    elif target_ts >= timestamps[-1]:
        nearest_idx = len(timestamps) - 1
        lower_idx = upper_idx = nearest_idx
    else:
        upper_idx = bisect.bisect_left(timestamps, target_ts)
        lower_idx = upper_idx - 1
        if abs(timestamps[lower_idx] - target_ts) <= abs(timestamps[upper_idx] - target_ts):
            nearest_idx = lower_idx
        else:
            nearest_idx = upper_idx

    source_ts = timestamps[nearest_idx]
    lower_row = rows[lower_idx]
    upper_row = rows[upper_idx]
    lower_ts = timestamps[lower_idx]
    upper_ts = timestamps[upper_idx]
    nearest_row = rows[nearest_idx]
    ratio = 0.0 if upper_ts == lower_ts else (target_ts - lower_ts) / (upper_ts - lower_ts)

    values: dict[str, str] = {}
    for col in value_columns:
        if col not in header:
            values[col] = ""
            continue
        if col in discrete_columns:
            values[col] = nearest_row.get(col, "")
            continue
        lower_value = _safe_float(lower_row.get(col))
        upper_value = _safe_float(upper_row.get(col))
        if lower_value is None or upper_value is None:
            values[col] = nearest_row.get(col, "")
            continue
        values[col] = _format_float6(lower_value + (upper_value - lower_value) * ratio)
    return source_ts, values


def _frame_rows(session: Path) -> tuple[list[dict[str, str]], list[int]]:
    frames_path = session / "frames.csv"
    if not frames_path.is_file():
        return [], []
    _header, rows = _read_csv(frames_path)
    usable_rows: list[dict[str, str]] = []
    timestamps: list[int] = []
    for row in rows:
        ts = _safe_int(row.get(FRAME_TS_COLUMN))
        if ts is None:
            continue
        usable_rows.append(row)
        timestamps.append(ts)
    return usable_rows, timestamps


def rebuild_alignment(session: Path, apply: bool) -> FileChange:
    aligned_path = session / "aligned_timesteps.csv"
    frames, frame_timestamps = _frame_rows(session)
    if not frames:
        return FileChange(path=_rel(aligned_path), status="skipped_no_frames", rows=0)

    robot_header, robot_rows, _robot_ts_col, robot_ts = _read_sorted_series(
        session / "robot_state" / "robot_state.csv",
        ROBOT_TS_COLUMNS,
    )
    gripper_header, gripper_rows, _gripper_ts_col, gripper_ts = _read_sorted_series(
        session / "robot_state" / "gripper_state.csv",
        GRIPPER_TS_COLUMNS,
    )
    pressure_header, pressure_rows, _pressure_ts_col, pressure_ts = _read_sorted_series(
        session / "pressure" / "pressure.csv",
        PRESSURE_TS_COLUMNS,
    )

    robot_value_cols = [c for c in robot_header if c != "timestamp_us"]
    gripper_value_cols = [c for c in gripper_header if c != "timestamp_us"]
    pressure_value_cols = [f"CH{ch}" for ch in VALID_CHANNELS if f"CH{ch}" in pressure_header]

    output_header = (
        ["frame_id", "visual_timestamp_us", "robot_timestamp_us", "robot_offset_ms"]
        + [f"robot_{col}" for col in robot_value_cols]
        + ["gripper_timestamp_us", "gripper_offset_ms"]
        + [f"gripper_{col}" for col in gripper_value_cols]
        + ["pressure_timestamp_us", "pressure_offset_ms"]
        + [f"pressure_{col}" for col in pressure_value_cols]
    )

    output_rows: list[dict[str, str]] = []
    for idx, target_ts in enumerate(frame_timestamps):
        robot_source_ts, robot_values = _interpolate_row(
            target_ts,
            robot_header,
            robot_rows,
            robot_ts,
            robot_value_cols,
        )
        gripper_source_ts, gripper_values = _interpolate_row(
            target_ts,
            gripper_header,
            gripper_rows,
            gripper_ts,
            gripper_value_cols,
            discrete_columns=DISCRETE_GRIPPER_COLUMNS,
        )
        pressure_source_ts, pressure_values = _interpolate_row(
            target_ts,
            pressure_header,
            pressure_rows,
            pressure_ts,
            pressure_value_cols,
        )

        row: dict[str, str] = {
            "frame_id": str(idx),
            "visual_timestamp_us": str(target_ts),
            "robot_timestamp_us": "" if robot_source_ts is None else str(robot_source_ts),
            "robot_offset_ms": _format_offset_ms(robot_source_ts, target_ts),
            "gripper_timestamp_us": "" if gripper_source_ts is None else str(gripper_source_ts),
            "gripper_offset_ms": _format_offset_ms(gripper_source_ts, target_ts),
            "pressure_timestamp_us": "" if pressure_source_ts is None else str(pressure_source_ts),
            "pressure_offset_ms": _format_offset_ms(pressure_source_ts, target_ts),
        }
        row.update({f"robot_{col}": value for col, value in robot_values.items()})
        row.update({f"gripper_{col}": value for col, value in gripper_values.items()})
        row.update({f"pressure_{col}": value for col, value in pressure_values.items()})
        for bad_ch, adjacent_chs in INTERPOLATE_CHANNELS.items():
            bad_key = f"pressure_CH{bad_ch}"
            adjacent_keys = [f"pressure_CH{ch}" for ch in adjacent_chs]
            adjacent_values = [_safe_float(row.get(key)) for key in adjacent_keys]
            if bad_key in row and all(value is not None for value in adjacent_values):
                row[bad_key] = _format_int_mean([v for v in adjacent_values if v is not None])
        output_rows.append(row)

    existing_header: list[str] = []
    existing_rows: list[dict[str, str]] = []
    existed = aligned_path.is_file()
    if existed:
        existing_header, existing_rows = _read_csv(aligned_path)
    changed = existing_header != output_header or existing_rows != output_rows
    if changed:
        _write_csv(aligned_path, output_header, output_rows, apply)

    return FileChange(
        path=_rel(aligned_path),
        status="changed" if changed else "unchanged",
        rows=len(output_rows),
        details={
            "created": not existed,
            "frame_rows": len(frames),
            "robot_rows": len(robot_rows),
            "gripper_rows": len(gripper_rows),
            "pressure_rows": len(pressure_rows),
            "pressure_channels": len(pressure_value_cols),
        },
    )


def _count_csv_rows(path: Path) -> tuple[int, list[str]]:
    if not path.is_file():
        return 0, []
    header, rows = _read_csv(path)
    return len(rows), header


def _timestamp_rate(path: Path, preferred_ts: Iterable[str]) -> float | None:
    if not path.is_file():
        return None
    header, rows = _read_csv(path)
    ts_col = _timestamp_column(header, preferred_ts)
    timestamps = [_safe_int(row.get(ts_col)) for row in rows]
    timestamps = [ts for ts in timestamps if ts is not None]
    if len(timestamps) < 2:
        return None
    duration_s = (max(timestamps) - min(timestamps)) / 1_000_000
    if duration_s <= 0:
        return None
    return (len(timestamps) - 1) / duration_s


def _count_images(session: Path, rel_dir: str, suffix: str) -> int:
    path = session / rel_dir
    if not path.is_dir():
        return 0
    return sum(1 for item in path.glob(f"*{suffix}") if item.is_file())


def _complete_visual_count(session: Path, data_format: str) -> int:
    if data_format == "dual_realsense":
        counts = [
            _count_images(session, "world_camera/rgb", ".jpg"),
            _count_images(session, "world_camera/depth", ".png"),
            _count_images(session, "wrist_camera/rgb", ".jpg"),
            _count_images(session, "wrist_camera/depth", ".png"),
        ]
    elif data_format == "legacy_dji_realsense":
        counts = [
            _count_images(session, "dji", ".jpg"),
            _count_images(session, "realsense_rgb", ".jpg"),
            _count_images(session, "realsense_depth", ".png"),
        ]
    elif data_format == "legacy_dji_realsense_rgb_only":
        counts = [
            _count_images(session, "dji", ".jpg"),
            _count_images(session, "realsense_rgb", ".jpg"),
        ]
    else:
        counts = []
    return min(counts) if counts and all(count > 0 for count in counts) else 0


def _round_rate(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def write_manifest(sessions: list[Path], begin_end_names: set[str], out_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "clean_split_manifest.csv"
    json_path = out_dir / "clean_split_manifest.json"

    fieldnames = [
        "session",
        "rel_path",
        "date",
        "time",
        "split",
        "phase",
        "view_tag",
        "data_format",
        "usable_for_rgb",
        "usable_for_rgbd",
        "usable_for_aligned_training",
        "visual_complete_frames",
        "frames_rows",
        "aligned_rows",
        "pressure_rows",
        "pressure_channel_count",
        "pressure_hz",
        "robot_rows",
        "robot_hz",
        "gripper_rows",
        "gripper_hz",
        "notes",
    ]

    rows: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    split_frames: Counter[str] = Counter()

    for session in sessions:
        data_format, phase, view_tag, split, date = classify_session(session, begin_end_names)
        _date, time_part = _parse_session_datetime(session.name)
        frames_rows, _frames_header = _count_csv_rows(session / "frames.csv")
        aligned_rows, _aligned_header = _count_csv_rows(session / "aligned_timesteps.csv")
        pressure_rows, pressure_header = _count_csv_rows(session / "pressure" / "pressure.csv")
        robot_rows, _robot_header = _count_csv_rows(session / "robot_state" / "robot_state.csv")
        gripper_rows, _gripper_header = _count_csv_rows(session / "robot_state" / "gripper_state.csv")
        pressure_channel_count = len([c for c in pressure_header if _is_channel_col(c)])
        visual_complete_frames = _complete_visual_count(session, data_format)

        notes: list[str] = []
        if data_format == "legacy_dji_realsense_rgb_only":
            notes.append("rgb_only_no_depth")
        if view_tag == "0615_initial_realsense_position_incomplete_arm_view":
            notes.append("incomplete_arm_view")
        if data_format == "dual_realsense" and aligned_rows == 0:
            notes.append("missing_aligned_timesteps")
        if data_format == "dual_realsense" and frames_rows != visual_complete_frames:
            notes.append("frames_vs_visual_count_mismatch")
        if pressure_channel_count != 20 and pressure_channel_count != 0:
            notes.append(f"pressure_channels_{pressure_channel_count}")

        usable_for_rgb = visual_complete_frames > 0 and data_format != "unknown"
        usable_for_rgbd = (
            visual_complete_frames > 0
            and data_format in {"legacy_dji_realsense", "dual_realsense"}
        )
        usable_for_aligned = (
            data_format == "dual_realsense"
            and aligned_rows > 0
            and pressure_channel_count == 20
            and robot_rows > 0
            and gripper_rows > 0
        )

        split_counts[split] += 1
        split_frames[split] += visual_complete_frames
        rows.append(
            {
                "session": session.name,
                "rel_path": _rel(session),
                "date": date,
                "time": time_part,
                "split": split,
                "phase": phase,
                "view_tag": view_tag,
                "data_format": data_format,
                "usable_for_rgb": str(int(usable_for_rgb)),
                "usable_for_rgbd": str(int(usable_for_rgbd)),
                "usable_for_aligned_training": str(int(usable_for_aligned)),
                "visual_complete_frames": str(visual_complete_frames),
                "frames_rows": str(frames_rows),
                "aligned_rows": str(aligned_rows),
                "pressure_rows": str(pressure_rows),
                "pressure_channel_count": str(pressure_channel_count),
                "pressure_hz": _round_rate(_timestamp_rate(session / "pressure" / "pressure.csv", PRESSURE_TS_COLUMNS)),
                "robot_rows": str(robot_rows),
                "robot_hz": _round_rate(_timestamp_rate(session / "robot_state" / "robot_state.csv", ROBOT_TS_COLUMNS)),
                "gripper_rows": str(gripper_rows),
                "gripper_hz": _round_rate(
                    _timestamp_rate(session / "robot_state" / "gripper_state.csv", GRIPPER_TS_COLUMNS)
                ),
                "notes": ";".join(notes),
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "transvtla_clean_split_manifest/v1",
        "sessions": len(rows),
        "splits": {
            split: {
                "sessions": split_counts[split],
                "visual_complete_frames": split_frames[split],
            }
            for split in sorted(split_counts)
        },
        "manifest_csv": _rel(csv_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return csv_path, json_path, payload


def run_cleaning(args: argparse.Namespace) -> CleaningSummary:
    roots = [Path(root) for root in args.roots]
    sessions = discover_sessions(roots)
    if args.limit is not None:
        sessions = sessions[: args.limit]
    begin_end_names = build_begin_end_set(sessions)

    summary = CleaningSummary(apply=args.apply, sessions_seen=len(sessions))

    for session in sessions:
        pressure_path = session / "pressure" / "pressure.csv"
        if pressure_path.is_file():
            summary.pressure_files_seen += 1
            try:
                change = clean_pressure_file(pressure_path, args.apply)
                summary.detail_files.append(change)
                details = change.details
                if change.status == "changed":
                    summary.pressure_files_changed += 1
                if details.get("rows_reordered", 0):
                    summary.pressure_files_sorted += 1
                    summary.pressure_rows_reordered += int(details["rows_reordered"])
                if details.get("ch58_rows_changed", 0):
                    summary.pressure_ch58_files_changed += 1
                summary.pressure_ch58_rows_replaced += int(details.get("ch58_rows_replaced", 0))
                summary.pressure_ch58_rows_changed += int(details.get("ch58_rows_changed", 0))
                summary.pressure_ch58_rows_failed += int(details.get("ch58_rows_failed", 0))
                if details.get("trimmed_64_to_20"):
                    summary.pressure_64ch_files_trimmed += 1
            except Exception as exc:  # pragma: no cover - reported in summary.
                summary.errors.append(f"{_rel(pressure_path)}: {exc}")

        gripper_path = session / "robot_state" / "gripper_state.csv"
        if gripper_path.is_file():
            summary.gripper_files_seen += 1
            try:
                change = clean_gripper_file(gripper_path, args.apply)
                summary.detail_files.append(change)
                details = change.details
                if change.status == "changed":
                    summary.gripper_files_changed += 1
                summary.gripper_duplicate_rows_removed += int(details.get("duplicate_rows_removed", 0))
                summary.gripper_sys_state_filled += int(details.get("sys_state_filled", 0))
            except Exception as exc:  # pragma: no cover - reported in summary.
                summary.errors.append(f"{_rel(gripper_path)}: {exc}")

    for session in sessions:
        data_format = detect_format(session)
        if data_format != "dual_realsense":
            continue
        aligned_path = session / "aligned_timesteps.csv"
        should_rebuild = args.rebuild_aligned == "all" or not aligned_path.is_file()
        if not should_rebuild:
            continue
        summary.aligned_files_seen += 1
        try:
            change = rebuild_alignment(session, args.apply)
            summary.detail_files.append(change)
            if change.status == "changed":
                summary.aligned_files_written += 1
                summary.aligned_rows_written += change.rows
                if change.details.get("created"):
                    summary.aligned_files_created += 1
        except Exception as exc:  # pragma: no cover - reported in summary.
            summary.errors.append(f"{_rel(aligned_path)}: {exc}")

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    manifest_csv, manifest_json, _payload = write_manifest(sessions, begin_end_names, out_dir)
    summary.manifest_csv = _rel(manifest_csv)
    summary.manifest_json = _rel(manifest_json)

    summary_json = out_dir / "cleaning_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    details_csv = out_dir / "cleaning_detail.csv"
    with open(details_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "status", "rows", "details"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary.detail_files:
            writer.writerow(
                {
                    "path": item.path,
                    "status": item.status,
                    "rows": item.rows,
                    "details": json.dumps(item.details, ensure_ascii=False, sort_keys=True),
                }
            )

    return summary


def print_summary(summary: CleaningSummary) -> None:
    mode = "APPLY" if summary.apply else "DRY-RUN"
    print(f"mode={mode} sessions={summary.sessions_seen}")
    print(
        "pressure: "
        f"files={summary.pressure_files_seen}, changed={summary.pressure_files_changed}, "
        f"sorted={summary.pressure_files_sorted}, reordered_rows={summary.pressure_rows_reordered}, "
        f"ch58_changed_files={summary.pressure_ch58_files_changed}, "
        f"ch58_rows_replaced={summary.pressure_ch58_rows_replaced}, "
        f"ch58_rows_changed={summary.pressure_ch58_rows_changed}, "
        f"ch58_rows_failed={summary.pressure_ch58_rows_failed}, "
        f"trimmed_64ch={summary.pressure_64ch_files_trimmed}"
    )
    print(
        "gripper: "
        f"files={summary.gripper_files_seen}, changed={summary.gripper_files_changed}, "
        f"duplicates_removed={summary.gripper_duplicate_rows_removed}, "
        f"sys_state_filled={summary.gripper_sys_state_filled}"
    )
    print(
        "alignment: "
        f"rebuild_candidates={summary.aligned_files_seen}, written={summary.aligned_files_written}, "
        f"created={summary.aligned_files_created}, rows_written={summary.aligned_rows_written}"
    )
    print(f"manifest: {summary.manifest_csv}")
    print(f"manifest_json: {summary.manifest_json}")
    if summary.errors:
        print("errors:")
        for item in summary.errors[:20]:
            print(f"  {item}")
        if len(summary.errors) > 20:
            print(f"  ... {len(summary.errors) - 20} more")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean TransVTLA session data and write split manifests.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(path) for path in DEFAULT_ROOTS],
        help="Session roots to scan.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/audit_reports"), help="Report output directory.")
    parser.add_argument("--apply", action="store_true", help="Rewrite data files after creating one-time backups.")
    parser.add_argument(
        "--rebuild-aligned",
        choices=("all", "missing", "none"),
        default="all",
        help="Which dual-RealSense aligned_timesteps.csv files to rebuild.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional session limit for testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_cleaning(args)
    print_summary(summary)
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
