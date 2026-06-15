#!/usr/bin/env python3
"""Build per-session multimodal sync indices for RLDS conversion.

The visual frame timeline is the canonical timeline. New sessions use
frames.csv:capture_monotonic_us. Legacy sessions without frames.csv fall back to
a 30 Hz timeline anchored to the first available host-side timestamp.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session_schema import (
    common_frame_stems,
    find_session_dirs,
    frames_csv_path,
    gripper_state_csv_path,
    pressure_csv_path,
    read_frame_records,
    resolve_session_layout,
    robot_state_csv_path,
    sync_dir,
    sync_index_path,
    sync_summary_path,
)

DEFAULT_VISUAL_FPS = 30
DEFAULT_MAX_DT_US = 50_000


@dataclass(frozen=True)
class TimeSeries:
    name: str
    timestamps: list[int]
    timestamp_column: str = ""

    @property
    def count(self) -> int:
        return len(self.timestamps)


def _safe_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _read_csv_timestamps(path: Path, preferred_columns: Iterable[str]) -> TimeSeries:
    if not path.is_file():
        return TimeSeries(path.stem, [], "")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        column = next((c for c in preferred_columns if c in header), "")
        if not column:
            return TimeSeries(path.stem, [], "")
        timestamps = []
        for row in reader:
            ts = _safe_int(row.get(column))
            if ts is not None:
                timestamps.append(ts)
    return TimeSeries(path.stem, timestamps, column)


def _nearest(ts: int, series: TimeSeries, max_dt_us: int) -> tuple[int, int, int, bool]:
    if not series.timestamps:
        return -1, 0, 0, False
    idx = bisect.bisect_left(series.timestamps, ts)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(series.timestamps):
        candidates.append(idx)
    best = min(candidates, key=lambda i: abs(series.timestamps[i] - ts))
    source_ts = series.timestamps[best]
    dt = source_ts - ts
    return best, source_ts, dt, abs(dt) <= max_dt_us


def _frame_timeline(session_dir: Path, stems: list[str], fps: int) -> tuple[list[int], list[int], str]:
    records = read_frame_records(session_dir)
    timestamps: list[int] = []
    frame_ids: list[int] = []
    for i, row in enumerate(records):
        ts = _safe_int(row.get("capture_monotonic_us"))
        if ts is None:
            continue
        frame_id = _safe_int(row.get("frame_id"))
        timestamps.append(ts)
        frame_ids.append(frame_id if frame_id is not None else i + 1)
    if timestamps:
        return frame_ids, timestamps, "frames_csv"

    robot = _read_csv_timestamps(robot_state_csv_path(session_dir), ["timestamp_us"])
    pressure = _read_csv_timestamps(
        pressure_csv_path(session_dir),
        ["host_monotonic_us", "timestamp_us", "sensor_timestamp_us"],
    )
    gripper = _read_csv_timestamps(gripper_state_csv_path(session_dir), ["timestamp_us"])

    anchor = 0
    for series in (robot, pressure, gripper):
        if series.timestamps:
            anchor = series.timestamps[0]
            break
    interval_us = int(1_000_000 / fps)
    count = len(stems)
    return (
        list(range(1, count + 1)),
        [anchor + i * interval_us for i in range(count)],
        "fallback_30hz",
    )


def build_session_sync(
    session_dir: Path,
    fps: int = DEFAULT_VISUAL_FPS,
    max_dt_us: int = DEFAULT_MAX_DT_US,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    layout = resolve_session_layout(session_dir)
    stems = common_frame_stems(layout)
    frame_ids, frame_ts, sync_source = _frame_timeline(session_dir, stems, fps)

    usable = min(len(stems), len(frame_ts), len(frame_ids))
    stems = stems[:usable]
    frame_ids = frame_ids[:usable]
    frame_ts = frame_ts[:usable]

    pressure = _read_csv_timestamps(
        pressure_csv_path(session_dir),
        ["host_monotonic_us", "timestamp_us", "sensor_timestamp_us"],
    )
    robot = _read_csv_timestamps(robot_state_csv_path(session_dir), ["timestamp_us"])
    gripper = _read_csv_timestamps(gripper_state_csv_path(session_dir), ["timestamp_us"])

    rows: list[dict[str, object]] = []
    deltas = {"pressure": [], "robot": [], "gripper": []}
    valids = {"pressure": 0, "robot": 0, "gripper": 0}

    for frame_id, stem, ts in zip(frame_ids, stems, frame_ts):
        p_idx, p_ts, p_dt, p_valid = _nearest(ts, pressure, max_dt_us)
        r_idx, r_ts, r_dt, r_valid = _nearest(ts, robot, max_dt_us)
        g_idx, g_ts, g_dt, g_valid = _nearest(ts, gripper, max_dt_us)

        for name, dt, valid in (
            ("pressure", p_dt, p_valid),
            ("robot", r_dt, r_valid),
            ("gripper", g_dt, g_valid),
        ):
            if valid:
                valids[name] += 1
            if dt:
                deltas[name].append(abs(dt))

        rows.append({
            "frame_id": frame_id,
            "frame_stem": stem,
            "capture_monotonic_us": ts,
            "pressure_index": p_idx,
            "pressure_timestamp_us": p_ts,
            "pressure_dt_us": p_dt,
            "pressure_valid": int(p_valid),
            "robot_index": r_idx,
            "robot_timestamp_us": r_ts,
            "robot_dt_us": r_dt,
            "robot_valid": int(r_valid),
            "gripper_index": g_idx,
            "gripper_timestamp_us": g_ts,
            "gripper_dt_us": g_dt,
            "gripper_valid": int(g_valid),
        })

    def delta_summary(name: str) -> dict[str, object]:
        values = deltas[name]
        return {
            "valid_frames": valids[name],
            "mean_abs_dt_us": round(mean(values), 3) if values else None,
            "max_abs_dt_us": max(values) if values else None,
        }

    summary = {
        "schema": "transvtla_sync_index/v1",
        "session": str(session_dir),
        "sync_source": sync_source,
        "frames_csv": str(frames_csv_path(session_dir)),
        "visual_fps": fps,
        "max_dt_us": max_dt_us,
        "frames": len(rows),
        "modalities": {
            "pressure": {
                "rows": pressure.count,
                "timestamp_column": pressure.timestamp_column,
                **delta_summary("pressure"),
            },
            "robot": {
                "rows": robot.count,
                "timestamp_column": robot.timestamp_column,
                **delta_summary("robot"),
            },
            "gripper": {
                "rows": gripper.count,
                "timestamp_column": gripper.timestamp_column,
                **delta_summary("gripper"),
            },
        },
    }
    return rows, summary


def write_sync(session_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    out_dir = sync_dir(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(sync_index_path(session_dir), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(sync_index_path(session_dir), "w", newline="", encoding="utf-8") as f:
            f.write("")
    with open(sync_summary_path(session_dir), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build standard per-frame multimodal sync indices.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing sessions.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write sync files.")
    parser.add_argument("--fps", type=int, default=DEFAULT_VISUAL_FPS, help="Fallback visual FPS.")
    parser.add_argument("--max-dt-us", type=int, default=DEFAULT_MAX_DT_US, help="Validity threshold.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of sessions to process.")
    args = parser.parse_args()

    sessions = find_session_dirs(args.data_root)
    if args.limit is not None:
        sessions = sessions[: args.limit]
    print(f"mode={'dry-run' if args.dry_run else 'write'} sessions={len(sessions)} root={args.data_root}")

    failures = 0
    for session_dir in sessions:
        try:
            rows, summary = build_session_sync(session_dir, fps=args.fps, max_dt_us=args.max_dt_us)
        except Exception as exc:
            failures += 1
            print(f"[error] {session_dir.relative_to(args.data_root)} {exc}")
            continue

        rel = session_dir.relative_to(args.data_root)
        p = summary["modalities"]["pressure"]
        r = summary["modalities"]["robot"]
        g = summary["modalities"]["gripper"]
        print(
            f"[sync] {rel} frames={summary['frames']} source={summary['sync_source']} "
            f"valid pressure/robot/gripper={p['valid_frames']}/{r['valid_frames']}/{g['valid_frames']}"
        )
        if not args.dry_run:
            write_sync(session_dir, rows, summary)

    print(f"failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
