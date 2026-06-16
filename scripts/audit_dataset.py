#!/usr/bin/env python3
"""Audit TransVTLA real-data sessions across legacy and dual-RealSense layouts.

The script is intentionally read-only. It summarizes every session, audits
modality availability, frame/sample counts, inferred sampling rates, timestamp
health, and writes a cleaning plan for issues that can be fixed safely later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ROOTS = [
    Path("dataset/phase2_realdata_sessions/sessions"),
    Path("dataset/sessions"),
]

BEGIN_MARKER = "session_20260615_193300_begin"
END_MARKER = "session_20260615_201528_end"
DATE_RE = re.compile(r"session_(\d{8})_(\d{6})")

PRESSURE_MIN_ADC = 0.0
PRESSURE_MAX_ADC = 5007.0
MAX_SYNC_OFFSET_MS = 50.0


IMAGE_MODALITIES = {
    "world_rgb": ("world_camera/rgb", ".jpg", "raw"),
    "world_depth": ("world_camera/depth", ".png", "raw"),
    "wrist_rgb": ("wrist_camera/rgb", ".jpg", "raw"),
    "wrist_depth": ("wrist_camera/depth", ".png", "raw"),
    "dji_rgb": ("dji", ".jpg", "raw"),
    "dji_depth": ("dji_depth", ".png", "derived"),
    "legacy_realsense_rgb": ("realsense_rgb", ".jpg", "raw"),
    "legacy_realsense_depth": ("realsense_depth", ".png", "raw"),
}

CSV_MODALITIES = {
    "frames": ("frames.csv", ("capture_monotonic_us",)),
    "aligned_timesteps": ("aligned_timesteps.csv", ("visual_timestamp_us",)),
    "pressure": ("pressure/pressure.csv", ("host_monotonic_us", "timestamp_us", "sensor_timestamp_us")),
    "robot_state": ("robot_state/robot_state.csv", ("timestamp_us",)),
    "gripper_state": ("robot_state/gripper_state.csv", ("timestamp_us",)),
}

JSON_MODALITIES = {
    "camera_metadata": "camera_metadata.json",
    "sync_summary": "sync/sync_summary.json",
}

REQUIRED_BY_FORMAT = {
    "dual_realsense": [
        "world_rgb",
        "world_depth",
        "wrist_rgb",
        "wrist_depth",
        "frames",
        "camera_metadata",
        "pressure",
        "robot_state",
        "gripper_state",
    ],
    "legacy_dji_realsense": [
        "dji_rgb",
        "legacy_realsense_rgb",
        "legacy_realsense_depth",
        "pressure",
        "robot_state",
        "gripper_state",
    ],
    "legacy_dji_realsense_rgb_only": [
        "dji_rgb",
        "legacy_realsense_rgb",
        "pressure",
        "robot_state",
        "gripper_state",
    ],
}


@dataclass
class Issue:
    severity: str
    session: str
    rel_path: str
    phase: str
    view_tag: str
    modality: str
    code: str
    message: str
    suggestion: str = ""


@dataclass
class ModalityAudit:
    session: str
    rel_path: str
    phase: str
    view_tag: str
    data_format: str
    modality: str
    kind: str
    source_path: str
    present: bool
    count: int = 0
    duration_s: float | None = None
    hz_mean: float | None = None
    hz_median: float | None = None
    timestamp_column: str = ""
    min_timestamp_us: int | None = None
    max_timestamp_us: int | None = None
    duplicate_timestamps: int = 0
    timestamp_reversals: int = 0
    empty_cells: int = 0
    zero_size_files: int = 0
    min_frame_id: int | None = None
    max_frame_id: int | None = None
    missing_sequence_items: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionAudit:
    session: str
    path: str
    rel_path: str
    root: str
    date: str
    time: str
    phase: str
    data_format: str
    view_tag: str
    world_camera: str
    wrist_camera: str
    session_duration_s: float | None = None
    visual_complete_frames: int = 0
    visual_hz: float | None = None
    visual_hz_source: str = ""
    modalities_present: list[str] = field(default_factory=list)
    issue_counts: dict[str, int] = field(default_factory=dict)
    cleaning_actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    modalities: dict[str, ModalityAudit] = field(default_factory=dict)


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
        v = float(str(value))
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_session_datetime(name: str) -> tuple[str, str]:
    match = DATE_RE.search(name)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def discover_sessions(roots: Iterable[Path]) -> list[Path]:
    sessions: set[Path] = set()
    for raw_root in roots:
        root = raw_root if raw_root.is_absolute() else REPO_ROOT / raw_root
        if not root.exists():
            continue
        candidates = [root] if root.name.startswith("session_") else list(root.rglob("session_*"))
        for candidate in candidates:
            if not candidate.is_dir() or not candidate.name.startswith("session_"):
                continue
            if _looks_like_session(candidate):
                sessions.add(candidate.resolve())
    return sorted(sessions, key=lambda p: str(p))


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


def build_begin_end_set(sessions: Iterable[Path]) -> set[str]:
    by_parent: dict[Path, list[Path]] = defaultdict(list)
    for session in sessions:
        by_parent[session.parent].append(session)

    names: set[str] = set()
    for group in by_parent.values():
        ordered = sorted(group, key=lambda p: p.name)
        ordered_names = [p.name for p in ordered]
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


def classify_session(session: Path, begin_end_names: set[str]) -> tuple[str, str, str, str]:
    date, _time = _parse_session_datetime(session.name)
    data_format = detect_format(session)

    if date and date < "20260615":
        phase = "pre_0615_dji_world"
    elif data_format == "legacy_dji_realsense":
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

    if data_format in {"legacy_dji_realsense", "legacy_dji_realsense_rgb_only"}:
        world_camera = "DJI world RGB"
        wrist_camera = "RealSense wrist RGB-D" if data_format == "legacy_dji_realsense" else "RealSense wrist RGB"
    elif data_format == "dual_realsense":
        world_camera = "RealSense world RGB-D"
        wrist_camera = "RealSense wrist RGB-D"
    else:
        world_camera = "unknown"
        wrist_camera = "unknown"

    return phase, view_tag, world_camera, wrist_camera


def _timestamp_stats(timestamps: list[int]) -> dict[str, Any]:
    clean = [t for t in timestamps if t is not None]
    if len(clean) < 2:
        return {
            "duration_s": None,
            "hz_mean": None,
            "hz_median": None,
            "min_timestamp_us": clean[0] if clean else None,
            "max_timestamp_us": clean[0] if clean else None,
            "duplicate_timestamps": max(0, len(clean) - len(set(clean))),
            "timestamp_reversals": 0,
            "positive_deltas_us": [],
        }

    duplicate_timestamps = len(clean) - len(set(clean))
    reversals = sum(1 for a, b in zip(clean, clean[1:]) if b < a)
    min_ts = min(clean)
    max_ts = max(clean)
    duration_s = (max_ts - min_ts) / 1_000_000 if max_ts > min_ts else None
    positive_deltas = [b - a for a, b in zip(clean, clean[1:]) if b > a]
    hz_mean = (len(clean) - 1) / duration_s if duration_s and duration_s > 0 else None
    hz_median = None
    if positive_deltas:
        median_dt = statistics.median(positive_deltas)
        if median_dt > 0:
            hz_median = 1_000_000 / median_dt

    return {
        "duration_s": duration_s,
        "hz_mean": hz_mean,
        "hz_median": hz_median,
        "min_timestamp_us": min_ts,
        "max_timestamp_us": max_ts,
        "duplicate_timestamps": duplicate_timestamps,
        "timestamp_reversals": reversals,
        "positive_deltas_us": positive_deltas,
    }


def _external_rate(count: int, duration_s: float | None) -> float | None:
    if count <= 0 or not duration_s or duration_s <= 0:
        return None
    return count / duration_s


def _rate_threshold(modality: str, data_format: str) -> tuple[float | None, float | None]:
    visual = {
        "world_rgb",
        "world_depth",
        "wrist_rgb",
        "wrist_depth",
        "dji_rgb",
        "dji_depth",
        "legacy_realsense_rgb",
        "legacy_realsense_depth",
        "frames",
        "aligned_timesteps",
    }
    if modality in visual:
        if data_format == "legacy_dji_realsense":
            return 6.0, 8.0
        return 20.0, 30.0
    if modality == "pressure":
        return 80.0, 200.0
    if modality in {"robot_state", "gripper_state"}:
        return 10.0, 100.0
    return None, None


def _make_issue(
    session: SessionAudit,
    modality: str,
    severity: str,
    code: str,
    message: str,
    suggestion: str = "",
) -> Issue:
    return Issue(
        severity=severity,
        session=session.session,
        rel_path=session.rel_path,
        phase=session.phase,
        view_tag=session.view_tag,
        modality=modality,
        code=code,
        message=message,
        suggestion=suggestion,
    )


def audit_csv_modality(
    session: SessionAudit,
    session_path: Path,
    modality: str,
    rel_file: str,
    preferred_ts: tuple[str, ...],
    issues: list[Issue],
) -> ModalityAudit:
    path = session_path / rel_file
    audit = ModalityAudit(
        session=session.session,
        rel_path=session.rel_path,
        phase=session.phase,
        view_tag=session.view_tag,
        data_format=session.data_format,
        modality=modality,
        kind="csv",
        source_path=_rel(path),
        present=path.is_file(),
    )
    if not audit.present:
        return audit

    timestamps: list[int] = []
    channel_min: dict[str, float] = {}
    channel_max: dict[str, float] = {}
    pressure_out_of_range = 0
    pressure_out_columns: Counter[str] = Counter()
    max_abs_offset_ms: float | None = None
    rows_over_sync_limit = 0
    gripper_target_hz: list[float] = []
    disallowed_empty_cells = 0
    allowed_empty_cells = 0

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            audit.extra["columns"] = len(header)
            audit.extra["header"] = header
            ts_col = next((c for c in preferred_ts if c in header), "")
            audit.timestamp_column = ts_col
            channel_cols = [c for c in header if c.startswith("CH")]
            offset_cols = [c for c in header if c.endswith("_offset_ms")]

            for row in reader:
                audit.count += 1

                if ts_col:
                    ts = _safe_int(row.get(ts_col))
                    if ts is not None:
                        timestamps.append(ts)

                if modality == "gripper_state":
                    target = _safe_float(row.get("target_hz"))
                    if target is not None:
                        gripper_target_hz.append(target)

                for col in header:
                    value = row.get(col)
                    if value != "":
                        continue
                    if _is_allowed_empty(modality, row, col):
                        allowed_empty_cells += 1
                    else:
                        disallowed_empty_cells += 1

                if modality == "pressure":
                    for col in channel_cols:
                        value = _safe_float(row.get(col))
                        if value is None:
                            continue
                        channel_min[col] = min(channel_min.get(col, value), value)
                        channel_max[col] = max(channel_max.get(col, value), value)
                        if value < PRESSURE_MIN_ADC or value > PRESSURE_MAX_ADC:
                            pressure_out_of_range += 1
                            pressure_out_columns[col] += 1

                if modality == "aligned_timesteps":
                    row_max = 0.0
                    for col in offset_cols:
                        value = _safe_float(row.get(col))
                        if value is None:
                            continue
                        row_max = max(row_max, abs(value))
                    if row_max:
                        max_abs_offset_ms = max(max_abs_offset_ms or 0.0, row_max)
                        if row_max > MAX_SYNC_OFFSET_MS:
                            rows_over_sync_limit += 1

    except UnicodeDecodeError as exc:
        issues.append(_make_issue(session, modality, "error", "csv_decode_error", str(exc), "Check file encoding."))
        return audit
    except csv.Error as exc:
        issues.append(_make_issue(session, modality, "error", "csv_parse_error", str(exc), "Inspect or regenerate CSV."))
        return audit

    audit.empty_cells = disallowed_empty_cells
    audit.extra["allowed_empty_cells"] = allowed_empty_cells

    if timestamps:
        stats = _timestamp_stats(timestamps)
        audit.duration_s = _round(stats["duration_s"])
        audit.hz_mean = _round(stats["hz_mean"])
        audit.hz_median = _round(stats["hz_median"])
        audit.min_timestamp_us = stats["min_timestamp_us"]
        audit.max_timestamp_us = stats["max_timestamp_us"]
        audit.duplicate_timestamps = stats["duplicate_timestamps"]
        audit.timestamp_reversals = stats["timestamp_reversals"]
        if stats["positive_deltas_us"]:
            audit.extra["max_delta_ms"] = _round(max(stats["positive_deltas_us"]) / 1000.0)
            audit.extra["median_delta_ms"] = _round(statistics.median(stats["positive_deltas_us"]) / 1000.0)
    elif audit.count > 0 and preferred_ts:
        issues.append(
            _make_issue(
                session,
                modality,
                "warn",
                "missing_timestamp_column",
                f"No usable timestamp column from {preferred_ts}.",
                "Check CSV schema or regenerate the modality file.",
            )
        )

    if modality == "pressure":
        channel_cols = [c for c in audit.extra.get("header", []) if c.startswith("CH")]
        audit.extra["pressure_channel_count"] = len(channel_cols)
        audit.extra["pressure_min_adc"] = min(channel_min.values()) if channel_min else None
        audit.extra["pressure_max_adc"] = max(channel_max.values()) if channel_max else None
        audit.extra["pressure_out_of_range_values"] = pressure_out_of_range
        audit.extra["pressure_out_of_range_columns"] = dict(pressure_out_columns.most_common())

    if modality == "aligned_timesteps":
        audit.extra["max_abs_offset_ms"] = _round(max_abs_offset_ms)
        audit.extra["rows_over_50ms_offset"] = rows_over_sync_limit

    if modality == "gripper_state" and gripper_target_hz:
        audit.extra["target_hz_median"] = _round(statistics.median(gripper_target_hz))

    _add_csv_issues(session, audit, issues)
    return audit


def _is_allowed_empty(modality: str, row: dict[str, str], column: str) -> bool:
    if modality != "gripper_state":
        return False
    code = row.get("rm_plus_read_code")
    if code != "-2":
        return False
    return column.startswith("gripper_") or column == "sys_state"


def _add_csv_issues(session: SessionAudit, audit: ModalityAudit, issues: list[Issue]) -> None:
    if audit.present and audit.count == 0:
        issues.append(
            _make_issue(
                session,
                audit.modality,
                "error",
                "empty_csv",
                "CSV has a header but no data rows.",
                "Drop the session or recollect this modality.",
            )
        )

    if audit.timestamp_reversals:
        suggestion = "Sort rows by timestamp after backing up the original CSV."
        if audit.modality == "pressure" and "phase2_realdata_sessions" in audit.rel_path:
            suggestion = "Run scripts/fix_f1_pressure_sort.py --dry-run first, then apply a reviewed sort."
        issues.append(
            _make_issue(
                session,
                audit.modality,
                "warn",
                "non_monotonic_timestamps",
                f"{audit.timestamp_reversals} timestamp reversals found.",
                suggestion,
            )
        )

    if audit.duplicate_timestamps:
        suggestion = "Deduplicate equal timestamps after choosing a keep policy."
        if audit.modality == "gripper_state":
            suggestion = "Run scripts/fix_f2_f3_gripper.py in dry-run mode, then review dedupe output."
        issues.append(
            _make_issue(
                session,
                audit.modality,
                "warn",
                "duplicate_timestamps",
                f"{audit.duplicate_timestamps} duplicate timestamps found.",
                suggestion,
            )
        )

    if audit.empty_cells:
        issues.append(
            _make_issue(
                session,
                audit.modality,
                "warn",
                "empty_cells",
                f"{audit.empty_cells} non-allowed empty cells found.",
                "Fill only if the missing value is semantically recoverable; otherwise mark/drop affected rows.",
            )
        )

    min_hz, _target_hz = _rate_threshold(audit.modality, audit.data_format)
    if min_hz is not None and audit.hz_mean is not None and audit.hz_mean < min_hz:
        issues.append(
            _make_issue(
                session,
                audit.modality,
                "warn",
                "low_frequency",
                f"Observed {audit.hz_mean:.2f} Hz, below minimum {min_hz:.2f} Hz.",
                "Inspect dropped samples and consider excluding this session for timing-sensitive training.",
            )
        )

    if audit.modality == "pressure":
        channel_count = audit.extra.get("pressure_channel_count")
        if channel_count not in (20, 64):
            issues.append(
                _make_issue(
                    session,
                    audit.modality,
                    "warn",
                    "unexpected_pressure_channel_count",
                    f"Pressure CSV has {channel_count} CH columns, expected 20 or 64.",
                    "Inspect pressure schema before preprocessing.",
                )
            )
        elif channel_count == 64:
            issues.append(
                _make_issue(
                    session,
                    audit.modality,
                    "info",
                    "legacy_pressure_64ch",
                    "Pressure CSV uses legacy 64-channel layout.",
                    "Use scripts/trim_pressure_channels.py --apply after verifying backups.",
                )
            )

        bad_values = audit.extra.get("pressure_out_of_range_values") or 0
        if bad_values:
            issues.append(
                _make_issue(
                    session,
                    audit.modality,
                    "warn",
                    "pressure_adc_out_of_range",
                    f"{bad_values} pressure ADC values outside [{PRESSURE_MIN_ADC}, {PRESSURE_MAX_ADC}].",
                    "Prefer cleaning during preprocessing; keep raw CSV backed up.",
                )
            )

    if audit.modality == "aligned_timesteps":
        rows_over = audit.extra.get("rows_over_50ms_offset") or 0
        if rows_over:
            issues.append(
                _make_issue(
                    session,
                    audit.modality,
                    "warn",
                    "large_sync_offset",
                    f"{rows_over} aligned rows have modality offset over {MAX_SYNC_OFFSET_MS:.0f} ms.",
                    "Rebuild sync/alignment and inspect dropped or stale modality samples.",
                )
            )


def audit_image_modality(
    session: SessionAudit,
    session_path: Path,
    modality: str,
    rel_dir: str,
    suffix: str,
    kind: str,
    reference_duration_s: float | None,
    check_zero_size: bool,
    issues: list[Issue],
) -> ModalityAudit:
    path = session_path / rel_dir
    audit = ModalityAudit(
        session=session.session,
        rel_path=session.rel_path,
        phase=session.phase,
        view_tag=session.view_tag,
        data_format=session.data_format,
        modality=modality,
        kind=f"image_{kind}",
        source_path=_rel(path),
        present=path.is_dir(),
    )
    if not audit.present:
        return audit

    files = sorted(p for p in path.glob(f"*{suffix}") if p.is_file())
    audit.count = len(files)
    audit.duration_s = reference_duration_s
    audit.hz_mean = _round(_external_rate(audit.count, reference_duration_s))

    if check_zero_size:
        audit.zero_size_files = sum(1 for p in files if p.stat().st_size == 0)

    numeric_ids = [_safe_int(p.stem) for p in files]
    numeric_ids = [n for n in numeric_ids if n is not None]
    if numeric_ids and len(numeric_ids) == len(files):
        audit.min_frame_id = min(numeric_ids)
        audit.max_frame_id = max(numeric_ids)
        expected = audit.max_frame_id - audit.min_frame_id + 1
        audit.missing_sequence_items = max(0, expected - len(set(numeric_ids)))

    _add_image_issues(session, audit, issues)
    return audit


def _add_image_issues(session: SessionAudit, audit: ModalityAudit, issues: list[Issue]) -> None:
    if audit.present and audit.count == 0:
        issues.append(
            _make_issue(
                session,
                audit.modality,
                "error",
                "empty_image_dir",
                "Image directory exists but has no matching frames.",
                "Drop the session or recollect this modality.",
            )
        )

    if audit.zero_size_files:
        issues.append(
            _make_issue(
                session,
                audit.modality,
                "error",
                "zero_size_files",
                f"{audit.zero_size_files} zero-byte image files found.",
                "Remove or regenerate corrupt files; otherwise drop affected frames/session.",
            )
        )

    if audit.missing_sequence_items:
        issues.append(
            _make_issue(
                session,
                audit.modality,
                "warn",
                "missing_frame_ids",
                f"{audit.missing_sequence_items} frame ids missing in numeric sequence.",
                "Use complete-frame intersection for training or drop incomplete frames.",
            )
        )

    min_hz, _target_hz = _rate_threshold(audit.modality, audit.data_format)
    if min_hz is not None and audit.hz_mean is not None and audit.hz_mean < min_hz:
        issues.append(
            _make_issue(
                session,
                audit.modality,
                "warn",
                "low_frequency",
                f"Estimated {audit.hz_mean:.2f} Hz, below minimum {min_hz:.2f} Hz.",
                "Inspect dropped frames; mark low-rate sessions for separate evaluation.",
            )
        )


def audit_json_modality(
    session: SessionAudit,
    session_path: Path,
    modality: str,
    rel_file: str,
    issues: list[Issue],
) -> ModalityAudit:
    path = session_path / rel_file
    audit = ModalityAudit(
        session=session.session,
        rel_path=session.rel_path,
        phase=session.phase,
        view_tag=session.view_tag,
        data_format=session.data_format,
        modality=modality,
        kind="json",
        source_path=_rel(path),
        present=path.is_file(),
        count=1 if path.is_file() else 0,
    )
    if not audit.present:
        return audit

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        issues.append(
            _make_issue(
                session,
                modality,
                "error",
                "json_parse_error",
                f"{exc}",
                "Regenerate or repair this metadata JSON.",
            )
        )
        return audit

    if isinstance(payload, dict):
        audit.extra["schema"] = payload.get("schema", "")
        if modality == "camera_metadata":
            cameras = payload.get("cameras")
            if isinstance(cameras, dict):
                audit.extra["camera_count"] = len(cameras)
                audit.extra["camera_names"] = sorted(cameras.keys())
            profile = payload.get("standard_profile")
            if isinstance(profile, dict):
                audit.extra["standard_profile"] = profile
        if modality == "sync_summary":
            audit.extra["sync_source"] = payload.get("sync_source")
            audit.extra["frames"] = payload.get("frames")
    return audit


def complete_visual_count(modalities: dict[str, ModalityAudit], data_format: str) -> int:
    if data_format == "dual_realsense":
        required = ["world_rgb", "world_depth", "wrist_rgb", "wrist_depth"]
    elif data_format == "legacy_dji_realsense":
        required = ["dji_rgb", "legacy_realsense_rgb", "legacy_realsense_depth"]
    elif data_format == "legacy_dji_realsense_rgb_only":
        required = ["dji_rgb", "legacy_realsense_rgb"]
    else:
        required = []
    counts = [modalities[m].count for m in required if m in modalities and modalities[m].present]
    return min(counts) if len(counts) == len(required) and counts else 0


def add_session_level_issues(session: SessionAudit, issues: list[Issue]) -> None:
    required = REQUIRED_BY_FORMAT.get(session.data_format, [])
    for modality in required:
        audit = session.modalities.get(modality)
        if audit is None or not audit.present:
            issues.append(
                _make_issue(
                    session,
                    modality,
                    "error",
                    "missing_required_modality",
                    f"Required modality '{modality}' is missing.",
                    "Drop this session or recollect the missing modality.",
                )
            )

    if session.data_format == "unknown":
        issues.append(
            _make_issue(
                session,
                "session",
                "error",
                "unknown_session_layout",
                "Session does not match legacy DJI/RealSense or dual RealSense layout.",
                "Inspect directory structure and update the audit script if this is a new schema.",
            )
        )

    if session.data_format == "dual_realsense":
        aligned = session.modalities.get("aligned_timesteps")
        if aligned is None or not aligned.present:
            issues.append(
                _make_issue(
                    session,
                    "aligned_timesteps",
                    "warn",
                    "missing_aligned_timesteps",
                    "Dual RealSense session has no aligned_timesteps.csv.",
                    "Build/rebuild a sync index before training if per-frame aligned rows are required.",
                )
            )

    if session.visual_complete_frames == 0:
        issues.append(
            _make_issue(
                session,
                "visual",
                "error",
                "no_complete_visual_frames",
                "No complete frame intersection across required visual streams.",
                "Drop this session or inspect missing/corrupt camera files.",
            )
        )

    if session.data_format == "dual_realsense":
        counts = {
            key: session.modalities[key].count
            for key in ("world_rgb", "world_depth", "wrist_rgb", "wrist_depth")
            if key in session.modalities and session.modalities[key].present
        }
        if len(set(counts.values())) > 1:
            issues.append(
                _make_issue(
                    session,
                    "visual",
                    "warn",
                    "visual_count_mismatch",
                    f"Dual RealSense stream counts differ: {counts}.",
                    "Use complete-frame intersection or regenerate dropped stream files.",
                )
            )

        frames = session.modalities.get("frames")
        if frames is not None and frames.present and frames.count != session.visual_complete_frames:
            issues.append(
                _make_issue(
                    session,
                    "frames",
                    "warn",
                    "frames_csv_count_mismatch",
                    (
                        f"frames.csv rows={frames.count} but complete visual frames="
                        f"{session.visual_complete_frames}."
                    ),
                    "Use frames.csv as the canonical visual timeline or rebuild missing frame metadata.",
                )
            )

    if session.data_format == "legacy_dji_realsense":
        counts = {
            key: session.modalities[key].count
            for key in ("dji_rgb", "legacy_realsense_rgb", "legacy_realsense_depth")
            if key in session.modalities and session.modalities[key].present
        }
        if len(set(counts.values())) > 1:
            issues.append(
                _make_issue(
                    session,
                    "visual",
                    "warn",
                    "visual_count_mismatch",
                    f"Legacy stream counts differ: {counts}.",
                    "Use complete-frame intersection or rebuild generated depth for missing frames.",
                )
            )

    if session.data_format == "legacy_dji_realsense_rgb_only":
        counts = {
            key: session.modalities[key].count
            for key in ("dji_rgb", "legacy_realsense_rgb")
            if key in session.modalities and session.modalities[key].present
        }
        if len(set(counts.values())) > 1:
            issues.append(
                _make_issue(
                    session,
                    "visual",
                    "warn",
                    "visual_count_mismatch",
                    f"Legacy RGB-only stream counts differ: {counts}.",
                    "Use complete-frame intersection or inspect dropped camera files.",
                )
            )

    if session.view_tag == "0615_initial_realsense_position_incomplete_arm_view":
        issues.append(
            _make_issue(
                session,
                "view",
                "info",
                "known_incomplete_arm_view",
                "Known 2026-06-15 begin-end camera position cannot see the full robot arm.",
                "Keep this position as a separate split/tag; do not mix with repositioned RealSense data blindly.",
            )
        )


def audit_session(
    session_path: Path,
    root: Path,
    begin_end_names: set[str],
    check_zero_size: bool,
) -> tuple[SessionAudit, list[Issue]]:
    date, time_part = _parse_session_datetime(session_path.name)
    data_format = detect_format(session_path)
    phase, view_tag, world_camera, wrist_camera = classify_session(session_path, begin_end_names)
    session = SessionAudit(
        session=session_path.name,
        path=str(session_path),
        rel_path=_rel(session_path),
        root=_rel(root),
        date=date,
        time=time_part,
        phase=phase,
        data_format=data_format,
        view_tag=view_tag,
        world_camera=world_camera,
        wrist_camera=wrist_camera,
    )
    issues: list[Issue] = []

    for modality, (rel_file, preferred_ts) in CSV_MODALITIES.items():
        session.modalities[modality] = audit_csv_modality(
            session, session_path, modality, rel_file, preferred_ts, issues
        )

    for modality, rel_file in JSON_MODALITIES.items():
        session.modalities[modality] = audit_json_modality(session, session_path, modality, rel_file, issues)

    reference_duration = _choose_session_duration(session.modalities)
    session.session_duration_s = _round(reference_duration)

    for modality, (rel_dir, suffix, kind) in IMAGE_MODALITIES.items():
        session.modalities[modality] = audit_image_modality(
            session,
            session_path,
            modality,
            rel_dir,
            suffix,
            kind,
            reference_duration,
            check_zero_size,
            issues,
        )

    session.visual_complete_frames = complete_visual_count(session.modalities, data_format)
    if data_format == "dual_realsense":
        frames = session.modalities.get("frames")
        session.visual_hz = frames.hz_mean if frames and frames.hz_mean is not None else None
        session.visual_hz_source = "frames.csv:capture_monotonic_us" if session.visual_hz is not None else ""
    if session.visual_hz is None:
        session.visual_hz = _round(_external_rate(session.visual_complete_frames, reference_duration))
        session.visual_hz_source = "complete_visual_count/session_duration" if session.visual_hz is not None else ""

    session.modalities_present = sorted([k for k, v in session.modalities.items() if v.present])
    add_session_level_issues(session, issues)
    session.issue_counts = dict(Counter(issue.severity for issue in issues))
    session.cleaning_actions = sorted({issue.suggestion for issue in issues if issue.suggestion})
    return session, issues


def _choose_session_duration(modalities: dict[str, ModalityAudit]) -> float | None:
    frames = modalities.get("frames")
    if frames and frames.duration_s:
        return frames.duration_s

    aligned = modalities.get("aligned_timesteps")
    if aligned and aligned.duration_s:
        return aligned.duration_s

    candidates = []
    for key in ("pressure", "robot_state", "gripper_state"):
        audit = modalities.get(key)
        if audit and audit.duration_s:
            candidates.append(audit.duration_s)
    if not candidates:
        return None
    return max(candidates)


def _group_roots_for_sessions(sessions: list[Path], roots: list[Path]) -> dict[Path, Path]:
    resolved_roots = [(r if r.is_absolute() else REPO_ROOT / r).resolve() for r in roots if (r if r.is_absolute() else REPO_ROOT / r).exists()]
    mapping: dict[Path, Path] = {}
    for session in sessions:
        best = None
        for root in resolved_roots:
            try:
                session.relative_to(root)
            except ValueError:
                continue
            if best is None or len(str(root)) > len(str(best)):
                best = root
        mapping[session] = best or session.parent
    return mapping


def summarize(sessions: list[SessionAudit], issues: list[Issue], modalities: list[ModalityAudit]) -> dict[str, Any]:
    issue_counts = Counter(issue.code for issue in issues)
    severity_counts = Counter(issue.severity for issue in issues)

    modality_summary: dict[str, dict[str, Any]] = {}
    by_modality: dict[str, list[ModalityAudit]] = defaultdict(list)
    for modality in modalities:
        if modality.present:
            by_modality[modality.modality].append(modality)

    for name, rows in sorted(by_modality.items()):
        hz_values = [r.hz_mean for r in rows if r.hz_mean is not None]
        counts = [r.count for r in rows]
        modality_summary[name] = {
            "sessions_present": len(rows),
            "total_count": sum(counts),
            "min_count": min(counts) if counts else None,
            "max_count": max(counts) if counts else None,
            "mean_count": _round(statistics.mean(counts)) if counts else None,
            "mean_hz": _round(statistics.mean(hz_values)) if hz_values else None,
            "min_hz": _round(min(hz_values)) if hz_values else None,
            "max_hz": _round(max(hz_values)) if hz_values else None,
        }

    return {
        "schema": "transvtla_dataset_audit/v1",
        "repo_root": str(REPO_ROOT),
        "total_sessions": len(sessions),
        "by_root": dict(Counter(s.root for s in sessions)),
        "by_phase": dict(Counter(s.phase for s in sessions)),
        "by_format": dict(Counter(s.data_format for s in sessions)),
        "by_view_tag": dict(Counter(s.view_tag for s in sessions)),
        "visual_complete_frames_total": sum(s.visual_complete_frames for s in sessions),
        "issue_counts_by_severity": dict(severity_counts),
        "issue_counts_by_code": dict(issue_counts.most_common()),
        "sessions_with_errors": sum(1 for s in sessions if s.issue_counts.get("error", 0)),
        "sessions_with_warnings": sum(1 for s in sessions if s.issue_counts.get("warn", 0)),
        "modality_summary": modality_summary,
    }


def write_reports(
    out_dir: Path,
    summary: dict[str, Any],
    sessions: list[SessionAudit],
    modalities: list[ModalityAudit],
    issues: list[Issue],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "dataset_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "session_audit.jsonl", "w", encoding="utf-8") as f:
        for session in sessions:
            payload = asdict(session)
            payload["modalities"] = {k: asdict(v) for k, v in session.modalities.items()}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    _write_csv(out_dir / "session_audit.csv", [_session_row(s) for s in sessions])
    _write_csv(out_dir / "modality_audit.csv", [_modality_row(m) for m in modalities])
    _write_csv(out_dir / "issues.csv", [asdict(i) for i in issues])
    write_markdown_summary(out_dir / "dataset_audit_summary.md", summary, issues)
    write_cleaning_plan(out_dir / "cleaning_plan.md", summary, issues)


def _jsonish(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonish(v) for k, v in row.items()})


def _session_row(session: SessionAudit) -> dict[str, Any]:
    return {
        "session": session.session,
        "rel_path": session.rel_path,
        "root": session.root,
        "date": session.date,
        "time": session.time,
        "phase": session.phase,
        "data_format": session.data_format,
        "view_tag": session.view_tag,
        "world_camera": session.world_camera,
        "wrist_camera": session.wrist_camera,
        "session_duration_s": session.session_duration_s,
        "visual_complete_frames": session.visual_complete_frames,
        "visual_hz": session.visual_hz,
        "visual_hz_source": session.visual_hz_source,
        "modalities_present": session.modalities_present,
        "errors": session.issue_counts.get("error", 0),
        "warnings": session.issue_counts.get("warn", 0),
        "infos": session.issue_counts.get("info", 0),
        "cleaning_actions": session.cleaning_actions,
    }


def _modality_row(modality: ModalityAudit) -> dict[str, Any]:
    return {
        "session": modality.session,
        "rel_path": modality.rel_path,
        "phase": modality.phase,
        "view_tag": modality.view_tag,
        "data_format": modality.data_format,
        "modality": modality.modality,
        "kind": modality.kind,
        "source_path": modality.source_path,
        "present": int(modality.present),
        "count": modality.count,
        "duration_s": modality.duration_s,
        "hz_mean": modality.hz_mean,
        "hz_median": modality.hz_median,
        "timestamp_column": modality.timestamp_column,
        "min_timestamp_us": modality.min_timestamp_us,
        "max_timestamp_us": modality.max_timestamp_us,
        "duplicate_timestamps": modality.duplicate_timestamps,
        "timestamp_reversals": modality.timestamp_reversals,
        "empty_cells": modality.empty_cells,
        "zero_size_files": modality.zero_size_files,
        "min_frame_id": modality.min_frame_id,
        "max_frame_id": modality.max_frame_id,
        "missing_sequence_items": modality.missing_sequence_items,
        "extra": modality.extra,
    }


def write_markdown_summary(path: Path, summary: dict[str, Any], issues: list[Issue]) -> None:
    lines: list[str] = []
    lines.append("# Dataset Audit Summary")
    lines.append("")
    lines.append(f"- Total sessions: {summary['total_sessions']}")
    lines.append(f"- Complete visual frames total: {summary['visual_complete_frames_total']}")
    lines.append(f"- Sessions with errors: {summary['sessions_with_errors']}")
    lines.append(f"- Sessions with warnings: {summary['sessions_with_warnings']}")
    lines.append("")
    lines.append("## By Phase")
    lines.extend(_counter_lines(summary["by_phase"]))
    lines.append("")
    lines.append("## By Format")
    lines.extend(_counter_lines(summary["by_format"]))
    lines.append("")
    lines.append("## By View Tag")
    lines.extend(_counter_lines(summary["by_view_tag"]))
    lines.append("")
    lines.append("## Top Issues")
    lines.extend(_counter_lines(dict(list(summary["issue_counts_by_code"].items())[:20])))
    lines.append("")
    lines.append("## Modality Summary")
    lines.append("| modality | sessions | total_count | mean_count | mean_hz | min_hz | max_hz |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, row in summary["modality_summary"].items():
        lines.append(
            f"| {name} | {row['sessions_present']} | {row['total_count']} | "
            f"{row['mean_count']} | {row['mean_hz']} | {row['min_hz']} | {row['max_hz']} |"
        )
    lines.append("")
    if issues:
        lines.append("## Issue Samples")
        for issue in issues[:25]:
            lines.append(f"- [{issue.severity}] {issue.session} {issue.modality} {issue.code}: {issue.message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _counter_lines(counter: dict[str, Any]) -> list[str]:
    if not counter:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in counter.items()]


def write_cleaning_plan(path: Path, summary: dict[str, Any], issues: list[Issue]) -> None:
    by_code: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        by_code[issue.code].append(issue)

    lines: list[str] = []
    lines.append("# Dataset Cleaning Plan")
    lines.append("")
    lines.append("This plan is generated by scripts/audit_dataset.py. It is read-only guidance; review before applying fixes.")
    lines.append("")
    lines.append("## High-Level Policy")
    lines.append("- Keep pre-2026-06-15 DJI-world-camera data and post-2026-06-15 dual-RealSense data as separate splits/tags.")
    lines.append("- Keep 2026-06-15 begin-end sessions as a separate view tag because the initial RealSense position does not see the full robot arm.")
    lines.append("- Prefer frame intersections for training when camera stream counts differ.")
    lines.append("- Preserve raw CSV/image files before sorting, deduplicating, or trimming channels.")
    lines.append("")
    lines.append("## Suggested Fix Buckets")

    buckets = [
        ("non_monotonic_timestamps", "Sort affected CSV files by the selected timestamp column after backup."),
        ("duplicate_timestamps", "Deduplicate equal timestamps after deciding whether to keep first, last, or averaged rows."),
        ("legacy_pressure_64ch", "Trim legacy 64-channel pressure CSVs to the standard 20 modeling channels."),
        ("pressure_adc_out_of_range", "Clip/filter invalid pressure ADC values during preprocessing, not in raw data."),
        ("missing_aligned_timesteps", "Build a sync index or aligned timestep file before per-frame training."),
        ("frames_csv_count_mismatch", "Use frames.csv as the canonical visual timeline or rebuild missing frame metadata."),
        ("visual_count_mismatch", "Use complete-frame intersection or regenerate missing derived frames."),
        ("missing_required_modality", "Drop/recollect sessions with missing required raw modality."),
        ("zero_size_files", "Remove/regenerate corrupt files or drop affected frames."),
        ("known_incomplete_arm_view", "Keep the begin-end view as a separate split/tag."),
    ]
    for code, description in buckets:
        rows = by_code.get(code, [])
        lines.append("")
        lines.append(f"### {code}")
        lines.append(f"- Sessions/issues: {len(rows)}")
        lines.append(f"- Recommendation: {description}")
        for issue in rows[:20]:
            lines.append(f"- {issue.session}: {issue.modality} - {issue.message}")
        if len(rows) > 20:
            lines.append(f"- ... {len(rows) - 20} more; see issues.csv")

    lines.append("")
    lines.append("## Existing Helper Commands")
    lines.append("```bash")
    lines.append("python scripts/verify_fixes.py")
    lines.append("python scripts/trim_pressure_channels.py --data-root dataset/phase2_realdata_sessions/sessions --apply")
    lines.append("python scripts/build_sync_index.py --data-root dataset/phase2_realdata_sessions/sessions --dry-run")
    lines.append("python scripts/build_sync_index.py --data-root dataset/sessions --dry-run")
    lines.append("```")
    lines.append("")
    lines.append("## Report Files")
    lines.append("- session_audit.csv: one row per session.")
    lines.append("- modality_audit.csv: one row per session/modality.")
    lines.append("- issues.csv: normalized issue list with suggestions.")
    lines.append("- session_audit.jsonl: full structured per-session audit.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("DATASET AUDIT COMPLETE")
    print(f"  sessions: {summary['total_sessions']}")
    print(f"  by_format: {summary['by_format']}")
    print(f"  by_view_tag: {summary['by_view_tag']}")
    print(f"  issue_severity: {summary['issue_counts_by_severity']}")
    print(f"  reports: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit TransVTLA real-data sessions.")
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=DEFAULT_ROOTS,
        help="Session roots to scan. Defaults to phase2 sessions and dataset/sessions.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/audit_reports"), help="Report output directory.")
    parser.add_argument("--no-zero-size-check", action="store_true", help="Skip image zero-byte checks.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debugging.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    roots = args.roots
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir

    sessions_paths = discover_sessions(roots)
    if args.limit is not None:
        sessions_paths = sessions_paths[: args.limit]
    begin_end_names = build_begin_end_set(sessions_paths)
    root_mapping = _group_roots_for_sessions(sessions_paths, roots)

    sessions: list[SessionAudit] = []
    issues: list[Issue] = []
    modalities: list[ModalityAudit] = []

    for idx, session_path in enumerate(sessions_paths, start=1):
        session, session_issues = audit_session(
            session_path,
            root_mapping.get(session_path, session_path.parent),
            begin_end_names,
            check_zero_size=not args.no_zero_size_check,
        )
        sessions.append(session)
        issues.extend(session_issues)
        modalities.extend(session.modalities.values())
        if idx % 50 == 0:
            print(f"audited {idx}/{len(sessions_paths)} sessions...", file=sys.stderr)

    summary = summarize(sessions, issues, modalities)
    write_reports(out_dir, summary, sessions, modalities, issues)
    print_console_summary(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
