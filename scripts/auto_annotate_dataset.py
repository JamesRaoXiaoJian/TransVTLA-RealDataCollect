#!/usr/bin/env python3
"""Auto annotation utilities for TransVTLA real and simulated sessions.

The script is intentionally split into auditable stages:

1. prepare-vision
   Build image montages and JSONL prompts for a vision model.
2. build-annotations
   Merge task splits, tactile-derived stage splits, and optional vision results.
3. materialize-real-1000
   Create a 1000-entry real-data training view using copies or symlinks.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from channel_config import (  # noqa: E402
    LEFT_CHANNEL,
    LEFT_MATRIX_CHANNELS,
    PRESSURE_VALUE_COLUMNS,
    RIGHT_CHANNEL,
    RIGHT_MATRIX_CHANNELS,
)
from session_schema import (  # noqa: E402
    RGB_EXT,
    pressure_csv_path,
    read_frame_records,
)


DEFAULT_TASK_MANIFEST = Path("dataset/task_splits/task_session_manifest.csv")
DEFAULT_OUT_DIR = Path("dataset/auto_annotations")
DEFAULT_SIM_SHARDS = Path("dataset/franka_sim_10k_70f_848x480/shards")
BASELINE_ROWS = 50
MAX_PRESSURE_DROP = 3500.0

OBJECT_CLASSES: dict[str, dict[str, Any]] = {
    "big_transparent_cube": {
        "label": "big transparent cube",
        "material": "transparent plastic",
        "fragility": 0.15,
        "slip_risk": 0.35,
        "depth_uncertainty": 0.80,
        "force_min": 1.0,
        "force_max": 8.0,
    },
    "small_hollow_transparent_cube": {
        "label": "small hollow transparent cube",
        "material": "transparent plastic",
        "fragility": 0.20,
        "slip_risk": 0.40,
        "depth_uncertainty": 0.85,
        "force_min": 0.8,
        "force_max": 6.0,
    },
    "cola_cup": {
        "label": "cola cup",
        "material": "transparent plastic",
        "fragility": 0.20,
        "slip_risk": 0.45,
        "depth_uncertainty": 0.55,
        "force_min": 0.5,
        "force_max": 3.0,
    },
    "large_beaker": {
        "label": "large beaker",
        "material": "glass",
        "fragility": 0.85,
        "slip_risk": 0.45,
        "depth_uncertainty": 0.60,
        "force_min": 0.8,
        "force_max": 4.0,
    },
    "small_beaker": {
        "label": "small beaker",
        "material": "glass",
        "fragility": 0.85,
        "slip_risk": 0.40,
        "depth_uncertainty": 0.60,
        "force_min": 0.5,
        "force_max": 3.0,
    },
    "dropper_bottle": {
        "label": "dropper bottle",
        "material": "transparent plastic",
        "fragility": 0.30,
        "slip_risk": 0.50,
        "depth_uncertainty": 0.45,
        "force_min": 0.6,
        "force_max": 3.5,
    },
    "transparent_cube": {
        "label": "transparent cube",
        "material": "transparent plastic",
        "fragility": 0.15,
        "slip_risk": 0.35,
        "depth_uncertainty": 0.80,
        "force_min": 1.0,
        "force_max": 8.0,
    },
    "beaker": {
        "label": "beaker",
        "material": "glass",
        "fragility": 0.85,
        "slip_risk": 0.45,
        "depth_uncertainty": 0.60,
        "force_min": 0.6,
        "force_max": 4.0,
    },
    "test_tube": {
        "label": "test tube",
        "material": "glass",
        "fragility": 0.90,
        "slip_risk": 0.50,
        "depth_uncertainty": 0.45,
        "force_min": 0.4,
        "force_max": 2.5,
    },
}

VISION_OBJECT_CLASS_KEYS = [
    "big_transparent_cube",
    "small_hollow_transparent_cube",
    "cola_cup",
    "large_beaker",
    "small_beaker",
    "dropper_bottle",
]

SLOT_PAIRS = {"1-2", "1-3", "1-5", "2-3"}

TASK1 = "task1_simple_put_object_in_box"
TASK2 = "task2_complex_put_object_or_beaker_tripod"
TASK3 = "task3_simple_beaker_tripod"
TASK4 = "task4_test_tube_rack_insertion"
UNKNOWN_TASK = "unknown_or_mixed"
SIM_TASK = "sim_transparent_cube_box"

TIP_COLUMNS = [f"CH{LEFT_CHANNEL}", f"CH{RIGHT_CHANNEL}"]
PAD_CHANNELS = [ch for row in LEFT_MATRIX_CHANNELS for ch in row] + [
    ch for row in RIGHT_MATRIX_CHANNELS for ch in row
]
PAD_COLUMNS = [f"CH{ch}" for ch in PAD_CHANNELS]


@dataclass(frozen=True)
class SessionRecord:
    session: str
    rel_path: str
    abs_path: Path
    data_source: str
    task_id: str
    task_name: str = ""
    split: str = ""
    data_format: str = ""
    usable_for_aligned_training: str = ""
    extra: dict[str, str] | None = None


@dataclass(frozen=True)
class StageInfo:
    frame_count: int
    contact_pressure_index: int | None
    contact_frame_index: int | None
    contact_frame_stem: str
    contact_source: str
    stage1_start_frame_index: int | None
    stage1_start_stem: str
    stage0_start_frame_index: int | None
    stage0_end_frame_index: int | None
    stage1_end_frame_index: int | None
    threshold: float
    lead_frames: int
    notes: str


def repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_id(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def safe_key(record: SessionRecord) -> str:
    return f"{record.data_source}_{record.session}_{stable_id(record.rel_path, 8)}"


def normalize_object_type(value: str | None) -> str:
    if value is None:
        return ""
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "big_cube": "big_transparent_cube",
        "large_transparent_cube": "big_transparent_cube",
        "small_transparent_cube": "small_hollow_transparent_cube",
        "hollow_transparent_cube": "small_hollow_transparent_cube",
        "coke_cup": "cola_cup",
        "coca_cola_cup": "cola_cup",
        "large_glass_beaker": "large_beaker",
        "small_glass_beaker": "small_beaker",
        "dropping_bottle": "dropper_bottle",
    }
    key = aliases.get(key, key)
    return key if key in OBJECT_CLASSES else ""


def normalize_slot_pair(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip().replace(" ", "").replace("_", "-").replace("to", "-")
    return cleaned if cleaned in SLOT_PAIRS else ""


def is_real_task_needing_object(task_id: str) -> bool:
    return task_id in {TASK1, TASK2}


def is_real_task_needing_slot_pair(task_id: str) -> bool:
    return task_id == TASK4


def instruction_for(task_id: str, object_type: str = "", slot_pair: str = "") -> str:
    obj = OBJECT_CLASSES.get(object_type, {}).get("label", "")
    if task_id == SIM_TASK:
        return "Pick up the transparent cube and place it into the box."
    if task_id == TASK1:
        if obj:
            return f"Pick up the {obj} and place it into the box."
        return "Pick up the target object and place it into the box."
    if task_id == TASK2:
        if obj:
            return f"Find the {obj} in the current environment, pick it up, and place it into the box."
        return "Find the target object in the current environment, pick it up, and place it into the box."
    if task_id == TASK3:
        return "Pick up the beaker and place it on the tripod."
    if task_id == TASK4:
        if slot_pair:
            src, dst = slot_pair.split("-")
            return f"Pick up the test tube from slot {src} and insert it into slot {dst} of the test tube rack."
        return "Pick up the test tube from slot X and insert it into slot Y of the test tube rack."
    return ""


def read_task_manifest(path: Path) -> list[SessionRecord]:
    rows = read_csv(path)
    records: list[SessionRecord] = []
    for row in rows:
        rel_path = row["rel_path"]
        records.append(
            SessionRecord(
                session=row["session"],
                rel_path=rel_path,
                abs_path=repo_path(rel_path),
                data_source="real",
                task_id=row.get("task_id", ""),
                task_name=row.get("task_name", ""),
                split=row.get("split", ""),
                data_format=row.get("data_format", ""),
                usable_for_aligned_training=row.get("usable_for_aligned_training", ""),
                extra=row,
            )
        )
    return records


def discover_sim_sessions(sim_shards: Path) -> list[SessionRecord]:
    sim_shards = repo_path(sim_shards)
    if not sim_shards.is_dir():
        return []
    records: list[SessionRecord] = []
    for shard in sorted(p for p in sim_shards.iterdir() if p.is_dir()):
        try:
            sessions = sorted(p for p in shard.iterdir() if p.is_dir() and p.name.startswith("session_"))
        except OSError:
            continue
        for session_dir in sessions:
            records.append(
                SessionRecord(
                    session=session_dir.name,
                    rel_path=rel(session_dir),
                    abs_path=session_dir,
                    data_source="sim",
                    task_id=SIM_TASK,
                    task_name="Simulated transparent cube box placement",
                    split=shard.name,
                    data_format="dual_realsense",
                    usable_for_aligned_training="1",
                    extra={},
                )
            )
    return records


def read_vision_results(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    path = repo_path(path)
    if not path.is_file():
        return {}

    rows: list[dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append({k: "" if v is None else str(v) for k, v in obj.items()})
    else:
        rows = read_csv(path)

    results: dict[str, dict[str, str]] = {}
    for row in rows:
        session = row.get("session", "").strip()
        rel_path = row.get("rel_path", "").strip()
        key = rel_path or session
        if not key:
            continue
        normalized = dict(row)
        normalized["object_type"] = normalize_object_type(row.get("object_type"))
        normalized["slot_pair"] = normalize_slot_pair(row.get("slot_pair"))
        results[key] = normalized
    return results


def vision_for(record: SessionRecord, results: dict[str, dict[str, str]]) -> dict[str, str]:
    return results.get(record.rel_path) or results.get(record.session) or {}


def session_rgb_dirs(session_dir: Path) -> tuple[Path, Path] | None:
    world_rgb = session_dir / "world_camera" / "rgb"
    wrist_rgb = session_dir / "wrist_camera" / "rgb"
    try:
        if world_rgb.is_dir() and wrist_rgb.is_dir():
            return world_rgb, wrist_rgb
    except OSError:
        return None

    legacy_world = session_dir / "dji"
    legacy_wrist = session_dir / "realsense_rgb"
    try:
        if legacy_world.is_dir() and legacy_wrist.is_dir():
            return legacy_world, legacy_wrist
    except OSError:
        return None
    return None


def list_rgb_stems(session_dir: Path) -> list[str]:
    dirs = session_rgb_dirs(session_dir)
    if dirs is None:
        return []

    def stems(directory: Path) -> set[str]:
        try:
            return {p.stem for p in directory.iterdir() if p.is_file() and p.suffix.lower() == RGB_EXT}
        except OSError:
            return set()

    world = stems(dirs[0])
    wrist = stems(dirs[1])
    if not world or not wrist:
        return []
    return sorted(world.intersection(wrist))


def frame_timestamps(session_dir: Path) -> list[int]:
    timestamps: list[int] = []
    for row in read_frame_records(session_dir):
        value = row.get("capture_monotonic_us")
        if not value:
            continue
        try:
            timestamps.append(int(float(value)))
        except ValueError:
            continue
    return timestamps


def read_aligned_rows(session_dir: Path) -> list[dict[str, str]]:
    path = session_dir / "aligned_timesteps.csv"
    if not path.is_file():
        return []
    try:
        return read_csv(path)
    except (OSError, csv.Error):
        return []


def first_stable_index(values: list[float], threshold: float, consecutive: int) -> int | None:
    if not values:
        return None
    consecutive = max(1, consecutive)
    run = 0
    for idx, value in enumerate(values):
        if value >= threshold:
            run += 1
            if run >= consecutive:
                return idx - consecutive + 1
        else:
            run = 0
    for idx, value in enumerate(values):
        if value >= threshold:
            return idx
    return None


def pressure_activation(
    session_dir: Path,
    threshold: float,
    consecutive: int,
) -> tuple[int | None, str, int | None, str]:
    path = pressure_csv_path(session_dir)
    if not path.is_file():
        return None, "none", None, "missing pressure.csv"

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in PRESSURE_VALUE_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                return None, "none", None, f"pressure.csv missing columns: {missing[:3]}"
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        return None, "none", None, f"failed to read pressure.csv: {exc}"

    if not rows:
        return None, "none", None, "empty pressure.csv"

    baseline_n = min(BASELINE_ROWS, len(rows))
    baselines: dict[str, float] = {}
    for col in PRESSURE_VALUE_COLUMNS:
        values: list[float] = []
        for row in rows[:baseline_n]:
            try:
                values.append(float(row[col]))
            except (KeyError, ValueError):
                pass
        baselines[col] = sum(values) / len(values) if values else 0.0

    tip_values: list[float] = []
    pad_values: list[float] = []
    timestamps: list[int | None] = []
    for row in rows:
        tip_max = 0.0
        for col in TIP_COLUMNS:
            try:
                tip_max = max(tip_max, max(0.0, baselines[col] - float(row[col])) / MAX_PRESSURE_DROP)
            except (KeyError, ValueError):
                continue
        pad_max = 0.0
        for col in PAD_COLUMNS:
            try:
                pad_max = max(pad_max, max(0.0, baselines[col] - float(row[col])) / MAX_PRESSURE_DROP)
            except (KeyError, ValueError):
                continue
        tip_values.append(tip_max)
        pad_values.append(pad_max)
        ts_raw = row.get("host_monotonic_us") or row.get("timestamp_us") or row.get("sensor_timestamp_us")
        try:
            timestamps.append(int(float(ts_raw))) if ts_raw else timestamps.append(None)
        except ValueError:
            timestamps.append(None)

    tip_idx = first_stable_index(tip_values, threshold, consecutive)
    pad_idx = first_stable_index(pad_values, threshold, consecutive)
    if tip_idx is None and pad_idx is None:
        return None, "none", None, "no tactile activation crossed threshold"
    if tip_idx is not None and (pad_idx is None or tip_idx <= pad_idx):
        return tip_idx, "fingertip", timestamps[tip_idx], ""
    assert pad_idx is not None
    return pad_idx, "finger_pad", timestamps[pad_idx], "finger pad activated earlier than fingertip"


def nearest_index(timestamps: list[int], target: int) -> int | None:
    if not timestamps:
        return None
    best_idx = 0
    best_dt = abs(timestamps[0] - target)
    for idx, ts in enumerate(timestamps[1:], start=1):
        dt = abs(ts - target)
        if dt < best_dt:
            best_idx = idx
            best_dt = dt
    return best_idx


def contact_frame_from_timestamp(session_dir: Path, contact_ts: int | None, frame_count: int) -> int | None:
    if contact_ts is None or frame_count <= 0:
        return None

    aligned = read_aligned_rows(session_dir)
    aligned_ts: list[int] = []
    for row in aligned:
        raw = row.get("pressure_timestamp_us") or row.get("visual_timestamp_us")
        try:
            aligned_ts.append(int(float(raw))) if raw else aligned_ts.append(0)
        except ValueError:
            aligned_ts.append(0)
    aligned_ts = [ts for ts in aligned_ts if ts > 0]
    idx = nearest_index(aligned_ts, contact_ts)
    if idx is not None:
        return max(0, min(frame_count - 1, idx))

    ts = frame_timestamps(session_dir)
    idx = nearest_index(ts, contact_ts)
    if idx is not None:
        return max(0, min(frame_count - 1, idx))
    return None


def estimate_stage_info(
    session_dir: Path,
    lead_frames: int,
    threshold: float,
    consecutive: int,
) -> StageInfo:
    stems = list_rgb_stems(session_dir)
    frame_count = len(stems)
    contact_pressure_idx, contact_source, contact_ts, note = pressure_activation(
        session_dir, threshold=threshold, consecutive=consecutive
    )
    contact_frame_idx = contact_frame_from_timestamp(session_dir, contact_ts, frame_count)
    stage1_start_idx: int | None = None
    if contact_frame_idx is not None:
        stage1_start_idx = max(0, contact_frame_idx - lead_frames)

    def stem_at(idx: int | None) -> str:
        if idx is None or idx < 0 or idx >= len(stems):
            return ""
        return stems[idx]

    return StageInfo(
        frame_count=frame_count,
        contact_pressure_index=contact_pressure_idx,
        contact_frame_index=contact_frame_idx,
        contact_frame_stem=stem_at(contact_frame_idx),
        contact_source=contact_source,
        stage1_start_frame_index=stage1_start_idx,
        stage1_start_stem=stem_at(stage1_start_idx),
        stage0_start_frame_index=0 if frame_count else None,
        stage0_end_frame_index=(stage1_start_idx - 1) if stage1_start_idx is not None else None,
        stage1_end_frame_index=(frame_count - 1) if frame_count else None,
        threshold=threshold,
        lead_frames=lead_frames,
        notes=note,
    )


def annotation_object(record: SessionRecord, vision: dict[str, str]) -> tuple[str, str, bool]:
    if record.data_source == "sim":
        return "transparent_cube", OBJECT_CLASSES["transparent_cube"]["label"], False
    if record.task_id in {TASK1, TASK2}:
        key = normalize_object_type(vision.get("object_type"))
        if key:
            return key, OBJECT_CLASSES[key]["label"], False
        return "", "", True
    if record.task_id == TASK3:
        return "beaker", OBJECT_CLASSES["beaker"]["label"], False
    if record.task_id == TASK4:
        return "test_tube", OBJECT_CLASSES["test_tube"]["label"], False
    return "", "", True


def annotation_slot_pair(record: SessionRecord, vision: dict[str, str]) -> tuple[str, bool]:
    if record.task_id != TASK4:
        return "", False
    slot_pair = normalize_slot_pair(vision.get("slot_pair"))
    if slot_pair:
        return slot_pair, False
    return "", True


def stage_segments(stage: StageInfo) -> list[dict[str, Any]]:
    if stage.frame_count <= 0:
        return []
    if stage.stage1_start_frame_index is None:
        return [
            {
                "stage": 0,
                "name": "visual_pre_contact",
                "use_tactile": False,
                "start_frame_index": 0,
                "end_frame_index": stage.frame_count - 1,
            }
        ]
    segments: list[dict[str, Any]] = []
    if stage.stage1_start_frame_index > 0:
        segments.append(
            {
                "stage": 0,
                "name": "visual_pre_contact",
                "use_tactile": False,
                "start_frame_index": 0,
                "end_frame_index": stage.stage1_start_frame_index - 1,
            }
        )
    segments.append(
        {
            "stage": 1,
            "name": "tactile_guided_contact",
            "use_tactile": True,
            "start_frame_index": stage.stage1_start_frame_index,
            "end_frame_index": stage.frame_count - 1,
        }
    )
    return segments


def build_annotation(record: SessionRecord, vision: dict[str, str], stage: StageInfo) -> dict[str, Any]:
    object_type, object_label, needs_object_review = annotation_object(record, vision)
    slot_pair, needs_slot_review = annotation_slot_pair(record, vision)
    obj_attrs = OBJECT_CLASSES.get(object_type, {})
    instruction = instruction_for(record.task_id, object_type=object_type, slot_pair=slot_pair)
    if not instruction:
        instruction = instruction_for(record.task_id)

    needs_stage_review = record.data_source == "real" and stage.stage1_start_frame_index is None
    stage_check = vision.get("stage_visual_check", "").strip().lower()
    if stage_check in {"fail", "failed", "no"}:
        needs_stage_review = True
    needs_review = needs_object_review or needs_slot_review or needs_stage_review or record.task_id == UNKNOWN_TASK

    annotation = {
        "schema": "transvtla_auto_annotation/v1",
        "session": record.session,
        "rel_path": record.rel_path,
        "data_source": record.data_source,
        "task": {
            "id": record.task_id,
            "name": record.task_name,
            "success": True,
        },
        "language_instruction": instruction,
        "object": {
            "type": object_type,
            "label": object_label,
            "attributes": {"item_type": object_label},
            "material": obj_attrs.get("material", ""),
            "fragility": obj_attrs.get("fragility", ""),
            "slip_risk": obj_attrs.get("slip_risk", ""),
            "depth_uncertainty": obj_attrs.get("depth_uncertainty", ""),
            "force_min": obj_attrs.get("force_min", ""),
            "force_max": obj_attrs.get("force_max", ""),
        },
        "slots": {
            "pair": slot_pair,
            "source": slot_pair.split("-")[0] if slot_pair else "",
            "target": slot_pair.split("-")[1] if slot_pair else "",
        },
        "stages": {
            "labels": {
                "0": "visual_pre_contact",
                "1": "tactile_guided_contact",
            },
            "tactile_enabled_stage": 1,
            "segments": stage_segments(stage),
            "contact_pressure_index": stage.contact_pressure_index,
            "contact_frame_index": stage.contact_frame_index,
            "contact_frame_stem": stage.contact_frame_stem,
            "contact_source": stage.contact_source,
            "stage1_start_frame_index": stage.stage1_start_frame_index,
            "stage1_start_stem": stage.stage1_start_stem,
            "stage0_start_frame_index": stage.stage0_start_frame_index,
            "stage0_end_frame_index": stage.stage0_end_frame_index,
            "stage1_end_frame_index": stage.stage1_end_frame_index,
            "frame_count": stage.frame_count,
            "heuristic": {
                "lead_frames": stage.lead_frames,
                "accepted_lead_frame_range": [35, 45],
                "normalized_pressure_threshold": stage.threshold,
                "notes": stage.notes,
            },
        },
        "vision": {
            "object_type": vision.get("object_type", ""),
            "slot_pair": vision.get("slot_pair", ""),
            "confidence": vision.get("confidence", ""),
            "stage_visual_check": vision.get("stage_visual_check", ""),
            "notes": vision.get("notes", ""),
        },
        "quality": {
            "success": True,
            "needs_review": needs_review,
            "review_reasons": [
                reason
                for reason, enabled in [
                    ("missing_object_visual_label", needs_object_review),
                    ("missing_slot_pair_visual_label", needs_slot_review),
                    ("missing_or_failed_stage_split", needs_stage_review),
                    ("unknown_task", record.task_id == UNKNOWN_TASK),
                ]
                if enabled
            ],
        },
    }
    return annotation


def flatten_annotation(record: SessionRecord, annotation: dict[str, Any]) -> dict[str, Any]:
    obj = annotation["object"]
    stages = annotation["stages"]
    task = annotation["task"]
    slots = annotation["slots"]
    quality = annotation["quality"]
    return {
        "session": record.session,
        "rel_path": record.rel_path,
        "annotation_key": safe_key(record),
        "data_source": record.data_source,
        "split": record.split,
        "data_format": record.data_format,
        "usable_for_aligned_training": record.usable_for_aligned_training,
        "task_id": task["id"],
        "task_name": task["name"],
        "language_instruction": annotation["language_instruction"],
        "object_type": obj["type"],
        "object_label": obj["label"],
        "item_type": obj["attributes"]["item_type"],
        "material": obj["material"],
        "fragility": obj["fragility"],
        "slip_risk": obj["slip_risk"],
        "depth_uncertainty": obj["depth_uncertainty"],
        "force_min": obj["force_min"],
        "force_max": obj["force_max"],
        "slot_pair": slots["pair"],
        "slot_source": slots["source"],
        "slot_target": slots["target"],
        "success": 1,
        "needs_review": int(bool(quality["needs_review"])),
        "review_reasons": "|".join(quality["review_reasons"]),
        "frame_count": stages["frame_count"],
        "stage1_start_frame_index": stages["stage1_start_frame_index"],
        "stage1_start_stem": stages["stage1_start_stem"],
        "contact_frame_index": stages["contact_frame_index"],
        "contact_frame_stem": stages["contact_frame_stem"],
        "contact_source": stages["contact_source"],
        "stage_segments_json": json.dumps(stages["segments"], ensure_ascii=False),
        "vision_confidence": annotation["vision"]["confidence"],
        "vision_stage_visual_check": annotation["vision"]["stage_visual_check"],
        "vision_notes": annotation["vision"]["notes"],
    }


def selected_indices(frame_count: int, stage: StageInfo) -> list[int]:
    if frame_count <= 0:
        return []
    base = [0, frame_count // 4, frame_count // 2, (3 * frame_count) // 4, frame_count - 1]
    for idx in (stage.stage1_start_frame_index, stage.contact_frame_index):
        if idx is not None:
            base.extend([max(0, idx - 2), idx, min(frame_count - 1, idx + 2)])
    return sorted(set(i for i in base if 0 <= i < frame_count))


def build_montage(session_dir: Path, out_path: Path, stage: StageInfo) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    dirs = session_rgb_dirs(session_dir)
    if dirs is None:
        return False
    world_rgb, wrist_rgb = dirs
    stems = list_rgb_stems(session_dir)
    indices = selected_indices(len(stems), stage)
    if not indices:
        return False

    cell_w, cell_h = 256, 160
    label_h = 24
    cols = 2
    rows = len(indices)
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for row_idx, frame_idx in enumerate(indices):
        stem = stems[frame_idx]
        label = f"frame={frame_idx} stem={stem}"
        if frame_idx == stage.stage1_start_frame_index:
            label += " stage1_start"
        if frame_idx == stage.contact_frame_index:
            label += f" contact:{stage.contact_source}"
        y = row_idx * (cell_h + label_h)
        draw.text((4, y + 4), label, fill=(0, 0, 0))
        for col_idx, (camera_dir, camera_label) in enumerate(
            [(world_rgb, "world"), (wrist_rgb, "wrist")]
        ):
            path = camera_dir / f"{stem}{RGB_EXT}"
            try:
                img = Image.open(path).convert("RGB")
            except OSError:
                continue
            img.thumbnail((cell_w, cell_h))
            x = col_idx * cell_w + (cell_w - img.width) // 2
            img_y = y + label_h + (cell_h - img.height) // 2
            canvas.paste(img, (x, img_y))
            draw.text((col_idx * cell_w + 4, y + label_h + 4), camera_label, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return True


def vision_prompt(record: SessionRecord, image_rel_path: str, stage: StageInfo) -> dict[str, Any]:
    needs_object = is_real_task_needing_object(record.task_id)
    needs_slot = is_real_task_needing_slot_pair(record.task_id)
    allowed_objects = [OBJECT_CLASSES[k]["label"] for k in VISION_OBJECT_CLASS_KEYS]
    prompt_parts = [
        "Annotate this robot manipulation episode from the montage.",
        "Use the world and wrist camera images. Return strict JSON only.",
        "The stage check should pass only if the wrist/gripper view at stage1_start is roughly above the target object.",
    ]
    if needs_object:
        prompt_parts.append(
            "Select exactly one object_type key from: "
            + ", ".join(VISION_OBJECT_CLASS_KEYS)
            + ". Labels: "
            + ", ".join(allowed_objects)
            + "."
        )
    if needs_slot:
        prompt_parts.append("Select exactly one slot_pair from: " + ", ".join(sorted(SLOT_PAIRS)) + ".")
    prompt_parts.append(
        'Return JSON with keys: session, object_type, slot_pair, stage_visual_check, confidence, notes. '
        'Use null for object_type or slot_pair when not required.'
    )
    return {
        "schema": "transvtla_vision_request/v1",
        "session": record.session,
        "rel_path": record.rel_path,
        "task_id": record.task_id,
        "task_name": record.task_name,
        "image": image_rel_path,
        "needs_object_type": needs_object,
        "needs_slot_pair": needs_slot,
        "allowed_object_type_keys": VISION_OBJECT_CLASS_KEYS if needs_object else [],
        "allowed_slot_pairs": sorted(SLOT_PAIRS) if needs_slot else [],
        "stage1_start_frame_index": stage.stage1_start_frame_index,
        "contact_frame_index": stage.contact_frame_index,
        "contact_source": stage.contact_source,
        "prompt": " ".join(prompt_parts),
    }


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("vision response was not a JSON object")
    return obj


def normalize_vision_result(result: dict[str, Any], request_row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized["session"] = str(normalized.get("session") or request_row.get("session", ""))
    normalized["rel_path"] = str(normalized.get("rel_path") or request_row.get("rel_path", ""))
    normalized["task_id"] = str(normalized.get("task_id") or request_row.get("task_id", ""))
    normalized["object_type"] = normalize_object_type(normalized.get("object_type"))
    normalized["slot_pair"] = normalize_slot_pair(normalized.get("slot_pair"))
    normalized["stage_visual_check"] = str(normalized.get("stage_visual_check") or "").strip().lower()
    if normalized["stage_visual_check"] in {"yes", "true", "ok"}:
        normalized["stage_visual_check"] = "pass"
    if normalized["stage_visual_check"] in {"no", "false"}:
        normalized["stage_visual_check"] = "fail"
    normalized["confidence"] = str(normalized.get("confidence") or "")
    normalized["notes"] = str(normalized.get("notes") or "")
    return normalized


def post_vision_request(
    api_url: str,
    api_key: str,
    model: str,
    request_row: dict[str, Any],
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    image_path = repo_path(request_row["image"])
    prompt = (
        request_row["prompt"]
        + "\n\nSession: "
        + str(request_row["session"])
        + "\nTask id: "
        + str(request_row["task_id"])
    )
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url(image_path)},
                ],
            }
        ],
        "temperature": temperature,
    }
    http_request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    api_response = json.loads(body)
    text = extract_response_text(api_response)
    result = parse_json_object(text)
    result["session"] = str(result.get("session") or request_row["session"])
    result["rel_path"] = str(result.get("rel_path") or request_row["rel_path"])
    result["task_id"] = str(result.get("task_id") or request_row["task_id"])
    result["object_type"] = normalize_object_type(result.get("object_type"))
    result["slot_pair"] = normalize_slot_pair(result.get("slot_pair"))
    return result


def command_run_openai_vision(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable {args.api_key_env}.")
    model = args.model or os.environ.get("OPENAI_VISION_MODEL", "")
    if not model:
        raise SystemExit("Pass --model or set OPENAI_VISION_MODEL.")

    requests_path = repo_path(args.requests)
    output_path = repo_path(args.output)
    rows = read_jsonl(requests_path)
    if args.limit:
        rows = rows[: args.limit]

    completed: set[str] = set()
    if args.resume and output_path.is_file():
        for row in read_jsonl(output_path):
            key = str(row.get("rel_path") or row.get("session") or "")
            if key:
                completed.add(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    counters: Counter[str] = Counter()
    with open(output_path, mode, encoding="utf-8") as f:
        for idx, request_row in enumerate(rows, start=1):
            key = str(request_row.get("rel_path") or request_row.get("session") or "")
            if key in completed:
                counters["skipped_existing"] += 1
                continue
            try:
                result = post_vision_request(
                    args.api_url,
                    api_key,
                    model,
                    request_row,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
            except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
                result = {
                    "session": request_row.get("session", ""),
                    "rel_path": request_row.get("rel_path", ""),
                    "task_id": request_row.get("task_id", ""),
                    "object_type": "",
                    "slot_pair": "",
                    "stage_visual_check": "",
                    "confidence": "",
                    "notes": f"vision_call_failed: {exc}",
                }
                counters["failed"] += 1
            else:
                counters["ok"] += 1
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            if args.progress_every and idx % args.progress_every == 0:
                print(f"[run-openai-vision] processed {idx}/{len(rows)} requests", flush=True)

    summary = {
        "schema": "transvtla_run_openai_vision_summary/v1",
        "requests": len(rows),
        "output": rel(output_path),
        "model": model,
        "counters": dict(counters),
    }
    write_json(output_path.parent / "run_openai_vision_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def qwen_prompt(request_row: dict[str, Any]) -> str:
    needs_object = bool(request_row.get("needs_object_type"))
    needs_slot = bool(request_row.get("needs_slot_pair"))
    lines = [
        "You are labeling a robot manipulation episode from a montage image.",
        "The montage alternates world camera and wrist camera views. Frame labels may mark stage1_start and contact.",
        "Return strict JSON only. Do not include markdown.",
        "JSON keys: session, object_type, slot_pair, stage_visual_check, confidence, notes.",
        f"session: {request_row.get('session', '')}",
        f"task_id: {request_row.get('task_id', '')}",
    ]
    if needs_object:
        lines.append(
            "object_type must be exactly one of: "
            + ", ".join(VISION_OBJECT_CLASS_KEYS)
            + "."
        )
    else:
        lines.append("object_type must be null unless this task requires object classification.")
    if needs_slot:
        lines.append("slot_pair must be exactly one of: " + ", ".join(sorted(SLOT_PAIRS)) + ".")
    else:
        lines.append("slot_pair must be null unless this task requires a test-tube slot pair.")
    lines.append(
        "stage_visual_check must be pass if the wrist/gripper view at stage1_start is roughly above the target object; otherwise fail."
    )
    lines.append(
        "Use confidence as a decimal string between 0 and 1. Use short notes explaining ambiguous cases."
    )
    if request_row.get("consistency_review"):
        lines.extend(
            [
                "",
                "Second-pass consistency review:",
                "Neighboring sessions in the same collection sheet usually have the same task and target.",
                "Use the image as the authority. If the image is ambiguous, prefer the label that is consistent with the same-sheet majority and immediate neighbors.",
                f"first_pass_object_type: {request_row.get('first_pass_object_type', '')}",
                f"first_pass_slot_pair: {request_row.get('first_pass_slot_pair', '')}",
                f"consistency_reasons: {request_row.get('consistency_reasons', '')}",
                f"task_sheet_id: {request_row.get('task_sheet_id', '')}",
                f"task_row_in_sheet: {request_row.get('task_row_in_sheet', '')}",
                f"sheet_label_counts: {json.dumps(request_row.get('sheet_label_counts', {}), ensure_ascii=False)}",
                f"sheet_majority_label: {request_row.get('sheet_majority_label', '')}",
                f"previous_session_label: {request_row.get('previous_session_label', '')}",
                f"next_session_label: {request_row.get('next_session_label', '')}",
            ]
        )
    if request_row.get("third_pass_review"):
        lines.extend(
            [
                "",
                "Third-pass boundary and long-tail review:",
                "Use these class boundaries carefully:",
                "- big_transparent_cube: solid transparent cube or larger transparent cuboid-like block; no clear circular hollow center.",
                "- small_hollow_transparent_cube: small transparent cube with a visible circular/through-hole or hollow center.",
                "- cola_cup: transparent plastic drink cup, usually thin-walled and cup-like, not laboratory glassware.",
                "- large_beaker / small_beaker: laboratory beaker or glass cup with cylindrical shape; use size relative to gripper and box.",
                "- dropper_bottle: bottle-like target with cap/nozzle/neck, not an open cup or cube.",
                "If a sheet truly changes target object, keep the visual label even if neighbors differ.",
                "If only one or two adjacent sessions disagree and the image evidence is weak, prefer the immediate-neighbor/sheet-consistent label.",
                "For task4, identify the slot_pair only from visible source/target hole indices; if uncertain, keep the most consistent allowed pair from the sheet context.",
                f"issue_details: {request_row.get('issue_details', '')}",
            ]
        )
    return "\n".join(lines)


def load_qwen_model(model_name: str, dtype: str, device_map: str, attn_implementation: str | None):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch_dtype = "auto"
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": device_map,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def qwen_generate_one(
    model: Any,
    processor: Any,
    request_row: dict[str, Any],
    max_new_tokens: int,
) -> str:
    image_path = str(repo_path(request_row["image"]))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": qwen_prompt(request_row)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip() if output_text else ""


def command_run_qwen3vl_local(args: argparse.Namespace) -> None:
    requests_path = repo_path(args.requests)
    output_path = repo_path(args.output)
    rows = read_jsonl(requests_path)
    if args.limit:
        rows = rows[: args.limit]

    completed: set[str] = set()
    if args.resume and output_path.is_file():
        for row in read_jsonl(output_path):
            key = str(row.get("rel_path") or row.get("session") or "")
            if key:
                completed.add(key)

    print(f"[run-qwen3vl-local] loading {args.model}", flush=True)
    model, processor = load_qwen_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation or None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    counters: Counter[str] = Counter()
    with open(output_path, mode, encoding="utf-8") as f:
        for idx, request_row in enumerate(rows, start=1):
            key = str(request_row.get("rel_path") or request_row.get("session") or "")
            if key in completed:
                counters["skipped_existing"] += 1
                continue
            try:
                text = qwen_generate_one(model, processor, request_row, args.max_new_tokens)
                parsed = parse_json_object(text)
                result = normalize_vision_result(parsed, request_row)
            except Exception as exc:  # noqa: BLE001
                result = normalize_vision_result(
                    {
                        "object_type": "",
                        "slot_pair": "",
                        "stage_visual_check": "",
                        "confidence": "",
                        "notes": f"qwen3vl_failed: {type(exc).__name__}: {exc}",
                    },
                    request_row,
                )
                counters["failed"] += 1
            else:
                counters["ok"] += 1
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if args.progress_every and idx % args.progress_every == 0:
                print(f"[run-qwen3vl-local] processed {idx}/{len(rows)} requests", flush=True)

    summary = {
        "schema": "transvtla_run_qwen3vl_summary/v1",
        "requests": len(rows),
        "output": rel(output_path),
        "model": args.model,
        "counters": dict(counters),
    }
    write_json(output_path.parent / "run_qwen3vl_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_prepare_vision(args: argparse.Namespace) -> None:
    records = read_task_manifest(repo_path(args.task_manifest))
    if args.max_sessions:
        records = records[: args.max_sessions]
    needed = [
        r
        for r in records
        if is_real_task_needing_object(r.task_id)
        or is_real_task_needing_slot_pair(r.task_id)
        or r.task_id in {TASK1, TASK2, TASK3, TASK4}
    ]
    out_dir = repo_path(args.out_dir)
    image_dir = out_dir / "vision_inputs" / "images"
    requests: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for idx, record in enumerate(needed, start=1):
        if args.max_vision_requests and len(requests) >= args.max_vision_requests:
            break
        stage = estimate_stage_info(
            record.abs_path,
            lead_frames=args.lead_frames,
            threshold=args.pressure_threshold,
            consecutive=args.consecutive_pressure_rows,
        )
        image_name = f"{safe_key(record)}.jpg"
        image_path = image_dir / image_name
        if not args.no_images:
            ok = build_montage(record.abs_path, image_path, stage)
            counters["montage_written" if ok else "montage_failed"] += 1
        image_rel = rel(image_path)
        requests.append(vision_prompt(record, image_rel, stage))
        template_rows.append(
            {
                "session": record.session,
                "rel_path": record.rel_path,
                "task_id": record.task_id,
                "object_type": "",
                "slot_pair": "",
                "stage_visual_check": "",
                "confidence": "",
                "notes": "",
            }
        )
        if args.progress_every and idx % args.progress_every == 0:
            print(f"[prepare-vision] processed {idx}/{len(needed)} records", flush=True)

    append_jsonl(out_dir / "vision_inputs" / "vision_requests.jsonl", requests)
    write_csv(
        out_dir / "vision_inputs" / "vision_results_template.csv",
        template_rows,
        [
            "session",
            "rel_path",
            "task_id",
            "object_type",
            "slot_pair",
            "stage_visual_check",
            "confidence",
            "notes",
        ],
    )
    summary = {
        "schema": "transvtla_vision_prepare_summary/v1",
        "records_considered": len(records),
        "requests": len(requests),
        "image_dir": rel(image_dir),
        "counters": dict(counters),
    }
    write_json(out_dir / "vision_inputs" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_build_annotations(args: argparse.Namespace) -> None:
    records = read_task_manifest(repo_path(args.task_manifest))
    if args.include_sim:
        records.extend(discover_sim_sessions(repo_path(args.sim_shards)))
    if args.max_sessions:
        records = records[: args.max_sessions]

    vision_results = read_vision_results(Path(args.vision_results) if args.vision_results else None)
    out_dir = repo_path(args.out_dir)
    sidecar_dir = out_dir / "session_annotations"
    manifest_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for idx, record in enumerate(records, start=1):
        vision = vision_for(record, vision_results)
        stage = estimate_stage_info(
            record.abs_path,
            lead_frames=args.lead_frames,
            threshold=args.pressure_threshold,
            consecutive=args.consecutive_pressure_rows,
        )
        annotation = build_annotation(record, vision, stage)
        manifest_rows.append(flatten_annotation(record, annotation))
        sidecar_path = sidecar_dir / f"{safe_key(record)}.json"
        write_json(sidecar_path, annotation)
        if args.write_session_files:
            write_json(record.abs_path / "annotation.auto.json", annotation)
        counters[record.data_source] += 1
        if annotation["quality"]["needs_review"]:
            counters["needs_review"] += 1
        if args.progress_every and idx % args.progress_every == 0:
            print(f"[build-annotations] processed {idx}/{len(records)} records", flush=True)

    fieldnames = [
        "session",
        "rel_path",
        "annotation_key",
        "data_source",
        "split",
        "data_format",
        "usable_for_aligned_training",
        "task_id",
        "task_name",
        "language_instruction",
        "object_type",
        "object_label",
        "item_type",
        "material",
        "fragility",
        "slip_risk",
        "depth_uncertainty",
        "force_min",
        "force_max",
        "slot_pair",
        "slot_source",
        "slot_target",
        "success",
        "needs_review",
        "review_reasons",
        "frame_count",
        "stage1_start_frame_index",
        "stage1_start_stem",
        "contact_frame_index",
        "contact_frame_stem",
        "contact_source",
        "stage_segments_json",
        "vision_confidence",
        "vision_stage_visual_check",
        "vision_notes",
    ]
    manifest_path = out_dir / "annotation_manifest.csv"
    write_csv(manifest_path, manifest_rows, fieldnames)
    summary = {
        "schema": "transvtla_annotation_summary/v1",
        "records": len(records),
        "manifest": rel(manifest_path),
        "sidecars": rel(sidecar_dir),
        "counters": dict(counters),
        "vision_results_loaded": len(vision_results),
        "write_session_files": bool(args.write_session_files),
    }
    write_json(out_dir / "annotation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def copy_or_link_session(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst):
        return "exists"
    if mode == "copy":
        shutil.copytree(src, dst)
        return "copied"
    rel_target = os.path.relpath(src, dst.parent)
    dst.symlink_to(rel_target, target_is_directory=True)
    return "linked"


def command_materialize_real_1000(args: argparse.Namespace) -> None:
    manifest = read_csv(repo_path(args.annotation_manifest))
    rows = [
        row
        for row in manifest
        if row.get("data_source") == "real"
        and (args.allow_review or row.get("needs_review") in {"0", "", "False", "false"})
    ]
    base_rows = list(rows)
    candidate_rows = [
        row
        for row in rows
        if row.get("data_format") == "dual_realsense"
        and row.get("usable_for_aligned_training") in {"1", "true", "True", ""}
    ]
    if not candidate_rows:
        raise SystemExit("No dual RealSense candidates found for duplication.")

    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        by_task[row.get("task_id", "")].append(row)
    task_ids = sorted(k for k, v in by_task.items() if v)

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(base_rows):
        out_row = dict(row)
        out_row["view_index"] = index
        out_row["duplicate_of"] = ""
        out_row["source_rel_path"] = row["rel_path"]
        output_rows.append(out_row)

    cursor_by_task: dict[str, int] = defaultdict(int)
    task_cursor = 0
    while len(output_rows) < args.target_count:
        task_id = task_ids[task_cursor % len(task_ids)]
        task_cursor += 1
        task_rows = by_task[task_id]
        source = task_rows[cursor_by_task[task_id] % len(task_rows)]
        cursor_by_task[task_id] += 1
        out_row = dict(source)
        out_row["view_index"] = len(output_rows)
        out_row["duplicate_of"] = source.get("annotation_key") or source["session"]
        out_row["source_rel_path"] = source["rel_path"]
        output_rows.append(out_row)

    out_dir = repo_path(args.out_dir)
    session_out_dir = out_dir / "sessions"
    counters: Counter[str] = Counter()
    materialized_rows: list[dict[str, Any]] = []
    for row in output_rows:
        src = repo_path(row["source_rel_path"])
        name = f"{int(row['view_index']):04d}__{row['session']}__{stable_id(row['source_rel_path'], 6)}"
        dst = session_out_dir / name
        status = copy_or_link_session(src, dst, args.mode)
        counters[status] += 1
        out_row = dict(row)
        out_row["view_session"] = name
        out_row["view_rel_path"] = rel(dst)
        out_row["materialize_status"] = status
        materialized_rows.append(out_row)

    fieldnames = list(materialized_rows[0].keys()) if materialized_rows else []
    write_csv(out_dir / "manifest.csv", materialized_rows, fieldnames)
    with open(out_dir / "sessions.txt", "w", encoding="utf-8") as f:
        for row in materialized_rows:
            f.write(f"{row['view_rel_path']}\n")
    summary = {
        "schema": "transvtla_real_1000_view/v1",
        "target_count": args.target_count,
        "base_real_rows": len(base_rows),
        "duplicates_added": max(0, len(materialized_rows) - len(base_rows)),
        "candidate_rows": len(candidate_rows),
        "mode": args.mode,
        "manifest": rel(out_dir / "manifest.csv"),
        "sessions_dir": rel(session_out_dir),
        "counters": dict(counters),
        "by_task": dict(Counter(row.get("task_id", "") for row in materialized_rows)),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def row_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "") or default))
    except ValueError:
        return default


def is_effective_label_change(prev: dict[str, str], cur: dict[str, str]) -> bool:
    if prev.get("task_id") != cur.get("task_id"):
        return False
    if cur.get("task_id") in {TASK1, TASK2}:
        return prev.get("object_type", "") != cur.get("object_type", "")
    if cur.get("task_id") == TASK4:
        return prev.get("slot_pair", "") != cur.get("slot_pair", "")
    return False


def command_validate_neighbor_consistency(args: argparse.Namespace) -> None:
    rows = read_csv(repo_path(args.annotation_manifest))
    rows = [row for row in rows if row.get("data_source") == "real"]
    rows.sort(
        key=lambda row: (
            row.get("split", ""),
            row.get("task_id", ""),
            row.get("rel_path", ""),
        )
    )

    by_sheet: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    task_rows = read_csv(repo_path(args.task_manifest))
    sheet_by_rel = {
        row["rel_path"]: row.get("task_sheet_id", "") for row in task_rows if row.get("rel_path")
    }
    row_in_sheet_by_rel = {
        row["rel_path"]: row.get("task_row_in_sheet", "") for row in task_rows if row.get("rel_path")
    }
    for row in rows:
        sheet = sheet_by_rel.get(row["rel_path"], "")
        if not sheet:
            sheet = row.get("split", "")
        row["task_sheet_id"] = sheet
        row["task_row_in_sheet"] = row_in_sheet_by_rel.get(row["rel_path"], "")
        by_sheet[(row.get("task_id", ""), sheet)].append(row)

    issue_rows: list[dict[str, Any]] = []
    flagged_reasons: dict[str, set[str]] = defaultdict(set)
    summary_counter: Counter[str] = Counter()
    for (task_id, sheet), group in by_sheet.items():
        group.sort(
            key=lambda row: (
                row_int(row, "task_row_in_sheet", 0),
                row.get("session", ""),
            )
        )
        label_keys = ["object_type"] if task_id in {TASK1, TASK2} else ["slot_pair"] if task_id == TASK4 else []
        if not label_keys:
            continue
        labels = [tuple(row.get(k, "") for k in label_keys) for row in group]
        non_empty = [label for label in labels if any(label)]
        if len(set(non_empty)) > 1:
            summary_counter["sheet_multi_label"] += 1
            issue_rows.append(
                {
                    "issue_type": "sheet_multi_label",
                    "task_id": task_id,
                    "task_sheet_id": sheet,
                    "session": "",
                    "neighbor_session": "",
                    "label_before": "",
                    "label_after": "",
                    "details": f"sheet has labels {sorted(set(non_empty))}",
                }
            )
        changes = 0
        for prev, cur in zip(group, group[1:]):
            if not is_effective_label_change(prev, cur):
                continue
            changes += 1
            flagged_reasons[prev["rel_path"]].add("neighbor_label_change")
            flagged_reasons[cur["rel_path"]].add("neighbor_label_change")
            summary_counter["neighbor_label_change"] += 1
            issue_rows.append(
                {
                    "issue_type": "neighbor_label_change",
                    "task_id": task_id,
                    "task_sheet_id": sheet,
                    "session": prev.get("session", ""),
                    "neighbor_session": cur.get("session", ""),
                    "label_before": prev.get("object_type") or prev.get("slot_pair", ""),
                    "label_after": cur.get("object_type") or cur.get("slot_pair", ""),
                    "details": "adjacent sessions in the same task sheet changed label",
                }
            )
        if changes >= args.max_changes_per_sheet:
            summary_counter["sheet_many_changes"] += 1
            for row in group:
                flagged_reasons[row["rel_path"]].add("sheet_many_changes")
            issue_rows.append(
                {
                    "issue_type": "sheet_many_changes",
                    "task_id": task_id,
                    "task_sheet_id": sheet,
                    "session": "",
                    "neighbor_session": "",
                    "label_before": "",
                    "label_after": "",
                    "details": f"{changes} adjacent label changes in one task sheet",
                }
            )

    out_dir = repo_path(args.out_dir)
    fieldnames = [
        "issue_type",
        "task_id",
        "task_sheet_id",
        "session",
        "neighbor_session",
        "label_before",
        "label_after",
        "details",
    ]
    issues_path = out_dir / "neighbor_consistency_issues.csv"
    write_csv(issues_path, issue_rows, fieldnames)
    flag_rows = [
        {
            "rel_path": rel_path,
            "session": next((row["session"] for row in rows if row["rel_path"] == rel_path), ""),
            "reasons": "|".join(sorted(reasons)),
        }
        for rel_path, reasons in sorted(flagged_reasons.items())
    ]
    flags_path = out_dir / "neighbor_consistency_session_flags.csv"
    write_csv(flags_path, flag_rows, ["rel_path", "session", "reasons"])

    all_rows = read_csv(repo_path(args.annotation_manifest))
    checked_rows: list[dict[str, Any]] = []
    for row in all_rows:
        checked = dict(row)
        reasons = set(filter(None, checked.get("review_reasons", "").split("|")))
        if checked.get("rel_path") in flagged_reasons:
            checked["needs_review"] = "1"
            reasons.update(flagged_reasons[checked["rel_path"]])
        checked["review_reasons"] = "|".join(sorted(reasons))
        checked_rows.append(checked)
    checked_manifest = out_dir / "annotation_manifest_consistency_checked.csv"
    write_csv(
        checked_manifest,
        checked_rows,
        list(checked_rows[0].keys()) if checked_rows else [],
    )
    summary = {
        "schema": "transvtla_neighbor_consistency/v1",
        "real_rows": len(rows),
        "groups": len(by_sheet),
        "issues": len(issue_rows),
        "flagged_sessions": len(flag_rows),
        "issues_path": rel(issues_path),
        "flags_path": rel(flags_path),
        "checked_manifest": rel(checked_manifest),
        "counters": dict(summary_counter),
    }
    write_json(out_dir / "neighbor_consistency_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def effective_label(row: dict[str, str]) -> str:
    if row.get("task_id") in {TASK1, TASK2}:
        return row.get("object_type", "")
    if row.get("task_id") == TASK4:
        return row.get("slot_pair", "")
    return ""


def command_prepare_consistency_rerun(args: argparse.Namespace) -> None:
    requests = read_jsonl(repo_path(args.requests))
    requests_by_rel = {str(row.get("rel_path", "")): row for row in requests}
    manifest_rows = read_csv(repo_path(args.annotation_manifest))
    manifest_by_rel = {row["rel_path"]: row for row in manifest_rows if row.get("rel_path")}
    flag_rows = read_csv(repo_path(args.flags))
    flagged_by_rel = {row["rel_path"]: row for row in flag_rows if row.get("rel_path")}
    issue_details_by_session: dict[str, list[str]] = defaultdict(list)
    if args.issues:
        for row in read_csv(repo_path(args.issues)):
            for key in ("session", "neighbor_session"):
                session = row.get(key, "")
                if session:
                    issue_details_by_session[session].append(
                        f"{row.get('issue_type', '')}:{row.get('label_before', '')}->{row.get('label_after', '')}:{row.get('details', '')}"
                    )

    task_rows = read_csv(repo_path(args.task_manifest))
    sheet_by_rel = {row["rel_path"]: row.get("task_sheet_id", "") for row in task_rows if row.get("rel_path")}
    row_in_sheet_by_rel = {
        row["rel_path"]: row.get("task_row_in_sheet", "") for row in task_rows if row.get("rel_path")
    }

    by_sheet: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        if row.get("data_source") != "real":
            continue
        task_id = row.get("task_id", "")
        if task_id not in {TASK1, TASK2, TASK4}:
            continue
        sheet = sheet_by_rel.get(row["rel_path"], row.get("split", ""))
        enriched = dict(row)
        enriched["task_sheet_id"] = sheet
        enriched["task_row_in_sheet"] = row_in_sheet_by_rel.get(row["rel_path"], "")
        by_sheet[(task_id, sheet)].append(enriched)
    for group in by_sheet.values():
        group.sort(key=lambda row: (row_int(row, "task_row_in_sheet", 0), row.get("session", "")))

    output_rows: list[dict[str, Any]] = []
    missing: Counter[str] = Counter()
    for rel_path, flag in sorted(flagged_by_rel.items(), key=lambda item: item[0]):
        request = requests_by_rel.get(rel_path)
        manifest = manifest_by_rel.get(rel_path)
        if request is None or manifest is None:
            missing["request_or_manifest"] += 1
            continue
        task_id = manifest.get("task_id", "")
        sheet = sheet_by_rel.get(rel_path, manifest.get("split", ""))
        group = by_sheet.get((task_id, sheet), [])
        idx = next((i for i, row in enumerate(group) if row.get("rel_path") == rel_path), -1)
        prev_label = effective_label(group[idx - 1]) if idx > 0 else ""
        next_label = effective_label(group[idx + 1]) if 0 <= idx < len(group) - 1 else ""
        label_counts = Counter(effective_label(row) for row in group if effective_label(row))
        majority_label = label_counts.most_common(1)[0][0] if label_counts else ""

        rerun = dict(request)
        review_pass = str(args.review_pass or "consistency")
        rerun.update(
            {
                "consistency_review": True,
                "third_pass_review": review_pass == "third",
                "review_pass": review_pass,
                "consistency_reasons": flag.get("reasons", ""),
                "issue_details": " || ".join(issue_details_by_session.get(flag.get("session", ""), [])),
                "first_pass_object_type": manifest.get("object_type", ""),
                "first_pass_slot_pair": manifest.get("slot_pair", ""),
                "first_pass_stage_visual_check": manifest.get("vision_stage_visual_check", ""),
                "first_pass_confidence": manifest.get("vision_confidence", ""),
                "task_sheet_id": sheet,
                "task_row_in_sheet": row_in_sheet_by_rel.get(rel_path, ""),
                "sheet_label_counts": dict(label_counts),
                "sheet_majority_label": majority_label,
                "previous_session_label": prev_label,
                "next_session_label": next_label,
            }
        )
        output_rows.append(rerun)
        if args.limit and len(output_rows) >= args.limit:
            break

    output_path = repo_path(args.output)
    append_jsonl(output_path, output_rows)
    summary = {
        "schema": "transvtla_consistency_rerun_requests/v1",
        "flags": len(flag_rows),
        "requests": len(output_rows),
        "output": rel(output_path),
        "missing": dict(missing),
    }
    write_json(output_path.parent / "consistency_rerun_requests_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_merge_vision_results(args: argparse.Namespace) -> None:
    base_rows = read_jsonl(repo_path(args.base_results))
    override_rows = read_jsonl(repo_path(args.override_results))
    review_label = str(getattr(args, "review_label", "consistency_review") or "consistency_review")
    override_by_rel = {
        str(row.get("rel_path", "")): row
        for row in override_rows
        if row.get("rel_path") and not str(row.get("notes", "")).startswith("qwen3vl_failed:")
    }
    merged: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in base_rows:
        rel_path = str(row.get("rel_path", ""))
        override = override_by_rel.get(rel_path)
        if override is None:
            merged.append(row)
            counters["kept_base"] += 1
            continue
        merged_row = dict(row)
        merged_row.update(override)
        merged_row["notes"] = (
            f"{review_label}: {override.get('notes', '')} "
            f"| first_pass_object_type={row.get('object_type', '')} "
            f"| first_pass_slot_pair={row.get('slot_pair', '')}"
        )
        merged.append(merged_row)
        counters["overridden"] += 1

    output_path = repo_path(args.output)
    append_jsonl(output_path, merged)
    summary = {
        "schema": "transvtla_merged_vision_results/v1",
        "base": len(base_rows),
        "override": len(override_rows),
        "merged": len(merged),
        "output": rel(output_path),
        "counters": dict(counters),
    }
    write_json(output_path.parent / "merge_vision_results_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-manifest", default=str(DEFAULT_TASK_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--lead-frames", type=int, default=40)
    parser.add_argument("--pressure-threshold", type=float, default=0.05)
    parser.add_argument("--consecutive-pressure-rows", type=int, default=5)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-vision", help="Create vision-model image requests.")
    add_common_args(prepare)
    prepare.add_argument("--max-vision-requests", type=int, default=0)
    prepare.add_argument("--no-images", action="store_true")
    prepare.set_defaults(func=command_prepare_vision)

    run_vision = subparsers.add_parser(
        "run-openai-vision",
        help="Run an OpenAI-compatible vision model over prepared JSONL requests.",
    )
    run_vision.add_argument(
        "--requests",
        default=str(DEFAULT_OUT_DIR / "vision_inputs" / "vision_requests.jsonl"),
    )
    run_vision.add_argument(
        "--output",
        default=str(DEFAULT_OUT_DIR / "vision_results.jsonl"),
    )
    run_vision.add_argument("--api-url", default="https://api.openai.com/v1/responses")
    run_vision.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_vision.add_argument("--model", default="")
    run_vision.add_argument("--temperature", type=float, default=0.0)
    run_vision.add_argument("--timeout", type=int, default=120)
    run_vision.add_argument("--limit", type=int, default=0)
    run_vision.add_argument("--resume", action="store_true")
    run_vision.add_argument("--sleep-seconds", type=float, default=0.0)
    run_vision.add_argument("--progress-every", type=int, default=25)
    run_vision.set_defaults(func=command_run_openai_vision)

    run_qwen = subparsers.add_parser(
        "run-qwen3vl-local",
        help="Run local Qwen3-VL over prepared JSONL requests.",
    )
    run_qwen.add_argument(
        "--requests",
        default=str(DEFAULT_OUT_DIR / "vision_inputs" / "vision_requests.jsonl"),
    )
    run_qwen.add_argument(
        "--output",
        default=str(DEFAULT_OUT_DIR / "vision_results_qwen3vl.jsonl"),
    )
    run_qwen.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    run_qwen.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    run_qwen.add_argument("--device-map", default="auto")
    run_qwen.add_argument("--attn-implementation", default="")
    run_qwen.add_argument("--max-new-tokens", type=int, default=160)
    run_qwen.add_argument("--limit", type=int, default=0)
    run_qwen.add_argument("--resume", action="store_true")
    run_qwen.add_argument("--progress-every", type=int, default=25)
    run_qwen.set_defaults(func=command_run_qwen3vl_local)

    build = subparsers.add_parser("build-annotations", help="Build final annotation manifest.")
    add_common_args(build)
    build.add_argument("--vision-results", default="")
    build.add_argument("--sim-shards", default=str(DEFAULT_SIM_SHARDS))
    build.add_argument("--include-sim", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--write-session-files", action="store_true")
    build.set_defaults(func=command_build_annotations)

    mat = subparsers.add_parser("materialize-real-1000", help="Create a 1000-entry real training view.")
    mat.add_argument("--annotation-manifest", default=str(DEFAULT_OUT_DIR / "annotation_manifest.csv"))
    mat.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "real_1000_train_view"))
    mat.add_argument("--target-count", type=int, default=1000)
    mat.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    mat.add_argument("--allow-review", action="store_true")
    mat.set_defaults(func=command_materialize_real_1000)

    validate = subparsers.add_parser(
        "validate-neighbor-consistency",
        help="Flag adjacent real sessions whose visual labels change unexpectedly.",
    )
    validate.add_argument("--annotation-manifest", default=str(DEFAULT_OUT_DIR / "annotation_manifest.csv"))
    validate.add_argument("--task-manifest", default=str(DEFAULT_TASK_MANIFEST))
    validate.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "validation"))
    validate.add_argument("--max-changes-per-sheet", type=int, default=2)
    validate.set_defaults(func=command_validate_neighbor_consistency)

    rerun = subparsers.add_parser(
        "prepare-consistency-rerun",
        help="Create second-pass vision requests for sessions flagged by neighbor consistency.",
    )
    rerun.add_argument(
        "--requests",
        default=str(DEFAULT_OUT_DIR / "vision_inputs" / "vision_requests.jsonl"),
    )
    rerun.add_argument(
        "--annotation-manifest",
        default=str(Path("dataset/auto_annotations_qwen3vl/annotation_manifest.csv")),
    )
    rerun.add_argument(
        "--flags",
        default=str(Path("dataset/auto_annotations_qwen3vl/validation/neighbor_consistency_session_flags.csv")),
    )
    rerun.add_argument("--task-manifest", default=str(DEFAULT_TASK_MANIFEST))
    rerun.add_argument("--issues", default="")
    rerun.add_argument("--review-pass", choices=["consistency", "third"], default="consistency")
    rerun.add_argument(
        "--output",
        default=str(Path("dataset/auto_annotations_qwen3vl/consistency_rerun/vision_requests_consistency.jsonl")),
    )
    rerun.add_argument("--limit", type=int, default=0)
    rerun.set_defaults(func=command_prepare_consistency_rerun)

    merge = subparsers.add_parser(
        "merge-vision-results",
        help="Merge second-pass vision results over first-pass results by rel_path.",
    )
    merge.add_argument("--base-results", default=str(DEFAULT_OUT_DIR / "vision_results_qwen3vl.jsonl"))
    merge.add_argument(
        "--override-results",
        default=str(Path("dataset/auto_annotations_qwen3vl/consistency_rerun/vision_results_consistency.jsonl")),
    )
    merge.add_argument(
        "--output",
        default=str(Path("dataset/auto_annotations_qwen3vl/vision_results_qwen3vl_merged.jsonl")),
    )
    merge.add_argument("--review-label", default="consistency_review")
    merge.set_defaults(func=command_merge_vision_results)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
