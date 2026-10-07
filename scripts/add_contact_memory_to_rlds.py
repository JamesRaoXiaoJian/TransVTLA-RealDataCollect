#!/usr/bin/env python3
"""Add calibrated contact-memory tensors to an existing TransVTLA RLDS dataset.

The source dataset is left untouched. This script rewrites TFRecord shards into
a new TFDS version directory and appends per-step observation fields:

  contact_memory       float32 [6, 3]
  valid_contact_mask   bool    [6]
  contact_type         int64   [6]
  contact_reliability  float32 [6]
  contact_time         int64   [6]

Contact positions are generated from the original per-session pressure.csv
instead of RLDS observation.tactile, so baseline drift can be handled from raw
ADC values.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RXJ_ROOT = Path("/home/james/RXJ/TransVTLA-RealDataCollect")
# RXJ_ROOT must take precedence because /media/files/TransVTLA-RealDataCollect
# also has a fuser package but it does not contain gripper_touch.py.
for _path in (REPO_ROOT, RXJ_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from channel_config import (  # noqa: E402
    INTERPOLATE_CHANNELS,
    LEFT_CHANNEL,
    LEFT_MATRIX_CHANNELS,
    RIGHT_CHANNEL,
    RIGHT_MATRIX_CHANNELS,
    VALID_CHANNELS,
)
from fuser.gripper_touch import (  # noqa: E402
    default_calibration,
    estimate_contact_from_pressure,
    pressure_drop_by_channel,
    rpy_xyz_to_matrix,
)
from scripts.build_annotated_rlds import (  # noqa: E402
    EpisodeRow,
    aligned_state,
    load_gripper_csv,
    load_robot_csv,
    nearest_indices,
    parse_bool,
    repo_path,
    stage_arrays,
    value_or,
)
from scripts.patch_rlds_tactile import legacy_rlds_visual_stems  # noqa: E402
from session_schema import gripper_state_csv_path, robot_state_csv_path  # noqa: E402


DATASET_NAME = "transvtla_annotated_rlds"
STATE_KEY = "steps/observation/state"
SOURCE_FILE_KEY = "episode_metadata/source_file"
ANNOTATION_KEY_KEY = "episode_metadata/annotation_key"

CONTACT_MEMORY_KEY = "steps/observation/contact_memory"
VALID_CONTACT_MASK_KEY = "steps/observation/valid_contact_mask"
CONTACT_TYPE_KEY = "steps/observation/contact_type"
CONTACT_RELIABILITY_KEY = "steps/observation/contact_reliability"
CONTACT_TIME_KEY = "steps/observation/contact_time"

CONTACT_TYPE_ACTIVE = 0
CONTACT_TYPE_GRASP = 1
NUM_SLOTS = 6
ACTIVE_SLOTS = range(0, 4)
LEFT_GRASP_SLOT = 4
RIGHT_GRASP_SLOT = 5

REGION_LEFT_TIP = "left_tip"
REGION_RIGHT_TIP = "right_tip"
REGION_LEFT_INNER = "left_inner"
REGION_RIGHT_INNER = "right_inner"
REGIONS = (REGION_LEFT_TIP, REGION_RIGHT_TIP, REGION_LEFT_INNER, REGION_RIGHT_INNER)
LEFT_REGIONS = (REGION_LEFT_TIP, REGION_LEFT_INNER)
RIGHT_REGIONS = (REGION_RIGHT_TIP, REGION_RIGHT_INNER)

REGION_CHANNELS: dict[str, tuple[int, ...]] = {
    REGION_LEFT_TIP: (LEFT_CHANNEL,),
    REGION_RIGHT_TIP: (RIGHT_CHANNEL,),
    REGION_LEFT_INNER: tuple(ch for row in LEFT_MATRIX_CHANNELS for ch in row),
    REGION_RIGHT_INNER: tuple(ch for row in RIGHT_MATRIX_CHANNELS for ch in row),
}
CHANNEL_INDEX = {int(channel): idx for idx, channel in enumerate(VALID_CHANNELS)}
REGION_INDICES = {
    region: tuple(CHANNEL_INDEX[int(channel)] for channel in channels)
    for region, channels in REGION_CHANNELS.items()
}


@dataclass(frozen=True)
class ContactDecisionConfig:
    baseline_rows: int = 50
    min_delta_floor: float = 20.0
    noise_k: float = 4.0
    min_region_strength: float = 20.0
    stable_frames: int = 3
    second_best_margin_ratio: float = 1.2
    new_anchor_distance_m: float = 0.015
    grasp_min_region_strength: float = 20.0


@dataclass
class ContactTrack:
    contact_memory: np.ndarray
    valid_contact_mask: np.ndarray
    contact_type: np.ndarray
    contact_reliability: np.ndarray
    contact_time: np.ndarray
    active_events: int
    grasp_events: int
    valid_steps: int


@dataclass(frozen=True)
class EpisodeContactResult:
    track: ContactTrack
    source_file: str
    annotation_key: str
    pressure_csv: str
    frames: int
    nonzero_slots: int
    active_events: int
    grasp_events: int
    max_reliability: float
    max_region_strength: float


def robust_pressure_stats(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel robust baseline and MAD noise for raw ADC rows."""

    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != len(VALID_CHANNELS):
        raise ValueError(f"pressure rows must be [N,{len(VALID_CHANNELS)}], got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("cannot estimate baseline from empty pressure rows")
    baseline = np.median(arr, axis=0).astype(np.float32)
    mad = np.median(np.abs(arr - baseline[None, :]), axis=0).astype(np.float32)
    return baseline, mad


def apply_interpolation_matrix(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    squeeze = False
    if out.ndim == 1:
        out = out[None, :]
        squeeze = True
    for bad_channel, adjacent_channels in INTERPOLATE_CHANNELS.items():
        bad_idx = CHANNEL_INDEX.get(int(bad_channel))
        adj_indices = [CHANNEL_INDEX[int(ch)] for ch in adjacent_channels if int(ch) in CHANNEL_INDEX]
        if bad_idx is None or not adj_indices:
            continue
        out[:, bad_idx] = out[:, adj_indices].mean(axis=1)
    return out[0] if squeeze else out


def pressure_delta(current_rows: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    current = apply_interpolation_matrix(current_rows)
    base = apply_interpolation_matrix(baseline)
    return np.maximum(base[None, :] - current, 0.0).astype(np.float32)


def region_strengths_from_delta(delta: np.ndarray) -> np.ndarray:
    strengths = np.zeros((delta.shape[0], len(REGIONS)), dtype=np.float32)
    for region_idx, region in enumerate(REGIONS):
        strengths[:, region_idx] = delta[:, REGION_INDICES[region]].sum(axis=1)
    return strengths


def dynamic_region_thresholds(noise: np.ndarray, config: ContactDecisionConfig) -> np.ndarray:
    noise_interp = apply_interpolation_matrix(noise)
    channel_thresholds = config.min_delta_floor + config.noise_k * np.maximum(noise_interp, 0.0)
    thresholds = np.zeros((len(REGIONS),), dtype=np.float32)
    for region_idx, region in enumerate(REGIONS):
        thresholds[region_idx] = max(
            float(config.min_region_strength),
            float(channel_thresholds[list(REGION_INDICES[region])].sum()),
        )
    return thresholds


def _stable_candidate(candidate: np.ndarray, labels: np.ndarray, t: int, frames: int) -> bool:
    start = t - frames + 1
    if start < 0:
        return False
    if not bool(candidate[start : t + 1].all()):
        return False
    return bool(np.all(labels[start : t + 1] == labels[t]))


def _reliability(
    best_strength: float,
    threshold: float,
    second_strength: float,
    stable_frames: int,
    config: ContactDecisionConfig,
) -> float:
    strength_score = best_strength / max(threshold, 1e-6)
    if second_strength <= 1e-6:
        margin_score = 1.0
    else:
        margin_score = max(0.0, min(1.0, ((best_strength / second_strength) - 1.0)))
    reliability = (
        0.45 * max(0.0, min(1.0, strength_score / 3.0))
        + 0.25 * max(0.0, min(1.0, stable_frames / max(config.stable_frames, 1)))
        + 0.20 * margin_score
        + 0.10
    )
    return float(max(0.0, min(1.0, reliability)))


def _has_near_anchor(memory: np.ndarray, valid: np.ndarray, xyz: np.ndarray, config: ContactDecisionConfig) -> bool:
    active_valid = valid[list(ACTIVE_SLOTS)]
    if not bool(active_valid.any()):
        return False
    active_points = memory[list(ACTIVE_SLOTS)][active_valid]
    distances = np.linalg.norm(active_points - xyz[None, :], axis=1)
    return bool(np.any(distances <= config.new_anchor_distance_m))


def _first_free_active_slot(valid: np.ndarray) -> int | None:
    for slot in ACTIVE_SLOTS:
        if not bool(valid[slot]):
            return int(slot)
    return None


def build_contact_track(
    current_rows: np.ndarray,
    baseline: np.ndarray,
    noise: np.ndarray,
    *,
    use_tactile: np.ndarray,
    contact_xyz: np.ndarray,
    contact_region: np.ndarray,
    config: ContactDecisionConfig,
    region_xyz: Mapping[str, np.ndarray] | None = None,
) -> ContactTrack:
    """Build causal [T,6,*] contact-memory tensors from aligned pressure rows."""

    rows = np.asarray(current_rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != len(VALID_CHANNELS):
        raise ValueError(f"current_rows must be [T,{len(VALID_CHANNELS)}], got {rows.shape}")
    num_steps = rows.shape[0]
    use_tactile = np.asarray(use_tactile, dtype=bool)[:num_steps]
    if use_tactile.shape[0] != num_steps:
        raise ValueError("use_tactile length must match current_rows")
    contact_xyz = np.asarray(contact_xyz, dtype=np.float32)
    if contact_xyz.shape != (num_steps, 3):
        raise ValueError(f"contact_xyz must be [T,3], got {contact_xyz.shape}")
    contact_region = np.asarray(contact_region)
    if contact_region.shape[0] != num_steps:
        raise ValueError("contact_region length must match current_rows")

    delta = pressure_delta(rows, baseline)
    strengths = region_strengths_from_delta(delta)
    thresholds = dynamic_region_thresholds(noise, config)
    best_idx = strengths.argmax(axis=1)
    best_strength = strengths[np.arange(num_steps), best_idx]
    second_strength = np.partition(strengths, -2, axis=1)[:, -2]
    margin_ok = np.where(
        second_strength <= 1e-6,
        True,
        best_strength >= second_strength * float(config.second_best_margin_ratio),
    )
    candidate = use_tactile & margin_ok & (best_strength >= thresholds[best_idx])

    memory = np.zeros((num_steps, NUM_SLOTS, 3), dtype=np.float32)
    valid = np.zeros((num_steps, NUM_SLOTS), dtype=bool)
    contact_type = np.zeros((num_steps, NUM_SLOTS), dtype=np.int64)
    contact_type[:, LEFT_GRASP_SLOT:] = CONTACT_TYPE_GRASP
    reliability = np.zeros((num_steps, NUM_SLOTS), dtype=np.float32)
    contact_time = np.full((num_steps, NUM_SLOTS), -1, dtype=np.int64)
    active_events = 0
    grasp_events = 0

    region_labels = np.array([REGIONS[i] for i in best_idx])
    for t in range(num_steps):
        if t > 0:
            memory[t] = memory[t - 1]
            valid[t] = valid[t - 1]
            contact_type[t] = contact_type[t - 1]
            reliability[t] = reliability[t - 1]
            contact_time[t] = contact_time[t - 1]

        if not _stable_candidate(candidate, region_labels, t, max(1, int(config.stable_frames))):
            continue

        xyz = contact_xyz[t]
        if not np.isfinite(xyz).all() or not str(contact_region[t]):
            continue
        rel = _reliability(
            float(best_strength[t]),
            float(thresholds[best_idx[t]]),
            float(second_strength[t]),
            config.stable_frames,
            config,
        )
        if not _has_near_anchor(memory[t], valid[t], xyz, config):
            slot = _first_free_active_slot(valid[t])
            if slot is not None:
                memory[t, slot] = xyz
                valid[t, slot] = True
                contact_type[t, slot] = CONTACT_TYPE_ACTIVE
                reliability[t, slot] = rel
                contact_time[t, slot] = t
                active_events += 1

        if region_xyz is None:
            continue
        left_idx = max((REGIONS.index(region) for region in LEFT_REGIONS), key=lambda idx: strengths[t, idx])
        right_idx = max((REGIONS.index(region) for region in RIGHT_REGIONS), key=lambda idx: strengths[t, idx])
        left_region = REGIONS[left_idx]
        right_region = REGIONS[right_idx]
        left_ok = strengths[t, left_idx] >= max(thresholds[left_idx], config.grasp_min_region_strength)
        right_ok = strengths[t, right_idx] >= max(thresholds[right_idx], config.grasp_min_region_strength)
        if left_ok and right_ok:
            for slot, region in ((LEFT_GRASP_SLOT, left_region), (RIGHT_GRASP_SLOT, right_region)):
                if bool(valid[t, slot]):
                    continue
                points = region_xyz.get(region)
                if points is None or t >= len(points) or not np.isfinite(points[t]).all():
                    continue
                memory[t, slot] = points[t]
                valid[t, slot] = True
                contact_type[t, slot] = CONTACT_TYPE_GRASP
                reliability[t, slot] = rel
                contact_time[t, slot] = t
                grasp_events += 1

    return ContactTrack(
        contact_memory=memory,
        valid_contact_mask=valid,
        contact_type=contact_type,
        contact_reliability=reliability,
        contact_time=contact_time,
        active_events=active_events,
        grasp_events=grasp_events,
        valid_steps=int(np.count_nonzero(valid.any(axis=1))),
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def decode_one_text(feature: Any) -> str:
    values = feature.bytes_list.value
    return values[0].decode("utf-8", "ignore") if values else ""


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
    obs_features["contact_memory"] = tensor_feature("float32", ["6", "3"])
    obs_features["valid_contact_mask"] = tensor_feature("bool", ["6"])
    obs_features["contact_type"] = tensor_feature("int64", ["6"])
    obs_features["contact_reliability"] = tensor_feature("float32", ["6"])
    obs_features["contact_time"] = tensor_feature("int64", ["6"])
    write_json(path, features)


def update_dataset_info(work_dir: Path, version: str, shard_lengths: list[int], num_bytes: int) -> None:
    path = work_dir / "dataset_info.json"
    info = read_json(path)
    info["version"] = version
    if info.get("splits"):
        info["splits"][0]["numBytes"] = str(num_bytes)
        info["splits"][0]["shardLengths"] = [str(v) for v in shard_lengths]
    write_json(path, info)


def load_manifest(path: Path) -> tuple[dict[str, EpisodeRow], dict[str, EpisodeRow]]:
    by_key: dict[str, EpisodeRow] = {}
    by_source: dict[str, EpisodeRow] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            episode = EpisodeRow(index=idx, row=row, session_dir=repo_path(row["rel_path"]))
            by_key[episode.annotation_key] = episode
            by_source[episode.rel_path] = episode
    return by_key, by_source


def read_pressure_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    values: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        timestamp_col = "host_monotonic_us" if "host_monotonic_us" in fields else "timestamp_us"
        if timestamp_col not in fields and "sensor_timestamp_us" in fields:
            timestamp_col = "sensor_timestamp_us"
        channel_cols = [f"CH{ch}" for ch in VALID_CHANNELS]
        missing = [col for col in channel_cols if col not in fields]
        if missing:
            raise ValueError(f"missing pressure columns {missing} in {path}")
        for row in reader:
            ts_text = row.get(timestamp_col)
            if ts_text in (None, ""):
                continue
            try:
                ts = int(float(ts_text))
                vals = [float(row[f"CH{ch}"]) for ch in VALID_CHANNELS]
            except (TypeError, ValueError):
                continue
            timestamps.append(ts)
            values.append(vals)
    if not timestamps:
        raise ValueError(f"no timestamped pressure rows in {path}")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(values, dtype=np.float32)


def baseline_rows_for_episode(
    pressure_ts: np.ndarray,
    pressure_values: np.ndarray,
    visual_ts: np.ndarray,
    use_tactile: np.ndarray,
    config: ContactDecisionConfig,
) -> np.ndarray:
    tactile_indices = np.flatnonzero(use_tactile)
    if tactile_indices.size:
        cutoff = int(visual_ts[int(tactile_indices[0])])
        rows = pressure_values[pressure_ts < cutoff]
        if len(rows) >= max(5, min(config.baseline_rows, 10)):
            return rows[-max(config.baseline_rows, min(len(rows), config.baseline_rows)) :]
    return pressure_values[: min(len(pressure_values), config.baseline_rows)]


def row_to_channel_map(row: np.ndarray) -> dict[int, float]:
    return {int(channel): float(row[idx]) for idx, channel in enumerate(VALID_CHANNELS)}


def candidate_contacts(
    poses: np.ndarray,
    gripper: np.ndarray,
    current_rows: np.ndarray,
    baseline: np.ndarray,
    config: ContactDecisionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], float]:
    calibration = default_calibration()
    baseline_map = row_to_channel_map(baseline)
    contact_xyz = np.zeros((len(current_rows), 3), dtype=np.float32)
    contact_region = np.array([""] * len(current_rows), dtype=object)
    region_xyz = {region: np.zeros((len(current_rows), 3), dtype=np.float32) for region in REGIONS}
    max_strength = 0.0

    for t in range(len(current_rows)):
        T_base_flange = rpy_xyz_to_matrix(poses[t])
        raw_map = row_to_channel_map(current_rows[t])
        drops = pressure_drop_by_channel(calibration, raw_map, baseline_map)
        contact = estimate_contact_from_pressure(
            calibration,
            T_base_flange,
            float(gripper[t]),
            drops,
            min_region_strength=float(config.min_region_strength),
        )
        for region in REGIONS:
            p = (contact.get("regions") or {}).get(region, {}).get("p_base_m")
            if p is not None:
                region_xyz[region][t] = np.asarray(p, dtype=np.float32)
        p_base = contact.get("p_base_contact_m")
        region = str(contact.get("region") or "")
        if p_base is not None and region:
            contact_xyz[t] = np.asarray(p_base, dtype=np.float32)
            contact_region[t] = region
        max_strength = max(max_strength, float(contact.get("strength") or 0.0))
    return contact_xyz, contact_region, region_xyz, max_strength


def build_episode_contact(
    features: Any,
    manifest_by_key: Mapping[str, EpisodeRow],
    manifest_by_source: Mapping[str, EpisodeRow],
    config: ContactDecisionConfig,
) -> EpisodeContactResult:
    state_values = np.asarray(features[STATE_KEY].float_list.value, dtype=np.float32)
    if state_values.size % 7:
        raise ValueError(f"state length is not divisible by 7: {state_values.size}")
    n_steps = int(state_values.size // 7)
    source_file = decode_one_text(features[SOURCE_FILE_KEY])
    annotation_key = decode_one_text(features[ANNOTATION_KEY_KEY])
    episode = manifest_by_key.get(annotation_key) or manifest_by_source.get(source_file)
    if episode is None:
        raise KeyError(f"manifest row not found for annotation_key={annotation_key} source_file={source_file}")

    stems, _layout, reason = legacy_rlds_visual_stems(episode.session_dir)
    if not stems:
        raise RuntimeError(f"{source_file}: {reason}")
    robot_ts, robot_poses = load_robot_csv(robot_state_csv_path(episode.session_dir), np)
    gripper_ts, gripper_pos = load_gripper_csv(gripper_state_csv_path(episode.session_dir), np)
    visual_ts, poses, gripper = aligned_state(
        episode.session_dir,
        stems,
        robot_ts,
        robot_poses,
        gripper_ts,
        gripper_pos,
        np,
    )
    n = min(n_steps, len(visual_ts), len(poses), len(gripper))
    if n != n_steps:
        raise RuntimeError(f"{source_file}: aligned length {n} does not match RLDS steps {n_steps}")
    visual_ts = visual_ts[:n]
    poses = poses[:n]
    gripper = gripper[:n]
    if not np.isfinite(gripper).all() or float(np.max(gripper)) <= 1.5:
        state = state_values.reshape(n_steps, 7)
        gripper = np.clip(state[:, 6], 0.0, 1.0).astype(np.float32) * 1000.0

    _stage_label, use_tactile, _phase_names = stage_arrays(episode.row, n, np)
    pressure_path = episode.session_dir / "pressure" / "pressure.csv"
    pressure_ts, pressure_values = read_pressure_csv(pressure_path)
    pressure_indices = nearest_indices(visual_ts, pressure_ts, np)
    current_rows = pressure_values[pressure_indices]
    baseline_source = baseline_rows_for_episode(pressure_ts, pressure_values, visual_ts, use_tactile, config)
    baseline, noise = robust_pressure_stats(baseline_source)
    contact_xyz, contact_region, region_xyz, max_strength = candidate_contacts(
        poses,
        gripper,
        current_rows,
        baseline,
        config,
    )
    track = build_contact_track(
        current_rows,
        baseline,
        noise,
        use_tactile=use_tactile,
        contact_xyz=contact_xyz,
        contact_region=contact_region,
        region_xyz=region_xyz,
        config=config,
    )
    return EpisodeContactResult(
        track=track,
        source_file=source_file,
        annotation_key=annotation_key,
        pressure_csv=str(pressure_path),
        frames=n,
        nonzero_slots=int(np.count_nonzero(track.valid_contact_mask)),
        active_events=int(track.active_events),
        grasp_events=int(track.grasp_events),
        max_reliability=float(track.contact_reliability.max()) if track.contact_reliability.size else 0.0,
        max_region_strength=float(max_strength),
    )


def clear_and_extend_float(feature: Any, values: np.ndarray) -> None:
    del feature.float_list.value[:]
    feature.float_list.value.extend(np.asarray(values, dtype=np.float32).reshape(-1).tolist())


def clear_and_extend_int64(feature: Any, values: np.ndarray) -> None:
    del feature.int64_list.value[:]
    feature.int64_list.value.extend(np.asarray(values, dtype=np.int64).reshape(-1).tolist())


def add_contact_to_example(
    raw: bytes,
    tf: Any,
    manifest_by_key: Mapping[str, EpisodeRow],
    manifest_by_source: Mapping[str, EpisodeRow],
    config: ContactDecisionConfig,
) -> tuple[bytes, EpisodeContactResult]:
    example = tf.train.Example.FromString(raw)
    features = example.features.feature
    result = build_episode_contact(features, manifest_by_key, manifest_by_source, config)
    track = result.track
    clear_and_extend_float(features[CONTACT_MEMORY_KEY], track.contact_memory)
    clear_and_extend_int64(features[VALID_CONTACT_MASK_KEY], track.valid_contact_mask.astype(np.int64))
    clear_and_extend_int64(features[CONTACT_TYPE_KEY], track.contact_type)
    clear_and_extend_float(features[CONTACT_RELIABILITY_KEY], track.contact_reliability)
    clear_and_extend_int64(features[CONTACT_TIME_KEY], track.contact_time)
    return example.SerializeToString(), result


def output_shard_name(index: int, total: int) -> str:
    return f"{DATASET_NAME}-train.tfrecord-{index:05d}-of-{total:05d}"


def resolve_shard_window(
    shards: list[Path],
    start_shard: int,
    end_shard: int | None,
    max_shards: int | None,
) -> list[tuple[int, Path]]:
    total = len(shards)
    if start_shard < 0:
        raise ValueError(f"start_shard must be >= 0, got {start_shard}")
    if start_shard > total:
        raise ValueError(f"start_shard {start_shard} is past shard count {total}")
    stop = total if end_shard is None else end_shard
    if stop < start_shard:
        raise ValueError(f"end_shard {stop} is before start_shard {start_shard}")
    if stop > total:
        raise ValueError(f"end_shard {stop} is past shard count {total}")
    if max_shards is not None:
        if max_shards < 0:
            raise ValueError(f"max_shards must be >= 0, got {max_shards}")
        stop = min(stop, start_shard + max_shards)
    return list(enumerate(shards[start_shard:stop], start=start_shard))


def missing_shard_indices(work_dir: Path, total_shards: int) -> list[int]:
    missing: list[int] = []
    for idx in range(total_shards):
        path = work_dir / output_shard_name(idx, total_shards)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(idx)
    return missing


def summarize_output_dataset(
    work_dir: Path,
    total_shards: int,
    source_version_dir: Path,
) -> dict[str, Any]:
    source_info = read_json(source_version_dir / "dataset_info.json")
    splits = source_info.get("splits") or []
    if not splits:
        raise SystemExit(f"Source dataset_info.json has no splits: {source_version_dir}")
    shard_lengths = [int(v) for v in splits[0].get("shardLengths", [])]
    if len(shard_lengths) != total_shards:
        raise SystemExit(
            f"Source shardLengths has {len(shard_lengths)} entries, expected {total_shards}"
        )
    total_output_bytes = 0

    for shard_idx in range(total_shards):
        path = work_dir / output_shard_name(shard_idx, total_shards)
        if not path.is_file():
            raise SystemExit(f"Missing output shard before publish: {path}")
        total_output_bytes += path.stat().st_size

    return {
        "shard_lengths": shard_lengths,
        "total_output_bytes": total_output_bytes,
        "total_examples": sum(shard_lengths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-version-dir",
        type=Path,
        default=Path(
            "/media/files/TransVTLA-RealDataCollect/dataset/"
            "rlds_real1000_strict/transvtla_annotated_rlds/3.1.0"
        ),
    )
    parser.add_argument("--output-version", default="3.2.0")
    parser.add_argument("--output-version-dir", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/media/files/TransVTLA-RealDataCollect/dataset/"
            "auto_annotations_qwen3vl_thirdpass/final_annotations/"
            "real_1000_train_view_strict/manifest.csv"
        ),
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=None)
    parser.add_argument(
        "--resume-work-dir",
        type=Path,
        default=None,
        help="Continue writing missing shards in an existing temporary output directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--baseline-rows", type=int, default=50)
    parser.add_argument("--min-delta-floor", type=float, default=20.0)
    parser.add_argument("--noise-k", type=float, default=4.0)
    parser.add_argument("--min-region-strength", type=float, default=20.0)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--second-best-margin-ratio", type=float, default=1.2)
    parser.add_argument("--new-anchor-distance-m", type=float, default=0.015)
    parser.add_argument("--summary-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import tensorflow as tf

    source_version_dir = args.source_version_dir
    if not source_version_dir.is_dir():
        raise SystemExit(f"Source version directory not found: {source_version_dir}")
    output_version_dir = args.output_version_dir
    if output_version_dir is None:
        output_version_dir = source_version_dir.parent / args.output_version

    all_shards = sorted(source_version_dir.glob(f"{DATASET_NAME}-train.tfrecord-*"))
    if not all_shards:
        raise SystemExit(f"No TFRecord shards found in {source_version_dir}")
    try:
        selected_shards = resolve_shard_window(
            all_shards,
            args.start_shard,
            args.end_shard,
            args.max_shards,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not selected_shards:
        raise SystemExit("Shard selection is empty")
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    config = ContactDecisionConfig(
        baseline_rows=args.baseline_rows,
        min_delta_floor=args.min_delta_floor,
        noise_k=args.noise_k,
        min_region_strength=args.min_region_strength,
        stable_frames=max(1, args.stable_frames),
        second_best_margin_ratio=args.second_best_margin_ratio,
        new_anchor_distance_m=args.new_anchor_distance_m,
    )
    manifest_by_key, manifest_by_source = load_manifest(args.manifest)
    total_shard_count = len(all_shards)
    resume_mode = args.resume_work_dir is not None
    if resume_mode:
        work_dir = args.resume_work_dir
        if not work_dir.is_dir():
            raise SystemExit(f"Resume work directory not found: {work_dir}")
        if output_version_dir.exists() and not args.overwrite:
            raise SystemExit(f"Output version directory already exists: {output_version_dir}")
        existing_missing = set(missing_shard_indices(work_dir, total_shard_count))
        selected_shards = [
            (shard_idx, src)
            for shard_idx, src in selected_shards
            if shard_idx in existing_missing
        ]
        trash_dirs: list[Path] = []
        if not (work_dir / "features.json").is_file() or not (work_dir / "dataset_info.json").is_file():
            copy_metadata(source_version_dir, work_dir)
    else:
        work_dir, trash_dirs = prepare_output_dir(output_version_dir, args.overwrite)
        copy_metadata(source_version_dir, work_dir)

    shard_lengths: list[int] = []
    total_examples = 0
    total_steps = 0
    total_nonzero_slots = 0
    total_active_events = 0
    total_grasp_events = 0
    total_output_bytes = 0
    max_reliability = 0.0
    max_region_strength = 0.0
    started = time.time()

    print(f"source={source_version_dir}")
    print(f"work_dir={work_dir}")
    print(f"output={output_version_dir}")
    print(f"manifest={args.manifest}")
    if selected_shards:
        selected_start = selected_shards[0][0]
        selected_end = selected_shards[-1][0] + 1
    else:
        selected_start = args.start_shard
        selected_end = args.start_shard
    print(
        f"shards={len(selected_shards)} selected=[{selected_start},{selected_end}) "
        f"total_shards={total_shard_count} resume={resume_mode} config={config}"
    )

    for local_idx, (shard_idx, src) in enumerate(selected_shards):
        dst = work_dir / output_shard_name(shard_idx, total_shard_count)
        shard_examples = 0
        shard_steps = 0
        with tf.io.TFRecordWriter(str(dst)) as writer:
            for raw in tf.compat.v1.io.tf_record_iterator(str(src)):
                rewritten, result = add_contact_to_example(
                    raw,
                    tf,
                    manifest_by_key,
                    manifest_by_source,
                    config,
                )
                writer.write(rewritten)
                shard_examples += 1
                shard_steps += result.frames
                total_examples += 1
                total_steps += result.frames
                total_nonzero_slots += result.nonzero_slots
                total_active_events += result.active_events
                total_grasp_events += result.grasp_events
                max_reliability = max(max_reliability, result.max_reliability)
                max_region_strength = max(max_region_strength, result.max_region_strength)

        shard_lengths.append(shard_examples)
        shard_bytes = dst.stat().st_size
        total_output_bytes += shard_bytes
        if local_idx == 0 or (local_idx + 1) % args.summary_every == 0 or local_idx == len(selected_shards) - 1:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[{local_idx + 1:04d}/{len(selected_shards):04d}] "
                f"global={shard_idx:05d}/{total_shard_count:05d} {src.name} "
                f"examples={shard_examples} steps={shard_steps} "
                f"total_examples={total_examples} total_steps={total_steps} "
                f"nonzero_slots={total_nonzero_slots} active_events={total_active_events} "
                f"grasp_events={total_grasp_events} elapsed_min={elapsed / 60.0:.1f}",
                flush=True,
            )

    complete_output = not missing_shard_indices(work_dir, total_shard_count)
    if complete_output:
        output_stats = summarize_output_dataset(work_dir, total_shard_count, source_version_dir)
        shard_lengths = output_stats["shard_lengths"]
        total_output_bytes = output_stats["total_output_bytes"]
        output_total_examples = output_stats["total_examples"]
    elif resume_mode:
        remaining = missing_shard_indices(work_dir, total_shard_count)
        preview = ", ".join(str(idx) for idx in remaining[:10])
        raise SystemExit(
            f"Resume work directory is still incomplete: {len(remaining)} shards missing "
            f"({preview}{'...' if len(remaining) > 10 else ''})"
        )
    else:
        output_total_examples = total_examples

    update_features_json(work_dir)
    update_dataset_info(work_dir, args.output_version, shard_lengths, total_output_bytes)
    summary = {
        "source_version_dir": str(source_version_dir),
        "output_version_dir": str(output_version_dir),
        "output_version": args.output_version,
        "manifest": str(args.manifest),
        "num_shards": total_shard_count if complete_output else len(selected_shards),
        "selected_shard_start": selected_start,
        "selected_shard_end": selected_end,
        "complete_output": complete_output,
        "resume_work_dir": str(args.resume_work_dir) if args.resume_work_dir else None,
        "num_examples": output_total_examples,
        "processed_run_examples": total_examples,
        "processed_run_steps": total_steps,
        "processed_run_nonzero_contact_slots": total_nonzero_slots,
        "processed_run_active_events": total_active_events,
        "processed_run_grasp_events": total_grasp_events,
        "max_reliability": max_reliability,
        "max_region_strength_adc_drop": max_region_strength,
        "contact_fields": {
            "contact_memory": ["float32", [6, 3]],
            "valid_contact_mask": ["bool", [6]],
            "contact_type": ["int64", [6]],
            "contact_reliability": ["float32", [6]],
            "contact_time": ["int64", [6]],
        },
        "coordinate_frame": "robot_base / TransVTLA_world",
        "unit": "meter",
        "pressure_source": "original pressure/pressure.csv via episode_metadata/source_file",
        "drift_handling": {
            "baseline_method": "per-channel median over pre-contact or first baseline_rows raw ADC rows",
            "noise_method": "per-channel MAD",
            "delta": "max(baseline_adc - current_adc, 0)",
        },
        "decision_config": config.__dict__,
        "notes": [
            "Original dataset version directory is not modified.",
            "dataset_statistics_*.json files are intentionally not copied.",
            "contact_time is causal and never points to a future timestep.",
            "When resuming, contact event counters describe only shards processed in that run.",
        ],
    }
    write_json(work_dir / "contact_memory_generation_summary.json", summary)
    if complete_output or not resume_mode:
        publish_output_dir(work_dir, output_version_dir, trash_dirs)
        print(f"done: {output_version_dir}")
        print(f"summary: {output_version_dir / 'contact_memory_generation_summary.json'}")
    else:
        print(f"partial output remains in: {work_dir}")
        print(f"summary: {work_dir / 'contact_memory_generation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
