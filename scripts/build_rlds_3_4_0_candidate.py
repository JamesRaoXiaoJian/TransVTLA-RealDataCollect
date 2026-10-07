#!/usr/bin/env python3
"""Build RLDS 3.4.0 candidate with estimated D435 camera-group metadata.

This script rewrites an existing TransVTLA RLDS version into a new candidate
version directory.  It never modifies the source version in-place.

The 3.4.0 candidate policy is deliberately conservative:

* d435_calib_0 keeps the original point cloud and is marked verified original.
* d435_shift_1 / d435_shift_2 regenerate point_cloud from primary_depth using
  group-specific image-drift-estimated T_base_cam.
* DJI pseudo-depth and needs_manual_review sessions are kept for RGB-action
  compatibility, but marked unavailable for metric pointcloud / LC training.

The estimated shift calibrations are not manual 2D-3D refined calibrations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


DATASET_NAME = "transvtla_annotated_rlds"
OBS_PREFIX = "steps/observation"
SOURCE_FILE_KEY = "episode_metadata/source_file"
PRIMARY_DEPTH_KEY = f"{OBS_PREFIX}/primary_depth"
STATE_KEY = f"{OBS_PREFIX}/state"
POINT_CLOUD_KEY = f"{OBS_PREFIX}/point_cloud"
NEXT_POINT_CLOUD_KEY = f"{OBS_PREFIX}/next_point_cloud"

DEFAULT_DATASET_ROOT = Path(
    "/media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_strict/transvtla_annotated_rlds"
)
DEFAULT_SOURCE_VERSION_DIR = DEFAULT_DATASET_ROOT / "3.3.0_candidate"
DEFAULT_OUTPUT_VERSION_DIR = DEFAULT_DATASET_ROOT / "3.4.0_candidate"
DEFAULT_REPO_ROOT = Path("/media/files/TransVTLA-RealDataCollect")
DEFAULT_TEST_PROJECT_ROOT = Path("/media/files/Test_TransVTLA")
DEFAULT_CAMERA_GROUP_CONFIG = (
    DEFAULT_TEST_PROJECT_ROOT / "configs" / "camera_groups" / "transvtla_world_camera_groups.json"
)
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_TEST_PROJECT_ROOT / "reports" / "rlds_3_4_0_candidate_manifest.csv"
TODAY = time.strftime("%Y%m%d")
DEFAULT_REPORT_JSON = DEFAULT_TEST_PROJECT_ROOT / "reports" / f"rlds_3_4_0_candidate_build_{TODAY}.json"
DEFAULT_REPORT_MD = DEFAULT_TEST_PROJECT_ROOT / "reports" / f"rlds_3_4_0_candidate_build_{TODAY}.md"

METRIC_GROUPS = {"d435_calib_0", "d435_shift_1", "d435_shift_2"}
SHIFT_GROUPS = {"d435_shift_1", "d435_shift_2"}
EXCLUDED_GROUPS = {"dji_primary_pseudo_depth", "needs_manual_review"}


@dataclass(frozen=True)
class CameraInfo:
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    depth_scale: float
    source: str


@dataclass(frozen=True)
class EpisodePolicy:
    session_id: str
    camera_group: str
    calibration_quality: str
    pointcloud_action: str
    use_for_rgb_action: str
    use_for_metric_pointcloud: bool
    use_for_lc_projection: str
    use_for_contact_memory_training: str
    extrinsic_refined_by_manual_keypoints: bool
    requires_overlay_review: bool
    risk_note: str
    T_base_cam_source: str
    T_base_cam_estimated: np.ndarray
    delta_T_estimated: np.ndarray
    pitch_delta_deg: float
    y_bias_raw_px: float


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


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


def decode_one_text(feature: Any) -> str:
    values = feature.bytes_list.value
    if not values:
        return ""
    return values[0].decode("utf-8", "ignore")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "candidate"}


def session_id_from_source_file(source_file: str) -> str:
    for part in Path(source_file).parts:
        if part.startswith("session_"):
            return part
    return ""


def scaled_intrinsics(K: np.ndarray, width: int, height: int, image_size: int) -> np.ndarray:
    K_scaled = K.astype(np.float64).copy()
    K_scaled[0, 0] *= image_size / float(width)
    K_scaled[0, 2] *= image_size / float(width)
    K_scaled[1, 1] *= image_size / float(height)
    K_scaled[1, 2] *= image_size / float(height)
    return K_scaled


def camera_to_world(points_cam: np.ndarray, R_base_cam: np.ndarray, t_base_cam: np.ndarray) -> np.ndarray:
    R_world_to_cam = R_base_cam.T
    t_world_to_cam = -R_world_to_cam @ t_base_cam
    return ((points_cam.astype(np.float64) - t_world_to_cam) @ R_world_to_cam).astype(np.float32)


def choose_indices(count: int, target_count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if count == target_count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, target_count, dtype=np.int64)


def depth_to_point_cloud(
    depth_u16: np.ndarray,
    camera: CameraInfo,
    image_size: int,
    num_points: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, int]:
    depth_m = depth_u16.astype(np.float32) * float(camera.depth_scale)
    valid = np.isfinite(depth_m) & (depth_m > min_depth_m) & (depth_m < max_depth_m)
    ys, xs = np.nonzero(valid)
    valid_count = int(xs.size)
    if valid_count == 0:
        return np.zeros((num_points, 3), dtype=np.float32), 0

    K = scaled_intrinsics(camera.K, camera.width, camera.height, image_size)
    z = depth_m[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - K[0, 2]) / K[0, 0] * z
    y = (ys.astype(np.float64) - K[1, 2]) / K[1, 1] * z
    points_cam = np.stack([x, y, z], axis=1).astype(np.float32)
    selected = choose_indices(points_cam.shape[0], num_points)
    points_world = camera_to_world(points_cam[selected], camera.R, camera.t)
    return points_world.astype(np.float32, copy=False), valid_count


def source_session_dir(repo_root: Path, source_file: str) -> Path | None:
    if not source_file:
        return None
    path = Path(source_file)
    if not path.is_absolute():
        path = repo_root / path
    return path if path.is_dir() else None


def camera_from_metadata(session_dir: Path, *, R: np.ndarray, t: np.ndarray, image_size: int) -> CameraInfo | None:
    metadata_path = session_dir / "camera_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = read_json(metadata_path)
        world = metadata["cameras"]["world_camera"]
        intrinsics = world["color"]["intrinsics"]
        depth_meta = world.get("depth", {})
        depth_scale = float(
            depth_meta.get(
                "saved_png_unit_m",
                depth_meta.get("sensor_depth_scale_m_per_unit", 0.001),
            )
        )
        K = np.asarray(intrinsics.get("K"), dtype=np.float64)
        if K.shape != (3, 3):
            K = np.asarray(
                [
                    [float(intrinsics["fx"]), 0.0, float(intrinsics["ppx"])],
                    [0.0, float(intrinsics["fy"]), float(intrinsics["ppy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        width = int(intrinsics.get("width", world.get("profile", {}).get("width", image_size)))
        height = int(intrinsics.get("height", world.get("profile", {}).get("height", image_size)))
    except Exception:
        return None
    return CameraInfo(
        name="world_camera",
        K=K,
        R=np.asarray(R, dtype=np.float64).reshape(3, 3),
        t=np.asarray(t, dtype=np.float64).reshape(3),
        width=width,
        height=height,
        depth_scale=depth_scale,
        source="session_metadata_world_camera_group_estimated",
    )


def fallback_camera(*, R: np.ndarray, t: np.ndarray) -> CameraInfo:
    K = np.asarray(
        [
            [602.716614, 0.0, 432.202942],
            [0.0, 603.465698, 249.251740],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return CameraInfo(
        name="world_camera",
        K=K,
        R=np.asarray(R, dtype=np.float64).reshape(3, 3),
        t=np.asarray(t, dtype=np.float64).reshape(3),
        width=848,
        height=480,
        depth_scale=0.001,
        source="fallback_d435_world_camera_group_estimated",
    )


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {str(row["session_id"]): dict(row) for row in rows}


def load_camera_groups(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    return {str(group["group_name"]): group for group in payload.get("groups", [])}


def group_T(group: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = np.asarray(group.get("T_base_cam_estimated"), dtype=np.float64).reshape(4, 4)
    delta = np.asarray(group.get("delta_T_estimated"), dtype=np.float64).reshape(4, 4)
    return T[:3, :3], T[:3, 3], delta


def default_unknown_policy(session_id: str, groups: Mapping[str, Mapping[str, Any]]) -> EpisodePolicy:
    group = groups["needs_manual_review"]
    calib = groups["d435_calib_0"]
    T = np.asarray(calib["T_base_cam_estimated"], dtype=np.float64).reshape(4, 4)
    return EpisodePolicy(
        session_id=session_id,
        camera_group="needs_manual_review",
        calibration_quality="unknown",
        pointcloud_action="exclude",
        use_for_rgb_action="optional",
        use_for_metric_pointcloud=False,
        use_for_lc_projection="false",
        use_for_contact_memory_training="false",
        extrinsic_refined_by_manual_keypoints=False,
        requires_overlay_review=True,
        risk_note=str(group.get("risk_note", "Unverified camera pose group excluded from metric LC.")),
        T_base_cam_source="unknown",
        T_base_cam_estimated=T,
        delta_T_estimated=np.eye(4, dtype=np.float64),
        pitch_delta_deg=0.0,
        y_bias_raw_px=0.0,
    )


def policy_from_manifest_row(
    *,
    session_id: str,
    manifest_row: Mapping[str, Any] | None,
    groups: Mapping[str, Mapping[str, Any]],
) -> EpisodePolicy:
    if manifest_row is None:
        return default_unknown_policy(session_id, groups)
    camera_group = str(manifest_row.get("camera_group") or "needs_manual_review")
    group = groups.get(camera_group) or groups["needs_manual_review"]
    if group.get("T_base_cam_estimated") is None:
        calib = groups["d435_calib_0"]
        T = np.asarray(calib["T_base_cam_estimated"], dtype=np.float64).reshape(4, 4)
        delta = np.eye(4, dtype=np.float64)
    else:
        T = np.asarray(group["T_base_cam_estimated"], dtype=np.float64).reshape(4, 4)
        delta = np.asarray(group.get("delta_T_estimated"), dtype=np.float64).reshape(4, 4)
    return EpisodePolicy(
        session_id=session_id,
        camera_group=camera_group,
        calibration_quality=str(manifest_row.get("calibration_quality") or group.get("calibration_quality") or "unknown"),
        pointcloud_action=str(manifest_row.get("pointcloud_action") or group.get("pointcloud_action") or "exclude"),
        use_for_rgb_action=str(manifest_row.get("use_for_rgb_action", "optional")),
        use_for_metric_pointcloud=parse_bool(manifest_row.get("use_for_metric_pointcloud")),
        use_for_lc_projection=str(manifest_row.get("use_for_lc_projection", "false")).lower(),
        use_for_contact_memory_training=str(manifest_row.get("use_for_contact_memory_training", "false")).lower(),
        extrinsic_refined_by_manual_keypoints=parse_bool(manifest_row.get("extrinsic_refined_by_manual_keypoints")),
        requires_overlay_review=parse_bool(manifest_row.get("requires_overlay_review")),
        risk_note=str(manifest_row.get("risk_note") or group.get("risk_note") or ""),
        T_base_cam_source=str(group.get("T_base_cam_source", "unknown")),
        T_base_cam_estimated=T,
        delta_T_estimated=delta,
        pitch_delta_deg=float(group.get("pitch_delta_signed_deg", group.get("pitch_delta_deg", 0.0)) or 0.0),
        y_bias_raw_px=float(group.get("y_bias_raw_px", 0.0) or 0.0),
    )


def get_camera_info(
    *,
    repo_root: Path,
    source_file: str,
    image_size: int,
    policy: EpisodePolicy,
    cache: dict[tuple[str, str], CameraInfo],
) -> CameraInfo:
    cache_key = (source_file or "__missing_source__", policy.camera_group)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    R = policy.T_base_cam_estimated[:3, :3]
    t = policy.T_base_cam_estimated[:3, 3]
    session_dir = source_session_dir(repo_root, source_file)
    camera = camera_from_metadata(session_dir, R=R, t=t, image_size=image_size) if session_dir is not None else None
    if camera is None:
        camera = fallback_camera(R=R, t=t)
    cache[cache_key] = camera
    return camera


def clear_and_extend_float(feature: Any, values: np.ndarray) -> None:
    del feature.float_list.value[:]
    feature.float_list.value.extend(np.asarray(values, dtype=np.float32).reshape(-1).tolist())


def clear_and_extend_int64(feature: Any, values: np.ndarray) -> None:
    del feature.int64_list.value[:]
    feature.int64_list.value.extend(np.asarray(values, dtype=np.int64).reshape(-1).tolist())


def set_text_feature(features: Any, key: str, value: str) -> None:
    feature = features[key]
    del feature.bytes_list.value[:]
    feature.bytes_list.value.append(str(value).encode("utf-8"))


def set_bool_feature(features: Any, key: str, value: bool) -> None:
    clear_and_extend_int64(features[key], np.asarray([1 if value else 0], dtype=np.int64))


def set_float_scalar(features: Any, key: str, value: float) -> None:
    clear_and_extend_float(features[key], np.asarray([float(value)], dtype=np.float32))


def metric_lc_candidate(policy: EpisodePolicy) -> bool:
    return (
        policy.camera_group in METRIC_GROUPS
        and bool(policy.use_for_metric_pointcloud)
        and policy.use_for_lc_projection in {"true", "candidate", "1", "yes"}
        and policy.use_for_contact_memory_training in {"true", "candidate", "1", "yes"}
    )


def add_episode_metadata(features: Any, policy: EpisodePolicy) -> None:
    set_text_feature(features, "episode_metadata/camera_group", policy.camera_group)
    set_text_feature(features, "episode_metadata/calibration_quality", policy.calibration_quality)
    set_text_feature(features, "episode_metadata/pointcloud_action", policy.pointcloud_action)
    set_text_feature(features, "episode_metadata/T_base_cam_source", policy.T_base_cam_source)
    set_text_feature(features, "episode_metadata/use_for_lc_projection", policy.use_for_lc_projection)
    set_text_feature(features, "episode_metadata/use_for_contact_memory_training", policy.use_for_contact_memory_training)
    set_text_feature(features, "episode_metadata/use_for_rgb_action", policy.use_for_rgb_action)
    set_text_feature(features, "episode_metadata/risk_note", policy.risk_note)
    set_bool_feature(features, "episode_metadata/use_for_metric_pointcloud", policy.use_for_metric_pointcloud)
    set_bool_feature(features, "episode_metadata/metric_lc_candidate", metric_lc_candidate(policy))
    set_bool_feature(
        features,
        "episode_metadata/extrinsic_refined_by_manual_keypoints",
        policy.extrinsic_refined_by_manual_keypoints,
    )
    set_bool_feature(features, "episode_metadata/requires_overlay_review", policy.requires_overlay_review)
    set_float_scalar(features, "episode_metadata/pitch_delta_deg", policy.pitch_delta_deg)
    set_float_scalar(features, "episode_metadata/y_bias_raw_px", policy.y_bias_raw_px)
    clear_and_extend_float(features["episode_metadata/T_base_cam_estimated"], policy.T_base_cam_estimated.astype(np.float32))
    clear_and_extend_float(features["episode_metadata/delta_T_estimated"], policy.delta_T_estimated.astype(np.float32))


def regenerate_pointcloud_if_needed(
    *,
    features: Any,
    policy: EpisodePolicy,
    repo_root: Path,
    source_file: str,
    image_size: int,
    num_points: int,
    min_depth_m: float,
    max_depth_m: float,
    camera_cache: dict[tuple[str, str], CameraInfo],
) -> dict[str, Any]:
    stats = {
        "regenerated": False,
        "zero_pointcloud_frames": None,
        "valid_depth_min": None,
        "valid_depth_max": None,
        "camera_source": None,
    }
    if policy.pointcloud_action != "regenerate_from_depth":
        return stats
    if PRIMARY_DEPTH_KEY not in features or STATE_KEY not in features:
        raise ValueError(f"Missing required fields for pointcloud regeneration: {PRIMARY_DEPTH_KEY}, {STATE_KEY}")

    n_steps = len(features[STATE_KEY].float_list.value) // 7
    depth_values = features[PRIMARY_DEPTH_KEY].int64_list.value
    expected_depth_values = n_steps * image_size * image_size
    if len(depth_values) != expected_depth_values:
        raise ValueError(
            f"primary_depth length mismatch: got {len(depth_values)}, expected {expected_depth_values}"
        )
    depth = np.asarray(depth_values, dtype=np.uint16).reshape(n_steps, image_size, image_size)
    camera = get_camera_info(
        repo_root=repo_root,
        source_file=source_file,
        image_size=image_size,
        policy=policy,
        cache=camera_cache,
    )
    point_cloud = np.empty((n_steps, num_points, 3), dtype=np.float32)
    valid_counts = np.empty((n_steps,), dtype=np.int64)
    for idx in range(n_steps):
        pc, valid_count = depth_to_point_cloud(
            depth[idx],
            camera=camera,
            image_size=image_size,
            num_points=num_points,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        point_cloud[idx] = pc
        valid_counts[idx] = valid_count

    next_point_cloud = np.concatenate([point_cloud[1:], point_cloud[-1:]], axis=0)
    clear_and_extend_float(features[POINT_CLOUD_KEY], point_cloud)
    clear_and_extend_float(features[NEXT_POINT_CLOUD_KEY], next_point_cloud)
    stats.update(
        {
            "regenerated": True,
            "zero_pointcloud_frames": int(np.count_nonzero(valid_counts == 0)),
            "valid_depth_min": int(valid_counts.min()) if len(valid_counts) else 0,
            "valid_depth_max": int(valid_counts.max()) if len(valid_counts) else 0,
            "camera_source": camera.source,
        }
    )
    return stats


def rewrite_example(
    *,
    raw: bytes,
    tf: Any,
    manifest_by_session: Mapping[str, Mapping[str, str]],
    camera_groups: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    image_size: int,
    num_points: int,
    min_depth_m: float,
    max_depth_m: float,
    camera_cache: dict[tuple[str, str], CameraInfo],
) -> tuple[bytes, dict[str, Any]]:
    example = tf.train.Example.FromString(raw)
    features = example.features.feature
    source_file = decode_one_text(features.get(SOURCE_FILE_KEY, tf.train.Feature()))
    session_id = session_id_from_source_file(source_file)
    policy = policy_from_manifest_row(
        session_id=session_id,
        manifest_row=manifest_by_session.get(session_id),
        groups=camera_groups,
    )
    pc_stats = regenerate_pointcloud_if_needed(
        features=features,
        policy=policy,
        repo_root=repo_root,
        source_file=source_file,
        image_size=image_size,
        num_points=num_points,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        camera_cache=camera_cache,
    )
    add_episode_metadata(features, policy)
    n_steps = len(features[STATE_KEY].float_list.value) // 7 if STATE_KEY in features else 0
    return example.SerializeToString(), {
        "session_id": session_id,
        "source_file": source_file,
        "camera_group": policy.camera_group,
        "calibration_quality": policy.calibration_quality,
        "pointcloud_action": policy.pointcloud_action,
        "metric_lc_candidate": metric_lc_candidate(policy),
        "use_for_metric_pointcloud": policy.use_for_metric_pointcloud,
        "steps": int(n_steps),
        **pc_stats,
    }


def text_feature() -> dict[str, Any]:
    return {"pythonClassName": "tensorflow_datasets.core.features.text_feature.Text", "text": {}}


def scalar_feature(dtype: str) -> dict[str, Any]:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
        "tensor": {"dtype": dtype, "encoding": "none", "shape": {}},
    }


def tensor_feature(dtype: str, dims: list[str]) -> dict[str, Any]:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
        "tensor": {"dtype": dtype, "encoding": "none", "shape": {"dimensions": dims}},
    }


def update_features_json(work_dir: Path) -> None:
    path = work_dir / "features.json"
    features = read_json(path)
    metadata_features = features["featuresDict"]["features"]["episode_metadata"]["featuresDict"]["features"]
    metadata_features.update(
        {
            "camera_group": text_feature(),
            "calibration_quality": text_feature(),
            "pointcloud_action": text_feature(),
            "T_base_cam_source": text_feature(),
            "use_for_lc_projection": text_feature(),
            "use_for_contact_memory_training": text_feature(),
            "use_for_rgb_action": text_feature(),
            "risk_note": text_feature(),
            "use_for_metric_pointcloud": scalar_feature("bool"),
            "metric_lc_candidate": scalar_feature("bool"),
            "extrinsic_refined_by_manual_keypoints": scalar_feature("bool"),
            "requires_overlay_review": scalar_feature("bool"),
            "pitch_delta_deg": scalar_feature("float32"),
            "y_bias_raw_px": scalar_feature("float32"),
            "T_base_cam_estimated": tensor_feature("float32", ["4", "4"]),
            "delta_T_estimated": tensor_feature("float32", ["4", "4"]),
        }
    )
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


def render_report(summary: Mapping[str, Any]) -> str:
    return f"""# RLDS 3.4.0 Candidate Build {TODAY}

## Scope

Built a `3.4.0_candidate` RLDS version from `3.3.0_candidate`.
This is an estimated-calibration candidate, not a strict calibration release.

```text
EXTRINSIC_REFINED_BY_MANUAL_KEYPOINTS = false
CALIBRATION_QUALITY = estimated
RLDS_3_4_0_STATUS = candidate
```

## Output

- source: `{summary.get('source_version_dir')}`
- output: `{summary.get('output_version_dir')}`
- report json: `{summary.get('report_json')}`

## Counts

```json
{json.dumps(jsonable(summary.get('counts')), indent=2, ensure_ascii=False)}
```

## Camera Groups

```json
{json.dumps(jsonable(summary.get('camera_group_counts')), indent=2, ensure_ascii=False)}
```

## Pointcloud Actions

```json
{json.dumps(jsonable(summary.get('pointcloud_action_counts')), indent=2, ensure_ascii=False)}
```

## Policy

- `d435_calib_0`: keep original pointcloud, verified original calibration.
- `d435_shift_1` / `d435_shift_2`: regenerate pointcloud from primary_depth using estimated group `T_base_cam`.
- `dji_primary_pseudo_depth` / `needs_manual_review`: retained as RGB-action-compatible episodes, excluded from metric LC.

No 2D pixel shift is used to repair pointclouds.

## Readiness

```json
{json.dumps(jsonable(summary.get('readiness')), indent=2, ensure_ascii=False)}
```

## Safety

```json
{json.dumps(jsonable(summary.get('safety')), indent=2, ensure_ascii=False)}
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-version-dir", type=Path, default=DEFAULT_SOURCE_VERSION_DIR)
    parser.add_argument("--output-version-dir", type=Path, default=DEFAULT_OUTPUT_VERSION_DIR)
    parser.add_argument("--output-version", default="3.4.0")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--camera-group-config", type=Path, default=DEFAULT_CAMERA_GROUP_CONFIG)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--summary-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import tensorflow as tf

    if not args.source_version_dir.is_dir():
        raise SystemExit(f"Source version directory not found: {args.source_version_dir}")
    if not args.candidate_manifest.is_file():
        raise SystemExit(f"Candidate manifest not found: {args.candidate_manifest}")
    if not args.camera_group_config.is_file():
        raise SystemExit(f"Camera group config not found: {args.camera_group_config}")

    shards = sorted(args.source_version_dir.glob(f"{DATASET_NAME}-train.tfrecord-*"))
    if args.max_shards is not None:
        shards = shards[: int(args.max_shards)]
    if not shards:
        raise SystemExit(f"No TFRecord shards found in {args.source_version_dir}")

    manifest = load_manifest(args.candidate_manifest)
    camera_groups = load_camera_groups(args.camera_group_config)
    if "d435_calib_0" not in camera_groups or "needs_manual_review" not in camera_groups:
        raise SystemExit("camera group config must include d435_calib_0 and needs_manual_review")

    work_dir, trash_dirs = prepare_output_dir(args.output_version_dir, args.overwrite)
    copy_metadata(args.source_version_dir, work_dir)
    shutil.copy2(args.candidate_manifest, work_dir / "rlds_3_4_0_candidate_manifest.csv")
    shutil.copy2(args.camera_group_config, work_dir / "transvtla_world_camera_groups.json")

    camera_cache: dict[tuple[str, str], CameraInfo] = {}
    shard_lengths: list[int] = []
    total_output_bytes = 0
    counts = Counter()
    group_counts = Counter()
    action_counts = Counter()
    quality_counts = Counter()
    zero_frames = 0
    valid_depth_mins: list[float] = []
    valid_depth_maxs: list[float] = []
    started = time.time()

    print(f"source={args.source_version_dir}")
    print(f"work_dir={work_dir}")
    print(f"output={args.output_version_dir}")
    print(f"shards={len(shards)} output_version={args.output_version}")
    print(f"manifest_sessions={len(manifest)}")

    try:
        for shard_idx, src in enumerate(shards):
            dst = work_dir / output_shard_name(shard_idx, len(shards))
            shard_examples = 0
            shard_steps = 0
            with tf.io.TFRecordWriter(str(dst)) as writer:
                for raw in tf.compat.v1.io.tf_record_iterator(str(src)):
                    rewritten, ex_stats = rewrite_example(
                        raw=raw,
                        tf=tf,
                        manifest_by_session=manifest,
                        camera_groups=camera_groups,
                        repo_root=args.repo_root,
                        image_size=args.image_size,
                        num_points=args.num_points,
                        min_depth_m=args.min_depth_m,
                        max_depth_m=args.max_depth_m,
                        camera_cache=camera_cache,
                    )
                    writer.write(rewritten)
                    shard_examples += 1
                    shard_steps += int(ex_stats["steps"])
                    counts["examples"] += 1
                    counts["steps"] += int(ex_stats["steps"])
                    if bool(ex_stats["metric_lc_candidate"]):
                        counts["metric_lc_candidate_examples"] += 1
                        counts["metric_lc_candidate_steps"] += int(ex_stats["steps"])
                    if bool(ex_stats["regenerated"]):
                        counts["pointcloud_regenerated_examples"] += 1
                        counts["pointcloud_regenerated_steps"] += int(ex_stats["steps"])
                        zero_frames += int(ex_stats["zero_pointcloud_frames"] or 0)
                        valid_depth_mins.append(float(ex_stats["valid_depth_min"] or 0))
                        valid_depth_maxs.append(float(ex_stats["valid_depth_max"] or 0))
                    group_counts[str(ex_stats["camera_group"])] += 1
                    action_counts[str(ex_stats["pointcloud_action"])] += 1
                    quality_counts[str(ex_stats["calibration_quality"])] += 1
            shard_lengths.append(shard_examples)
            total_output_bytes += dst.stat().st_size
            if (
                shard_idx == 0
                or (shard_idx + 1) % int(args.summary_every) == 0
                or shard_idx == len(shards) - 1
            ):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[{shard_idx + 1:04d}/{len(shards):04d}] {src.name} "
                    f"examples={shard_examples} steps={shard_steps} "
                    f"total_examples={counts['examples']} metric={counts['metric_lc_candidate_examples']} "
                    f"regenerated={counts['pointcloud_regenerated_examples']} elapsed_min={elapsed / 60.0:.1f}",
                    flush=True,
                )

        update_features_json(work_dir)
        update_dataset_info(work_dir, args.output_version, shard_lengths, total_output_bytes)
        readiness = {
            "RLDS_3_4_0_CANDIDATE_BUILT": True,
            "DJI_PSEUDO_DEPTH_EXCLUDED_FROM_METRIC_LC": group_counts.get("dji_primary_pseudo_depth", 0) == 0
            or action_counts.get("exclude", 0) >= group_counts.get("dji_primary_pseudo_depth", 0),
            "NEEDS_MANUAL_REVIEW_EXCLUDED_FROM_METRIC_LC": True,
            "EXTRINSIC_REFINED_BY_MANUAL_KEYPOINTS": False,
            "CALIBRATION_QUALITY": "estimated",
            "RLDS_3_4_0_STATUS": "candidate",
            "READY_FOR_STRICT_CALIBRATION_RELEASE": False,
            "READY_FOR_REAL_ROBOT_ROLLOUT": False,
        }
        safety = {
            "NO_ROBOT_COMMAND_SENT": True,
            "NO_GRIPPER_COMMAND_SENT": True,
            "NO_REAL_ROBOT_API_CALLED": True,
            "ACTION_CHUNK_NOT_EXECUTED": True,
            "REAL7B_NOT_RUN": True,
            "TRAINING_NOT_RUN": True,
            "RLDS_3_3_0_NOT_OVERWRITTEN": args.source_version_dir.exists()
            and args.source_version_dir.name == "3.3.0_candidate",
        }
        summary = {
            "schema": "rlds_3_4_0_candidate_build/v1",
            "source_version_dir": str(args.source_version_dir),
            "output_version_dir": str(args.output_version_dir),
            "output_version": args.output_version,
            "work_dir": str(work_dir),
            "candidate_manifest": str(args.candidate_manifest),
            "camera_group_config": str(args.camera_group_config),
            "report_json": str(args.report_json),
            "report_md": str(args.report_md),
            "elapsed_sec": float(time.time() - started),
            "num_shards": len(shards),
            "shard_lengths": shard_lengths,
            "num_bytes": int(total_output_bytes),
            "counts": {
                **dict(counts),
                "excluded_metric_examples": int(counts["examples"] - counts["metric_lc_candidate_examples"]),
                "pointcloud_regenerated_zero_frames": int(zero_frames),
                "regenerated_valid_depth_min_stats": stats(valid_depth_mins),
                "regenerated_valid_depth_max_stats": stats(valid_depth_maxs),
            },
            "camera_group_counts": dict(group_counts),
            "pointcloud_action_counts": dict(action_counts),
            "calibration_quality_counts": dict(quality_counts),
            "metric_lc_candidate_groups": sorted(METRIC_GROUPS),
            "excluded_groups": sorted(EXCLUDED_GROUPS),
            "readiness": readiness,
            "safety": safety,
        }
        write_json(work_dir / "rlds_3_4_0_candidate_build_summary.json", summary)
        write_json(args.report_json, summary)
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(render_report(summary), encoding="utf-8")
        publish_output_dir(work_dir, args.output_version_dir, trash_dirs)
        print(f"published={args.output_version_dir}")
        print(f"report={args.report_json}")
        return 0
    except Exception:
        print(f"build failed; temporary work dir retained for inspection: {work_dir}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
