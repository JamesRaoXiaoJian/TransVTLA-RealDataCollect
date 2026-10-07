#!/usr/bin/env python3
"""Add main-view point cloud tensors to an existing TransVTLA RLDS dataset.

The source dataset is left untouched. This script rewrites TFRecord shards into
a new TFDS version directory and appends:

  steps/observation/point_cloud      float32 [T, N, 3]
  steps/observation/next_point_cloud float32 [T, N, 3]

Each per-step point cloud is generated from observation.primary_depth. The
resulting step tensor shape decoded by TFDS is [N, 3], matching the MLA
PointTokenizer input contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


DATASET_NAME = "transvtla_annotated_rlds"
PRIMARY_DEPTH_KEY = "steps/observation/primary_depth"
STATE_KEY = "steps/observation/state"
SOURCE_FILE_KEY = "episode_metadata/source_file"
POINT_CLOUD_KEY = "steps/observation/point_cloud"
NEXT_POINT_CLOUD_KEY = "steps/observation/next_point_cloud"


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


D435_WORLD_R = np.asarray(
    [
        [-0.96943095, 0.09244450, -0.22728317],
        [0.20452160, -0.20729519, -0.95666066],
        [-0.13555273, -0.97390077, 0.18205148],
    ],
    dtype=np.float64,
)
D435_WORLD_T = np.asarray([0.33983827, 0.83416543, 0.11930434], dtype=np.float64)
D435_WORLD_K = np.asarray(
    [
        [602.716614, 0.0, 432.202942],
        [0.0, 603.465698, 249.251740],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

LEGACY_DJI_WORLD_R = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
LEGACY_DJI_WORLD_T = np.asarray([0.133, 0.58, 0.32], dtype=np.float64)
LEGACY_DJI_WORLD_K = np.asarray(
    [
        [553.635449, 0.0, 632.236057],
        [0.0, 553.887213, 340.647155],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def decode_one_text(feature: tf.train.Feature) -> str:
    values = feature.bytes_list.value
    if not values:
        return ""
    return values[0].decode("utf-8", "ignore")


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
    if work_dir.exists():
        raise SystemExit(f"Temporary output directory unexpectedly exists: {work_dir}")
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def scaled_intrinsics(K: np.ndarray, width: int, height: int, image_size: int) -> np.ndarray:
    K_scaled = K.astype(np.float64).copy()
    K_scaled[0, 0] *= image_size / float(width)
    K_scaled[0, 2] *= image_size / float(width)
    K_scaled[1, 1] *= image_size / float(height)
    K_scaled[1, 2] *= image_size / float(height)
    return K_scaled


def camera_to_world(points_cam: np.ndarray, R_base_cam: np.ndarray, t_base_cam: np.ndarray) -> np.ndarray:
    # Matches models/mla/fuser projection convention:
    # x_cam = x_world @ (R_base_cam.T).T + (-R_base_cam.T @ t_base_cam)
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


def camera_from_metadata(session_dir: Path, image_size: int) -> CameraInfo | None:
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
        R=D435_WORLD_R,
        t=D435_WORLD_T,
        width=width,
        height=height,
        depth_scale=depth_scale,
        source="session_metadata_d435_world",
    )


def fallback_camera(source_file: str) -> CameraInfo:
    if "phase2_realdata_sessions" in source_file or "/dji" in source_file:
        return CameraInfo(
            name="legacy_dji_world_camera",
            K=LEGACY_DJI_WORLD_K,
            R=LEGACY_DJI_WORLD_R,
            t=LEGACY_DJI_WORLD_T,
            width=1280,
            height=720,
            depth_scale=0.001,
            source="fallback_legacy_dji_world",
        )
    return CameraInfo(
        name="world_camera",
        K=D435_WORLD_K,
        R=D435_WORLD_R,
        t=D435_WORLD_T,
        width=848,
        height=480,
        depth_scale=0.001,
        source="fallback_d435_world",
    )


def get_camera_info(
    repo_root: Path,
    source_file: str,
    image_size: int,
    cache: dict[str, CameraInfo],
) -> CameraInfo:
    cache_key = source_file or "__missing_source__"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    session_dir = source_session_dir(repo_root, source_file)
    camera = camera_from_metadata(session_dir, image_size) if session_dir is not None else None
    if camera is None:
        camera = fallback_camera(source_file)
    cache[cache_key] = camera
    return camera


def add_pointcloud_to_example(
    raw: bytes,
    repo_root: Path,
    image_size: int,
    num_points: int,
    min_depth_m: float,
    max_depth_m: float,
    camera_cache: dict[str, CameraInfo],
) -> tuple[bytes, dict[str, Any]]:
    example = tf.train.Example.FromString(raw)
    features = example.features.feature
    if PRIMARY_DEPTH_KEY not in features or STATE_KEY not in features:
        raise ValueError(f"Missing required keys: {PRIMARY_DEPTH_KEY}, {STATE_KEY}")

    n_steps = len(features[STATE_KEY].float_list.value) // 7
    if n_steps <= 0:
        raise ValueError("Episode has no state steps")

    depth_values = features[PRIMARY_DEPTH_KEY].int64_list.value
    expected_depth_values = n_steps * image_size * image_size
    if len(depth_values) != expected_depth_values:
        raise ValueError(
            f"primary_depth length mismatch: got {len(depth_values)}, "
            f"expected {expected_depth_values}"
        )
    depth = np.asarray(depth_values, dtype=np.uint16).reshape(n_steps, image_size, image_size)
    source_file = decode_one_text(features.get(SOURCE_FILE_KEY, tf.train.Feature()))
    camera = get_camera_info(repo_root, source_file, image_size, camera_cache)

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
    del features[POINT_CLOUD_KEY].float_list.value[:]
    features[POINT_CLOUD_KEY].float_list.value.extend(point_cloud.reshape(-1).tolist())
    del features[NEXT_POINT_CLOUD_KEY].float_list.value[:]
    features[NEXT_POINT_CLOUD_KEY].float_list.value.extend(next_point_cloud.reshape(-1).tolist())

    stats = {
        "steps": int(n_steps),
        "camera_source": camera.source,
        "source_file": source_file,
        "zero_pointcloud_frames": int(np.count_nonzero(valid_counts == 0)),
        "valid_depth_min": int(valid_counts.min()) if len(valid_counts) else 0,
        "valid_depth_max": int(valid_counts.max()) if len(valid_counts) else 0,
    }
    return example.SerializeToString(), stats


def output_shard_name(index: int, total: int) -> str:
    return f"{DATASET_NAME}-train.tfrecord-{index:05d}-of-{total:05d}"


def copy_metadata(source_version_dir: Path, work_dir: Path) -> None:
    for src in source_version_dir.iterdir():
        if not src.is_file():
            continue
        name = src.name
        if name.startswith(f"{DATASET_NAME}-train.tfrecord"):
            continue
        if name.startswith("dataset_statistics_"):
            continue
        shutil.copy2(src, work_dir / name)


def point_cloud_feature(num_points: int) -> dict[str, Any]:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
        "tensor": {
            "dtype": "float32",
            "encoding": "none",
            "shape": {"dimensions": [str(num_points), "3"]},
        },
    }


def update_features_json(work_dir: Path, num_points: int) -> None:
    path = work_dir / "features.json"
    features = read_json(path)
    obs_features = (
        features["featuresDict"]["features"]["steps"]["sequence"]["feature"]["featuresDict"]
        ["features"]["observation"]["featuresDict"]["features"]
    )
    obs_features["point_cloud"] = point_cloud_feature(num_points)
    obs_features["next_point_cloud"] = point_cloud_feature(num_points)
    write_json(path, features)


def update_dataset_info(
    work_dir: Path,
    version: str,
    shard_lengths: list[int],
    num_bytes: int,
) -> None:
    path = work_dir / "dataset_info.json"
    info = read_json(path)
    info["version"] = version
    if info.get("splits"):
        info["splits"][0]["numBytes"] = str(num_bytes)
        info["splits"][0]["shardLengths"] = [str(v) for v in shard_lengths]
    write_json(path, info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-version-dir",
        type=Path,
        default=Path(
            "/media/files/TransVTLA-RealDataCollect/dataset/"
            "rlds_real1000_strict/transvtla_annotated_rlds/3.0.0"
        ),
    )
    parser.add_argument("--output-version", default="3.1.0")
    parser.add_argument("--output-version-dir", type=Path, default=None)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/media/files/TransVTLA-RealDataCollect"),
        help="Root used to resolve episode_metadata/source_file relative paths.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--max-shards", type=int, default=None, help="Debug option.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_version_dir = args.source_version_dir
    if not source_version_dir.is_dir():
        raise SystemExit(f"Source version directory not found: {source_version_dir}")

    output_version_dir = args.output_version_dir
    if output_version_dir is None:
        output_version_dir = source_version_dir.parent / args.output_version

    shards = sorted(source_version_dir.glob(f"{DATASET_NAME}-train.tfrecord-*"))
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    if not shards:
        raise SystemExit(f"No TFRecord shards found in {source_version_dir}")

    work_dir, trash_dirs = prepare_output_dir(output_version_dir, args.overwrite)
    copy_metadata(source_version_dir, work_dir)

    camera_cache: dict[str, CameraInfo] = {}
    shard_lengths: list[int] = []
    total_examples = 0
    total_steps = 0
    total_zero_frames = 0
    total_output_bytes = 0
    camera_source_counts: dict[str, int] = {}
    started = time.time()

    print(f"source={source_version_dir}")
    print(f"work_dir={work_dir}")
    print(f"output={output_version_dir}")
    print(
        f"num_points={args.num_points} depth_range=({args.min_depth_m}, {args.max_depth_m}) "
        f"shards={len(shards)}"
    )

    for shard_idx, src in enumerate(shards):
        dst = work_dir / output_shard_name(shard_idx, len(shards))
        shard_examples = 0
        shard_steps = 0
        with tf.io.TFRecordWriter(str(dst)) as writer:
            for raw in tf.compat.v1.io.tf_record_iterator(str(src)):
                rewritten, stats = add_pointcloud_to_example(
                    raw,
                    repo_root=args.repo_root,
                    image_size=args.image_size,
                    num_points=args.num_points,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_depth_m,
                    camera_cache=camera_cache,
                )
                writer.write(rewritten)
                shard_examples += 1
                shard_steps += int(stats["steps"])
                total_examples += 1
                total_steps += int(stats["steps"])
                total_zero_frames += int(stats["zero_pointcloud_frames"])
                camera_source = str(stats["camera_source"])
                camera_source_counts[camera_source] = camera_source_counts.get(camera_source, 0) + 1

        shard_lengths.append(shard_examples)
        shard_bytes = dst.stat().st_size
        total_output_bytes += shard_bytes
        if shard_idx == 0 or (shard_idx + 1) % 25 == 0 or shard_idx == len(shards) - 1:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[{shard_idx + 1:04d}/{len(shards):04d}] {src.name} "
                f"examples={shard_examples} steps={shard_steps} "
                f"total_examples={total_examples} total_steps={total_steps} "
                f"elapsed_min={elapsed / 60.0:.1f}",
                flush=True,
            )

    update_features_json(work_dir, args.num_points)
    update_dataset_info(work_dir, args.output_version, shard_lengths, total_output_bytes)
    summary = {
        "source_version_dir": str(source_version_dir),
        "output_version_dir": str(output_version_dir),
        "output_version": args.output_version,
        "num_points": args.num_points,
        "point_cloud_shape_per_step": [args.num_points, 3],
        "dtype": "float32",
        "view": "primary/world_camera",
        "depth_source": PRIMARY_DEPTH_KEY,
        "depth_range_m": [args.min_depth_m, args.max_depth_m],
        "coordinate_frame": "world/base frame using world_camera R/t",
        "camera_params_priority": [
            "session camera_metadata.json world_camera color intrinsics + D435 world_camera R/t",
            "fallback legacy DJI world params for sessions without camera_metadata.json",
            "fallback D435 world params for missing metadata outside legacy sessions",
        ],
        "num_shards": len(shards),
        "num_examples": total_examples,
        "num_steps": total_steps,
        "num_zero_pointcloud_frames": total_zero_frames,
        "camera_source_counts": camera_source_counts,
        "output_num_bytes": total_output_bytes,
        "notes": [
            "Original dataset version directory is not modified.",
            "next_point_cloud is point_cloud shifted by one step; terminal step repeats the final cloud.",
            "dataset_statistics_*.json files are intentionally not copied.",
        ],
    }
    write_json(work_dir / "pointcloud_generation_summary.json", summary)
    publish_output_dir(work_dir, output_version_dir, trash_dirs)
    print(f"done: {output_version_dir}")
    print(f"summary: {output_version_dir / 'pointcloud_generation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
