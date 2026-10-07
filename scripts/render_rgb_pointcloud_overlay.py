#!/usr/bin/env python3
"""Render a real RGB + point-cloud overlay from a TransVTLA session frame.

The saved RealSense depth PNG is already aligned to color pixels. This script
still follows the complete TransVTLA geometry path so the visualization checks
the same coordinate conventions used by the model:

  aligned depth pixel -> camera xyz -> base/world xyz -> projected RGB pixel

The camera-to-world and world-to-camera equations match
scripts/add_pointcloud_to_rlds.py and fuser/contrastive.py respectively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# fuser/camera.py: world_camera T_base_cam (2026-06-23 calibration).
WORLD_CAMERA_R_BASE_CAM = np.asarray(
    [
        [-0.96943095, 0.09244450, -0.22728317],
        [0.20452160, -0.20729519, -0.95666066],
        [-0.13555273, -0.97390077, 0.18205148],
    ],
    dtype=np.float64,
)
WORLD_CAMERA_T_BASE_CAM = np.asarray(
    [0.33983827, 0.83416543, 0.11930434],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--frame", default="0006")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pointcloud-output", type=Path, default=None)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.82)
    return parser.parse_args()


def load_camera_metadata(session_dir: Path) -> tuple[np.ndarray, float, int, int]:
    metadata_path = session_dir / "camera_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    world = metadata["cameras"]["world_camera"]
    intrinsics = world["color"]["intrinsics"]
    K = np.asarray(intrinsics["K"], dtype=np.float64)
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    depth_meta = world["depth"]
    depth_scale = float(
        depth_meta.get(
            "saved_png_unit_m",
            depth_meta.get("sensor_depth_scale_m_per_unit", 0.001),
        )
    )
    if depth_meta.get("aligned_to") != "color":
        raise ValueError("Depth is not declared aligned to the color stream")
    if K.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 color K, got {K.shape}")
    return K, depth_scale, width, height


def camera_to_world(points_cam: np.ndarray) -> np.ndarray:
    """Match scripts/add_pointcloud_to_rlds.py::camera_to_world."""
    R_world_to_cam = WORLD_CAMERA_R_BASE_CAM.T
    t_world_to_cam = -R_world_to_cam @ WORLD_CAMERA_T_BASE_CAM
    return (points_cam - t_world_to_cam) @ R_world_to_cam


def project_world_to_rgb(
    points_world: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match fuser/contrastive.py's world-camera projector before patch binning."""
    R_world_to_cam = WORLD_CAMERA_R_BASE_CAM.T
    t_world_to_cam = -R_world_to_cam @ WORLD_CAMERA_T_BASE_CAM
    points_cam = points_world @ R_world_to_cam.T + t_world_to_cam
    uvw = points_cam @ K.T
    pixels_uv = uvw[:, :2] / (uvw[:, 2:3] + 1e-12)
    return pixels_uv, points_cam[:, 2]


def depth_to_world(
    depth_u16: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_m = depth_u16.astype(np.float64) * depth_scale
    valid = (
        np.isfinite(depth_m)
        & (depth_m > min_depth_m)
        & (depth_m < max_depth_m)
    )
    ys, xs = np.nonzero(valid)
    z = depth_m[ys, xs]
    x = (xs.astype(np.float64) - K[0, 2]) / K[0, 0] * z
    y = (ys.astype(np.float64) - K[1, 2]) / K[1, 1] * z
    points_cam = np.stack([x, y, z], axis=1)
    points_world = camera_to_world(points_cam)
    source_pixels_uv = np.stack([xs, ys], axis=1).astype(np.float64)
    return points_world, source_pixels_uv, z


def depth_colors(depth_m: np.ndarray, low: float, high: float) -> np.ndarray:
    """Near=red/yellow, far=cyan/blue piecewise-linear depth colormap."""
    anchors_t = np.asarray([0.0, 0.22, 0.45, 0.68, 1.0], dtype=np.float64)
    anchors_rgb = np.asarray(
        [
            [255, 38, 28],
            [255, 215, 32],
            [46, 210, 80],
            [25, 210, 235],
            [44, 72, 255],
        ],
        dtype=np.float64,
    )
    normalized = np.clip((depth_m - low) / max(high - low, 1e-12), 0.0, 1.0)
    colors = np.empty((depth_m.size, 3), dtype=np.float64)
    for channel in range(3):
        colors[:, channel] = np.interp(
            normalized,
            anchors_t,
            anchors_rgb[:, channel],
        )
    return np.rint(colors).astype(np.uint8)


def render_overlay(
    rgb: np.ndarray,
    pixels_uv: np.ndarray,
    colors: np.ndarray,
    stride: int,
    radius: int,
    alpha: float,
) -> tuple[np.ndarray, int]:
    height, width = rgb.shape[:2]
    xy_all = np.rint(pixels_uv).astype(np.int64)
    # Sample in the projected image grid rather than every Nth flattened depth
    # element. The latter creates scanline bands whenever the depth mask has
    # holes; grid sampling preserves a spatially uniform sensor-like overlay.
    keep = (xy_all[:, 0] % stride == 0) & (xy_all[:, 1] % stride == 0)
    xy = xy_all[keep]
    point_colors = colors[keep]
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    xy = xy[inside]
    point_colors = point_colors[inside]

    overlay = rgb.astype(np.float64).copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = xy[:, 0] + dx
            y = xy[:, 1] + dy
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x = x[valid]
            y = y[valid]
            c = point_colors[valid].astype(np.float64)
            overlay[y, x] = (1.0 - alpha) * overlay[y, x] + alpha * c
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8), int(xy.shape[0])


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.point_radius < 0:
        raise SystemExit("--point-radius must be >= 0")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")

    session_dir = args.session_dir.resolve()
    rgb_path = session_dir / "world_camera" / "rgb" / f"{args.frame}.jpg"
    depth_path = session_dir / "world_camera" / "depth" / f"{args.frame}.png"
    output_path = args.output or (
        session_dir / "world_camera" / "rgb" / f"{args.frame}_rgb_pointcloud_real.png"
    )
    pointcloud_path = args.pointcloud_output or (
        session_dir / "world_camera" / "depth" / f"{args.frame}_pointcloud_world.npz"
    )

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth_u16 = np.asarray(Image.open(depth_path))
    K, depth_scale, width, height = load_camera_metadata(session_dir)
    if rgb.shape[:2] != (height, width):
        raise ValueError(
            f"RGB shape {rgb.shape[:2]} does not match metadata {(height, width)}"
        )
    if depth_u16.shape != (height, width):
        raise ValueError(
            f"Depth shape {depth_u16.shape} does not match metadata {(height, width)}"
        )
    if depth_u16.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth PNG, got {depth_u16.dtype}")

    points_world, source_pixels_uv, depth_m = depth_to_world(
        depth_u16,
        K,
        depth_scale,
        args.min_depth_m,
        args.max_depth_m,
    )
    projected_pixels_uv, projected_z = project_world_to_rgb(points_world, K)
    if points_world.size == 0:
        raise ValueError("No valid depth pixels in the requested range")

    roundtrip_error = np.linalg.norm(projected_pixels_uv - source_pixels_uv, axis=1)
    color_low, color_high = np.percentile(depth_m, [1.0, 99.0])
    colors = depth_colors(projected_z, float(color_low), float(color_high))
    overlay, rendered_count = render_overlay(
        rgb,
        projected_pixels_uv,
        colors,
        args.stride,
        args.point_radius,
        args.alpha,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(output_path)
    pointcloud_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pointcloud_path,
        points_world=points_world.astype(np.float32),
        colors_rgb=rgb[
            source_pixels_uv[:, 1].astype(np.int64),
            source_pixels_uv[:, 0].astype(np.int64),
        ],
        source_pixels_uv=source_pixels_uv.astype(np.float32),
        projected_pixels_uv=projected_pixels_uv.astype(np.float32),
        depth_m=depth_m.astype(np.float32),
        K=K.astype(np.float32),
        R_base_cam=WORLD_CAMERA_R_BASE_CAM.astype(np.float32),
        t_base_cam=WORLD_CAMERA_T_BASE_CAM.astype(np.float32),
    )

    print(f"rgb={rgb_path}")
    print(f"depth={depth_path}")
    print(f"overlay={output_path}")
    print(f"pointcloud={pointcloud_path}")
    print(f"valid_points={points_world.shape[0]}")
    print(f"rendered_points={rendered_count}")
    print(f"depth_range_m=[{depth_m.min():.6f}, {depth_m.max():.6f}]")
    print(f"color_range_p01_p99_m=[{color_low:.6f}, {color_high:.6f}]")
    print(f"roundtrip_mean_px={roundtrip_error.mean():.9f}")
    print(f"roundtrip_max_px={roundtrip_error.max():.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
