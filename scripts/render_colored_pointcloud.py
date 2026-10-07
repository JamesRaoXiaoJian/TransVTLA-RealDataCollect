#!/usr/bin/env python3
"""Render a TransVTLA RGB-D frame as an isolated colored 3D point cloud.

The output has a white background and no RGB image plane. Every rendered point
comes from the real depth map and is colored by its aligned RGB pixel.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Avoid attempts to write matplotlib's cache outside the workspace.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/transvtla-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-render-points", type=int, default=80000)
    parser.add_argument(
        "--max-render-depth-m",
        type=float,
        default=2.0,
        help="Visualization crop used to suppress the far wall; <=0 keeps all points.",
    )
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-62.0)
    parser.add_argument("--point-size", type=float, default=1.2)
    parser.add_argument("--zoom", type=float, default=2.1)
    parser.add_argument(
        "--view-mode",
        choices=("perspective", "front", "near-front"),
        default="perspective",
    )
    parser.add_argument("--front-stride", type=int, default=4)
    parser.add_argument("--front-point-radius", type=int, default=1)
    parser.add_argument("--yaw", type=float, default=12.0)
    parser.add_argument("--pitch", type=float, default=-4.0)
    parser.add_argument(
        "--point-brightness",
        type=float,
        default=1.0,
        help="Multiply RGB point colors to improve contrast on white backgrounds.",
    )
    parser.add_argument(
        "--color-style",
        choices=("rgb", "reference"),
        default="rgb",
        help="Use aligned RGB or a warm-brown/teal reference visualization palette.",
    )
    parser.add_argument(
        "--auto-frame",
        action="store_true",
        help="Crop white margins and refit the cloud to the output canvas.",
    )
    return parser.parse_args()


def equal_limits(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    low = np.percentile(points, 0.5, axis=0)
    high = np.percentile(points, 99.5, axis=0)
    center = (low + high) / 2.0
    half_extent = float(np.max(high - low) / 2.0)
    half_extent = max(half_extent, 1e-6)
    padding = 1.08
    return tuple(
        (float(c - padding * half_extent), float(c + padding * half_extent))
        for c in center
    )


def reference_colors(colors: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Map real point colors to the reference image's warm-brown/teal palette."""
    low, high = np.percentile(depth_m, [5.0, 95.0])
    depth_mix = np.clip((depth_m - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    warm_brown = np.asarray([151.0, 103.0, 75.0], dtype=np.float32)
    muted_teal = np.asarray([55.0, 126.0, 137.0], dtype=np.float32)
    palette = (
        warm_brown[None, :] * (1.0 - depth_mix[:, None])
        + muted_teal[None, :] * depth_mix[:, None]
    )
    luminance = (
        colors[:, 0].astype(np.float32) * 0.299
        + colors[:, 1].astype(np.float32) * 0.587
        + colors[:, 2].astype(np.float32) * 0.114
    ) / 255.0
    shading = 0.48 + 0.72 * luminance
    return np.clip(palette * shading[:, None], 0, 255).astype(np.uint8)


def auto_frame_image(path: Path, width: int, height: int) -> None:
    image = np.asarray(Image.open(path).convert("RGB"))
    foreground = np.any(image < 248, axis=2)
    ys, xs = np.nonzero(foreground)
    if xs.size == 0:
        return
    # Detached depth fragments can otherwise dominate the framing. Fit around
    # the dense central 89% while retaining a small margin, like the reference.
    padding = 16
    x0 = max(int(np.percentile(xs, 10.0)) - padding, 0)
    x1 = min(int(np.percentile(xs, 99.0)) + padding + 1, image.shape[1])
    y0 = max(int(np.percentile(ys, 10.0)) - padding, 0)
    y1 = min(int(np.percentile(ys, 99.0)) + padding + 1, image.shape[0])
    crop = Image.fromarray(image[y0:y1, x0:x1], mode="RGB")
    target_width = int(width * 0.88)
    target_height = int(height * 0.78)
    crop.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    left = (width - crop.width) // 2
    top = (height - crop.height) // 2
    canvas.paste(crop, (left, top))
    canvas.save(path)


def render_front(
    output: Path,
    width: int,
    height: int,
    pixels_uv: np.ndarray,
    colors: np.ndarray,
    stride: int,
    radius: int,
) -> int:
    if stride < 1:
        raise ValueError("--front-stride must be >= 1")
    if radius < 0:
        raise ValueError("--front-point-radius must be >= 0")
    xy = np.rint(pixels_uv).astype(np.int64)
    keep = (xy[:, 0] % stride == 0) & (xy[:, 1] % stride == 0)
    xy = xy[keep]
    colors = colors[keep]
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    xy = xy[inside]
    colors = colors[inside]

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = xy[:, 0] + dx
            y = xy[:, 1] + dy
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            canvas[y[valid], x[valid]] = colors[valid]
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(output)
    return int(xy.shape[0])


def render_near_front(
    output: Path,
    width: int,
    height: int,
    points_world: np.ndarray,
    source_pixels_uv: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    R_base_cam: np.ndarray,
    t_base_cam: np.ndarray,
    stride: int,
    radius: int,
    yaw_deg: float,
    pitch_deg: float,
) -> int:
    R_world_to_cam = R_base_cam.T
    t_world_to_cam = -R_world_to_cam @ t_base_cam
    points_cam = points_world @ R_world_to_cam.T + t_world_to_cam

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    R_yaw = np.asarray(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    R_pitch = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    pivot = np.asarray([0.0, 0.0, np.median(points_cam[:, 2])], dtype=np.float64)
    points_view = (points_cam - pivot) @ (R_pitch @ R_yaw).T + pivot
    uvw = points_view @ K.T
    projected_uv = uvw[:, :2] / (uvw[:, 2:3] + 1e-12)

    source_xy = np.rint(source_pixels_uv).astype(np.int64)
    keep = (
        (source_xy[:, 0] % stride == 0)
        & (source_xy[:, 1] % stride == 0)
        & (points_view[:, 2] > 0)
    )
    projected_xy = np.rint(projected_uv[keep]).astype(np.int64)
    point_colors = colors[keep]
    point_depth = points_view[keep, 2]
    inside = (
        (projected_xy[:, 0] >= 0)
        & (projected_xy[:, 0] < width)
        & (projected_xy[:, 1] >= 0)
        & (projected_xy[:, 1] < height)
    )
    projected_xy = projected_xy[inside]
    point_colors = point_colors[inside]
    point_depth = point_depth[inside]

    order = np.argsort(point_depth)[::-1]
    projected_xy = projected_xy[order]
    point_colors = point_colors[order]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = projected_xy[:, 0] + dx
            y = projected_xy[:, 1] + dy
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            canvas[y[valid], x[valid]] = point_colors[valid]
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(output)
    return int(projected_xy.shape[0])


def main() -> int:
    args = parse_args()
    payload = np.load(args.pointcloud)
    points = np.asarray(payload["points_world"], dtype=np.float32)
    colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
    depth_m = np.asarray(payload["depth_m"], dtype=np.float32)
    source_pixels_uv = np.asarray(payload["source_pixels_uv"], dtype=np.float32)
    K = np.asarray(payload["K"], dtype=np.float64)
    R_base_cam = np.asarray(payload["R_base_cam"], dtype=np.float64)
    t_base_cam = np.asarray(payload["t_base_cam"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points_world [N,3], got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"Expected colors_rgb {points.shape}, got {colors.shape}")

    valid = np.isfinite(points).all(axis=1)
    if args.max_render_depth_m > 0:
        valid &= depth_m < args.max_render_depth_m
    points = points[valid]
    colors = colors[valid]
    source_pixels_uv = source_pixels_uv[valid]
    depth_m = depth_m[valid]
    if points.shape[0] == 0:
        raise ValueError("No points remain after visualization filtering")
    if args.point_brightness <= 0:
        raise ValueError("--point-brightness must be > 0")
    if args.color_style == "reference":
        colors = reference_colors(colors, depth_m)
    colors = np.clip(
        colors.astype(np.float32) * args.point_brightness,
        0,
        255,
    ).astype(np.uint8)

    if args.view_mode == "front":
        rendered_count = render_front(
            args.output,
            args.width,
            args.height,
            source_pixels_uv,
            colors,
            args.front_stride,
            args.front_point_radius,
        )
        print(f"pointcloud={args.pointcloud.resolve()}")
        print(f"output={args.output.resolve()}")
        print(f"rendered_points={rendered_count}")
        print(f"max_render_depth_m={args.max_render_depth_m}")
        print("view=front_camera_optical_axis")
        return 0

    if args.view_mode == "near-front":
        rendered_count = render_near_front(
            args.output,
            args.width,
            args.height,
            points,
            source_pixels_uv,
            colors,
            K,
            R_base_cam,
            t_base_cam,
            args.front_stride,
            args.front_point_radius,
            args.yaw,
            args.pitch,
        )
        print(f"pointcloud={args.pointcloud.resolve()}")
        print(f"output={args.output.resolve()}")
        print(f"rendered_points={rendered_count}")
        print(f"max_render_depth_m={args.max_render_depth_m}")
        print(f"view=near_front,yaw:{args.yaw},pitch:{args.pitch}")
        return 0

    if points.shape[0] > args.max_render_points:
        selected = np.linspace(
            0,
            points.shape[0] - 1,
            args.max_render_points,
            dtype=np.int64,
        )
        points = points[selected]
        colors = colors[selected]

    # Draw far points first so near points visually occlude them.
    order = np.argsort(points[:, 1])
    points = points[order]
    colors = colors[order]
    limits = equal_limits(points)

    dpi = 100
    figure = plt.figure(
        figsize=(args.width / dpi, args.height / dpi),
        dpi=dpi,
        facecolor="white",
    )
    axes = figure.add_subplot(111, projection="3d")
    axes.set_facecolor("white")
    axes.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors.astype(np.float32) / 255.0,
        s=args.point_size,
        marker=".",
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    axes.set_xlim(*limits[0])
    axes.set_ylim(*limits[1])
    axes.set_zlim(*limits[2])
    axes.set_box_aspect((1.0, 1.0, 1.0), zoom=args.zoom)
    axes.view_init(elev=args.elevation, azim=args.azimuth)
    axes.set_axis_off()
    axes.set_proj_type("ortho")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        dpi=dpi,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)
    if args.auto_frame:
        auto_frame_image(args.output, args.width, args.height)
    print(f"pointcloud={args.pointcloud.resolve()}")
    print(f"output={args.output.resolve()}")
    print(f"rendered_points={points.shape[0]}")
    print(f"max_render_depth_m={args.max_render_depth_m}")
    print(f"view=elev:{args.elevation},azim:{args.azimuth},zoom:{args.zoom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
