#!/usr/bin/env python3
"""Verify RealSense SDK intrinsics at the standard collection profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pyrealsense2 as rs

from realsense_standard import STANDARD_RS_FPS, STANDARD_RS_HEIGHT, STANDARD_RS_WIDTH


DEFAULT_CHECKERBOARD = (10, 7)
DEFAULT_SQUARE_SIZE = 0.025


def make_checkerboard_points(inner_corners: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = inner_corners
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= square_size
    return points


def print_intrinsics(intr, depth_scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    K = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array(intr.coeffs, dtype=np.float64)

    print(f"RealSense SDK color intrinsics ({intr.width}x{intr.height}):")
    print(f"  fx={intr.fx:.4f}  fy={intr.fy:.4f}")
    print(f"  cx={intr.ppx:.4f}  cy={intr.ppy:.4f}")
    print(f"  distortion_model={intr.model}")
    print(f"  dist={intr.coeffs}")
    print(f"  depth_scale={depth_scale_m:.9f} m/unit")
    print("  fuser image_size_orig=(480, 848)")
    print(f"  K:\n{K}")
    return K, dist


def detect_and_report(
    image: np.ndarray,
    object_points: np.ndarray,
    inner_corners: tuple[int, int],
    K: np.ndarray,
    dist: np.ndarray,
) -> None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0) | getattr(cv2, "CALIB_CB_ACCURACY", 0)
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, inner_corners, flags)
        if ok:
            corners = corners.astype(np.float32)
    else:
        ok, corners = cv2.findChessboardCorners(gray, inner_corners, None)
        if ok:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not ok or corners is None:
        print("棋盘格未检测到。")
        return

    pnp_ok, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not pnp_ok:
        print("solvePnP 失败。")
        return

    imgpoints, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    errors = np.linalg.norm(corners.reshape(-1, 2) - imgpoints.reshape(-1, 2), axis=1)
    R, _ = cv2.Rodrigues(rvec)

    print(f"\n棋盘格检测成功，{len(corners)} 个角点")
    print(f"重投影误差: mean={errors.mean():.4f}px  max={errors.max():.4f}px  std={errors.std():.4f}px")
    print("棋盘格在相机坐标系中的位置:")
    print(f"  t={tvec.ravel()}")
    print(f"  distance={np.linalg.norm(tvec):.4f}m ({np.linalg.norm(tvec) * 1000:.1f}mm)")
    print(f"  R=\n{R}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify RealSense intrinsics at 848x480@30.")
    parser.add_argument("--serial", default=None, help="RealSense serial number")
    parser.add_argument("--width", type=int, default=STANDARD_RS_WIDTH)
    parser.add_argument("--height", type=int, default=STANDARD_RS_HEIGHT)
    parser.add_argument("--fps", type=int, default=STANDARD_RS_FPS)
    parser.add_argument("--inner-corners", type=int, nargs=2, default=list(DEFAULT_CHECKERBOARD), metavar=("COLS", "ROWS"))
    parser.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE, help="checkerboard square size in meters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inner_corners = tuple(args.inner_corners)
    object_points = make_checkerboard_points(inner_corners, args.square_size)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    try:
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
        K, dist = print_intrinsics(intr, float(depth_scale_m))

        frames = pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("no color frame")
        image = np.asanyarray(color_frame.get_data())
        detect_and_report(image, object_points, inner_corners, K, dist)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
