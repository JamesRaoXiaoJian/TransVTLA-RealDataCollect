#!/usr/bin/env python3
"""Calibrate the fixed world RealSense camera against the robot base frame.

Recommended setup:
  1. Fix the world RealSense in its final collection position.
  2. Use either:
     - EE reference: rigidly mount the checkerboard to the gripper/end-effector.
     - Base reference: fix the checkerboard at a measured pose in the robot base frame.
  3. Capture samples with the board visible.
  4. Solve and paste the printed CameraParams block into fuser/camera.py.

The exported fuser R/t convention is T_base_cam:
  - R: camera axes expressed in robot base/world coordinates.
  - t: camera origin expressed in robot base/world coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from realsense_standard import (
    DEPTH_PNG_UNIT,
    DEPTH_PNG_UNIT_M,
    STANDARD_RS_FPS,
    STANDARD_RS_HEIGHT,
    STANDARD_RS_WIDTH,
    standard_realsense_profile,
)

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


DEFAULT_INNER_CORNERS = (10, 7)
DEFAULT_SQUARE_SIZE = 0.025
DEFAULT_WIDTH = STANDARD_RS_WIDTH
DEFAULT_HEIGHT = STANDARD_RS_HEIGHT
DEFAULT_FPS = STANDARD_RS_FPS
DEFAULT_ARM_HOST = "192.168.31.92"
DEFAULT_ARM_PORT = 8080

OUTPUT_DIR = Path(__file__).parent
DEFAULT_DATA = OUTPUT_DIR / "world_realsense_calibration.json"
DEFAULT_EXPORT = OUTPUT_DIR / "world_realsense_extrinsics.json"
DEFAULT_IMAGE_DIR = OUTPUT_DIR / "world_realsense_calib_images"


@dataclass
class CaptureConfig:
    serial: str | None
    width: int
    height: int
    fps: int
    inner_corners: tuple[int, int]
    square_size: float
    data_path: Path
    image_dir: Path


class RobotArmAPI:
    """Minimal wrapper for reading T_base_ee from the RealMan arm SDK."""

    def __init__(self, host: str, port: int):
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.robot.rm_create_robot_arm(host, port)
        print(f"Connected robot arm: {host}:{port}")

    def get_ee_pose(self) -> np.ndarray:
        code, state = self.robot.rm_get_current_arm_state()
        if code != 0:
            raise RuntimeError(f"failed to read robot state, code={code}")

        pose = state["pose"]
        rm_matrix = self.robot.rm_algo_pos2matrix(pose)
        return np.array(rm_matrix.data, dtype=np.float64).reshape(4, 4)

    def disconnect(self) -> None:
        self.robot.rm_delete_robot_arm()
        print("Robot arm disconnected")


def now_us() -> int:
    return time.time_ns() // 1000


def make_checkerboard_points(inner_corners: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = inner_corners
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    pts *= square_size
    return pts


def detect_checkerboard(
    image: np.ndarray,
    inner_corners: tuple[int, int],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
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
        return False, None, None, None, None

    try:
        pnp_ok, rvec, tvec = cv2.solvePnP(
            object_points,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        pnp_ok, rvec, tvec = cv2.solvePnP(
            object_points,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not pnp_ok:
        return False, corners, None, None, None

    proj, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_px = float(cv2.norm(corners, proj, cv2.NORM_L2) / len(proj))
    return True, corners, rvec, tvec, reproj_px


def matrix_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inv


def rpy_deg_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = [math.radians(v) for v in (roll, pitch, yaw)]
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]],
        dtype=np.float64,
    )
    Ry = np.array(
        [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]],
        dtype=np.float64,
    )
    Rz = np.array(
        [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return Rz @ Ry @ Rx


def pose_to_matrix(xyz: list[float], rpy_deg: list[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_deg_to_matrix(*rpy_deg)
    T[:3, 3] = np.array(xyz, dtype=np.float64)
    return T


def load_transform_json(path: Path, key: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"{path} does not contain '{key}'")
    T = np.array(data[key], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{key} in {path} must be a 4x4 matrix")
    return T


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s],
            dtype=np.float64,
        )
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array(
                [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s],
                dtype=np.float64,
            )
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array(
                [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s],
                dtype=np.float64,
            )
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array(
                [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s],
                dtype=np.float64,
            )
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("no transforms to average")

    translations = np.stack([T[:3, 3] for T in transforms], axis=0)
    quats = np.stack([rotation_matrix_to_quaternion(T[:3, :3]) for T in transforms], axis=0)

    A = np.zeros((4, 4), dtype=np.float64)
    for q in quats:
        A += np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, int(np.argmax(eigvals))]
    if q_avg[0] < 0:
        q_avg = -q_avg

    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = quaternion_to_rotation_matrix(q_avg)
    T_avg[:3, 3] = np.median(translations, axis=0)
    return T_avg


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_err = R_a.T @ R_b
    cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_theta)))


def format_matrix_rows(M: np.ndarray, indent: str = "            ") -> str:
    lines = []
    for row in M:
        lines.append(f"{indent}[{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}],")
    return "\n".join(lines)


def get_realsense_intrinsics(profile: Any) -> tuple[np.ndarray, np.ndarray]:
    stream = profile.get_stream(rs.stream.color)
    intr = stream.as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    return K, dist


def start_realsense(config: CaptureConfig):
    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if config.serial:
        rs_config.enable_device(config.serial)
    rs_config.enable_stream(rs.stream.color, config.width, config.height, rs.format.bgr8, config.fps)
    rs_config.enable_stream(rs.stream.depth, config.width, config.height, rs.format.z16, config.fps)
    profile = pipeline.start(rs_config)
    K, dist = get_realsense_intrinsics(profile)
    depth_scale_m = DEPTH_PNG_UNIT_M
    try:
        depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
    except Exception:
        pass
    return pipeline, K, dist, depth_scale_m


def list_devices() -> None:
    if rs is None:
        print("pyrealsense2 is not installed")
        return
    devices = list(rs.context().query_devices())
    if not devices:
        print("No RealSense devices detected")
        return
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"{serial}\t{name}")


def load_capture_data(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "samples" not in data:
        raise ValueError(f"{path} has no samples")
    return data


def save_capture_data(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def draw_capture_ui(
    image: np.ndarray,
    corners: np.ndarray | None,
    detected: bool,
    sample_count: int,
    reproj_px: float | None,
    inner_corners: tuple[int, int],
) -> np.ndarray:
    display = image.copy()
    if detected and corners is not None:
        cv2.drawChessboardCorners(display, inner_corners, corners, True)
    color = (0, 255, 0) if detected else (0, 0, 255)
    status = "DETECTED" if detected else "SEARCHING"
    cv2.putText(display, f"Board: {status} | Captured: {sample_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if reproj_px is not None:
        cv2.putText(display, f"PnP reproj: {reproj_px:.3f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(display, "SPACE:capture  S:save  Q/ESC:quit", (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    return display


def capture(args: argparse.Namespace) -> None:
    cfg = CaptureConfig(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        inner_corners=tuple(args.inner_corners),
        square_size=args.square_size,
        data_path=Path(args.data),
        image_dir=Path(args.image_dir),
    )
    cfg.image_dir.mkdir(parents=True, exist_ok=True)

    object_points = make_checkerboard_points(cfg.inner_corners, cfg.square_size)
    robot = None if args.no_robot else RobotArmAPI(args.arm_host, args.arm_port)
    pipeline = None

    try:
        pipeline, K, dist, depth_scale_m = start_realsense(cfg)
        print(f"World RealSense intrinsics ({cfg.width}x{cfg.height}):")
        print(f"  fx={K[0,0]:.3f} fy={K[1,1]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}")
        print(f"  dist={dist.tolist()}")
        print(f"Images: {cfg.image_dir}")
        if robot is None:
            print("Robot pose capture disabled. Use solve --reference base with a measured T_base_target.")
        else:
            print("Move the robot so the board spans different image regions, depths, and tilts.")

        payload = {
            "schema": "world_realsense_calibration/v1",
            "camera": {
                "role": "world_camera",
                "serial": cfg.serial,
                "resolution": {"width": cfg.width, "height": cfg.height, "fps": cfg.fps},
                "standard_profile": standard_realsense_profile(),
                "camera_matrix": K.tolist(),
                "dist_coeffs": dist.tolist(),
                "sensor_depth_scale_m_per_unit": depth_scale_m,
                "depth_png_unit": DEPTH_PNG_UNIT,
                "depth_png_unit_m": DEPTH_PNG_UNIT_M,
            },
            "checkerboard": {
                "inner_corners": list(cfg.inner_corners),
                "square_size_m": cfg.square_size,
            },
            "samples": [],
        }

        sample_count = 0
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            ok, corners, rvec, tvec, reproj_px = detect_checkerboard(image, cfg.inner_corners, object_points, K, dist)
            display = draw_capture_ui(image, corners, ok, sample_count, reproj_px, cfg.inner_corners)

            cv2.namedWindow("World RealSense Calibration", cv2.WINDOW_NORMAL)
            cv2.imshow("World RealSense Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                save_capture_data(cfg.data_path, payload)
                print(f"Saved {len(payload['samples'])} samples to {cfg.data_path}")
                continue
            if key != ord(" "):
                continue

            if not ok or rvec is None or tvec is None:
                print("[capture] checkerboard not detected, skipped")
                continue

            T_base_ee = None
            if robot is not None:
                try:
                    T_base_ee = robot.get_ee_pose()
                except RuntimeError as exc:
                    print(f"[capture] robot pose failed: {exc}")
                    continue

            sample_count += 1
            image_name = f"world_realsense_{sample_count:03d}.png"
            image_path = cfg.image_dir / image_name
            cv2.imwrite(str(image_path), image)
            T_cam_target = matrix_from_rvec_tvec(rvec, tvec)
            try:
                image_ref = str(image_path.relative_to(cfg.data_path.parent))
            except ValueError:
                image_ref = str(image_path)

            sample = {
                "id": sample_count,
                "timestamp_us": now_us(),
                "image": image_ref,
                "reprojection_error_px": reproj_px,
                "T_cam_target": T_cam_target.tolist(),
                "rvec": rvec.reshape(3).tolist(),
                "tvec": tvec.reshape(3).tolist(),
            }
            if T_base_ee is not None:
                sample["T_base_ee"] = T_base_ee.tolist()
            payload["samples"].append(sample)

            board_pos = tvec.reshape(3)
            msg = f"[{sample_count}] target_cam=({board_pos[0]:.3f},{board_pos[1]:.3f},{board_pos[2]:.3f}) "
            if T_base_ee is not None:
                ee_pos = T_base_ee[:3, 3]
                msg += f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) "
            msg += f"reproj={reproj_px:.3f}px saved={image_name}"
            print(msg)
    finally:
        if pipeline is not None:
            pipeline.stop()
        if robot is not None:
            robot.disconnect()
        cv2.destroyAllWindows()


def transform_from_args(args: argparse.Namespace, prefix: str, json_key: str) -> np.ndarray:
    json_path = getattr(args, f"{prefix}_json")
    if json_path:
        return load_transform_json(Path(json_path), json_key)
    xyz = getattr(args, f"{prefix}_xyz")
    rpy = getattr(args, f"{prefix}_rpy_deg")
    return pose_to_matrix(xyz, rpy)


def build_reference_targets(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if args.reference == "ee":
        T_ee_target = transform_from_args(args, "ee_target", "T_ee_target")
        targets = []
        for sample in samples:
            if "T_base_ee" not in sample:
                raise ValueError("sample missing T_base_ee; use --reference base or recapture with robot pose")
            targets.append(np.array(sample["T_base_ee"], dtype=np.float64) @ T_ee_target)
        return targets, {"reference": "ee", "T_ee_target": T_ee_target.tolist()}

    T_base_target = transform_from_args(args, "base_target", "T_base_target")
    return [T_base_target for _ in samples], {"reference": "base", "T_base_target": T_base_target.tolist()}


def solve(args: argparse.Namespace) -> None:
    data = load_capture_data(Path(args.data))
    samples = data["samples"]
    if len(samples) < 4:
        raise SystemExit(f"Need at least 4 samples, got {len(samples)}")

    T_base_targets, reference_meta = build_reference_targets(samples, args)
    T_cam_targets = [np.array(sample["T_cam_target"], dtype=np.float64) for sample in samples]
    T_base_cams = [
        T_base_target @ invert_transform(T_cam_target)
        for T_base_target, T_cam_target in zip(T_base_targets, T_cam_targets)
    ]

    initial = average_transforms(T_base_cams)
    scores = []
    for T in T_base_cams:
        rot = rotation_error_deg(initial[:3, :3], T[:3, :3])
        trans_mm = np.linalg.norm(initial[:3, 3] - T[:3, 3]) * 1000.0
        scores.append(rot + trans_mm / 10.0)

    keep_indices = list(range(len(samples)))
    if args.trim_ratio > 0 and len(samples) >= 8:
        keep_n = max(4, int(round(len(samples) * (1.0 - args.trim_ratio))))
        keep_indices = sorted(np.argsort(scores)[:keep_n].tolist())

    T_base_cam = average_transforms([T_base_cams[i] for i in keep_indices])

    rot_errors = []
    trans_errors_mm = []
    per_sample = []
    for i, (T_base_target, T_cam_target, sample) in enumerate(zip(T_base_targets, T_cam_targets, samples)):
        T_pred = T_base_cam @ T_cam_target
        rot_deg = rotation_error_deg(T_base_target[:3, :3], T_pred[:3, :3])
        trans_mm = np.linalg.norm(T_base_target[:3, 3] - T_pred[:3, 3]) * 1000.0
        rot_errors.append(rot_deg)
        trans_errors_mm.append(trans_mm)
        per_sample.append(
            {
                "id": sample.get("id", i + 1),
                "used": i in keep_indices,
                "rotation_error_deg": rot_deg,
                "translation_error_mm": trans_mm,
                "reprojection_error_px": sample.get("reprojection_error_px"),
            }
        )

    K = np.array(data["camera"]["camera_matrix"], dtype=np.float64)
    dist = np.array(data["camera"]["dist_coeffs"], dtype=np.float64)
    resolution = data["camera"].get("resolution", {})

    quality = {
        "num_samples": len(samples),
        "num_used": len(keep_indices),
        "mean_rotation_error_deg": float(np.mean(rot_errors)),
        "median_rotation_error_deg": float(np.median(rot_errors)),
        "max_rotation_error_deg": float(np.max(rot_errors)),
        "mean_translation_error_mm": float(np.mean(trans_errors_mm)),
        "median_translation_error_mm": float(np.median(trans_errors_mm)),
        "max_translation_error_mm": float(np.max(trans_errors_mm)),
        "mean_reprojection_error_px": float(np.mean([s.get("reprojection_error_px", 0.0) for s in samples])),
    }

    result = {
        "schema": "world_realsense_extrinsics/v1",
        "camera": data["camera"],
        "checkerboard": data.get("checkerboard", {}),
        "reference": reference_meta,
        "T_base_cam": T_base_cam.tolist(),
        "T_cam_base": invert_transform(T_base_cam).tolist(),
        "fuser": {
            "camera_name": "world_camera",
            "K": K.tolist(),
            "R": T_base_cam[:3, :3].tolist(),
            "t": T_base_cam[:3, 3].tolist(),
            "image_size_orig": [int(resolution.get("height", DEFAULT_HEIGHT)), int(resolution.get("width", DEFAULT_WIDTH))],
            "dist_coeffs": dist.tolist(),
        },
        "quality": quality,
        "samples": per_sample,
    }

    export_path = Path(args.export)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print_quality(quality)
    print_fuser_snippet(K, T_base_cam, resolution)
    print(f"\nSaved extrinsics: {export_path}")


def print_quality(quality: dict[str, Any]) -> None:
    print("\nCalibration quality")
    print("=" * 60)
    print(f"samples: {quality['num_used']}/{quality['num_samples']} used")
    print(
        "rotation error: "
        f"mean={quality['mean_rotation_error_deg']:.3f}deg "
        f"median={quality['median_rotation_error_deg']:.3f}deg "
        f"max={quality['max_rotation_error_deg']:.3f}deg"
    )
    print(
        "translation error: "
        f"mean={quality['mean_translation_error_mm']:.2f}mm "
        f"median={quality['median_translation_error_mm']:.2f}mm "
        f"max={quality['max_translation_error_mm']:.2f}mm"
    )
    print(f"mean PnP reprojection: {quality['mean_reprojection_error_px']:.4f}px")


def print_fuser_snippet(K: np.ndarray, T_base_cam: np.ndarray, resolution: dict[str, Any]) -> None:
    width = int(resolution.get("width", DEFAULT_WIDTH))
    height = int(resolution.get("height", DEFAULT_HEIGHT))
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]

    print("\nfuser/camera.py snippet")
    print("=" * 60)
    print('    "world_camera": CameraParams(')
    print(f"        K=torch.tensor([  # {width}x{height}")
    print(f"            [{K[0,0]:.8f}, 0.0, {K[0,2]:.8f}],")
    print(f"            [0.0, {K[1,1]:.8f}, {K[1,2]:.8f}],")
    print("            [0.0, 0.0, 1.0]")
    print("        ], dtype=torch.float32),")
    print("        R=torch.tensor([")
    print(format_matrix_rows(R))
    print("        ], dtype=torch.float32),")
    print(f"        t=torch.tensor([{t[0]:.8f}, {t[1]:.8f}, {t[2]:.8f}], dtype=torch.float32)")
    print("    ),")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="World RealSense eye-to-hand calibration")
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="list connected RealSense devices")
    list_p.set_defaults(func=lambda _args: list_devices())

    capture_p = sub.add_parser("capture", help="capture calibration samples")
    capture_p.add_argument("--serial", default=None, help="world RealSense serial number")
    capture_p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    capture_p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    capture_p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    capture_p.add_argument("--inner-corners", type=int, nargs=2, default=list(DEFAULT_INNER_CORNERS), metavar=("COLS", "ROWS"))
    capture_p.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE, help="checkerboard square size in meters")
    capture_p.add_argument("--arm-host", default=DEFAULT_ARM_HOST)
    capture_p.add_argument("--arm-port", type=int, default=DEFAULT_ARM_PORT)
    capture_p.add_argument("--no-robot", action="store_true", help="do not connect to the robot; use solve --reference base")
    capture_p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    capture_p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    capture_p.set_defaults(func=capture)

    solve_p = sub.add_parser("solve", help="solve T_base_cam from saved samples")
    solve_p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    solve_p.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    solve_p.add_argument("--reference", choices=["ee", "base"], default="ee")
    solve_p.add_argument("--trim-ratio", type=float, default=0.15, help="discard worst ratio after an initial estimate when sample count >= 8")
    solve_p.add_argument("--ee-target-json", default=None, help="JSON containing T_ee_target")
    solve_p.add_argument("--ee-target-xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"))
    solve_p.add_argument("--ee-target-rpy-deg", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("ROLL", "PITCH", "YAW"))
    solve_p.add_argument("--base-target-json", default=None, help="JSON containing T_base_target")
    solve_p.add_argument("--base-target-xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"))
    solve_p.add_argument("--base-target-rpy-deg", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("ROLL", "PITCH", "YAW"))
    solve_p.set_defaults(func=solve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
