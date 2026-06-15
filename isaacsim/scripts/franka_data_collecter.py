from __future__ import annotations

import argparse
import csv
from concurrent.futures import Future, ThreadPoolExecutor, wait
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from channel_config import VALID_CHANNELS
except Exception:
    VALID_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

try:
    from realsense_standard import (
        CAMERA_METADATA_FILE,
        DEPTH_PNG_UNIT,
        DEPTH_PNG_UNIT_M,
        STANDARD_RS_FPS,
        STANDARD_RS_HEIGHT,
        STANDARD_RS_WIDTH,
        standard_realsense_profile,
    )
except Exception:
    CAMERA_METADATA_FILE = "camera_metadata.json"
    DEPTH_PNG_UNIT = "millimeter"
    DEPTH_PNG_UNIT_M = 0.001
    STANDARD_RS_FPS = 30
    STANDARD_RS_HEIGHT = 480
    STANDARD_RS_WIDTH = 848

    def standard_realsense_profile() -> dict[str, Any]:
        return {
            "width": STANDARD_RS_WIDTH,
            "height": STANDARD_RS_HEIGHT,
            "fps": STANDARD_RS_FPS,
            "color_stream": "color",
            "color_format": "bgr8",
            "depth_stream": "depth",
            "depth_format": "z16",
            "depth_aligned_to": "color",
            "depth_png_dtype": "uint16",
            "depth_png_unit": DEPTH_PNG_UNIT,
            "depth_png_unit_m": DEPTH_PNG_UNIT_M,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-env Franka front-box data collection with world/wrist cameras and transparent cube masks."
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the Isaac Sim GUI.")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Run with the Isaac Sim GUI.")
    parser.set_defaults(headless=True)
    parser.add_argument("--env-usd", type=Path, default=Path("USDFiles") / "franka_env.usd")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--env-spacing", type=float, default=3.5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps-per-episode", type=int, default=360)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--save-image-interval", type=int, default=10)
    parser.add_argument(
        "--render-every-step",
        action="store_true",
        help="Render every simulation step. By default, headless runs render only capture frames for faster SDG.",
    )
    parser.add_argument(
        "--camera-warmup-render-steps",
        type=int,
        default=1,
        help="Rendered warmup frames after camera annotators are attached.",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs" / "collected_data")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--kit-log-level",
        type=str,
        choices=["Verbose", "Info", "Warning", "Error", "Fatal"],
        default="Error",
        help="Minimum Isaac Kit log level sent to stdout/stderr and Kit log files.",
    )
    parser.add_argument(
        "--enable-crashreporter",
        action="store_true",
        help="Enable Isaac crash reporter plugin. Disabled by default to avoid previous-crash startup noise.",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Worker threads for asynchronous .npy/.npz disk writes. Use 1 for a single writer thread.",
    )
    parser.add_argument(
        "--max-pending-saves",
        type=int,
        default=128,
        help="Maximum queued async save jobs before the simulation thread applies backpressure.",
    )
    parser.add_argument("--transparent-label", type=str, default="transparent_obj")
    parser.add_argument("--debug-masks", action="store_true", help="Print semantic mask debug information during image capture.")
    parser.add_argument("--world-camera-width", type=int, default=640)
    parser.add_argument("--world-camera-height", type=int, default=480)
    parser.add_argument("--wrist-camera-width", type=int, default=320)
    parser.add_argument("--wrist-camera-height", type=int, default=240)
    parser.add_argument(
        "--save-legacy-arrays",
        action="store_true",
        help="Also save the previous .npy/.npz simulation arrays for debugging.",
    )
    parser.add_argument(
        "--pressure-contact-value",
        type=int,
        default=800,
        help="Synthetic tactile ADC drop used when the gripper contacts the cube.",
    )
    parser.add_argument(
        "--pressure-idle-value",
        type=int,
        default=3500,
        help="Synthetic tactile idle ADC baseline written when there is no contact.",
    )
    parser.add_argument(
        "--visual-fps",
        type=int,
        default=STANDARD_RS_FPS,
        help="Nominal FPS written to metadata and synthetic frame timestamps.",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--active-gpu", type=int, default=0, help="Renderer GPU index for Isaac Sim launch config.")
    parser.add_argument(
        "--max-gpu-count",
        type=int,
        default=1,
        help="Maximum renderer GPUs. Keep at 1 for stable headless SDG on multi-GPU workstations.",
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Enable Isaac Sim multi-GPU rendering.")
    parser.add_argument(
        "--ik-method",
        type=str,
        choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
        default="damped-least-squares",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cube-x-min", type=float, default=0.35)
    parser.add_argument("--cube-x-max", type=float, default=0.65)
    parser.add_argument("--cube-y-min", type=float, default=-0.35)
    parser.add_argument("--cube-y-max", type=float, default=0.35)
    parser.add_argument("--min-cube-box-distance", type=float, default=0.25)
    parser.add_argument("--place-approach-clearance", type=float, default=0.18)
    parser.add_argument("--place-release-height", type=float, default=0.12)
    parser.add_argument("--target-random-half-range", type=float, default=0.03)
    parser.add_argument("--no-save-data", action="store_false", dest="save_data")
    parser.set_defaults(save_data=True)
    return parser.parse_args()


args = parse_args()


def configure_python_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)


def build_simulation_app_config() -> dict[str, Any]:
    kit_log_args = [
        f"--/log/level={args.kit_log_level}",
        f"--/log/outputStreamLevel={args.kit_log_level}",
        f"--/log/debugConsoleLevel={args.kit_log_level}",
        f"--/log/fileLogLevel={args.kit_log_level}",
    ]
    return {
        "headless": args.headless,
        "active_gpu": args.active_gpu,
        "physics_gpu": args.active_gpu,
        "multi_gpu": args.multi_gpu,
        "max_gpu_count": args.max_gpu_count,
        "enable_crashreporter": args.enable_crashreporter,
        "extra_args": kit_log_args,
    }


configure_python_output()

from isaacsim import SimulationApp

simulation_app = SimulationApp(build_simulation_app_config())

import cv2
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.examples.franka.franka_experimental import FrankaExperimental
from isaacsim.sensors.camera import Camera
from pxr import Gf, Usd, UsdGeom


PHASE_NAMES = {
    0: "move_above_cube",
    1: "approach_cube",
    2: "close_gripper",
    3: "lift_cube",
    4: "move_high_above_box",
    5: "lower_straight_into_box",
    6: "open_gripper",
    7: "retract",
}


class LoadedFrontBoxFrankaPickPlace:
    def __init__(self, robot: FrankaExperimental, cube: RigidPrim):
        self.robot = robot
        self.cube = cube
        self.events_dt = [60, 40, 20, 40, 90, 55, 20, 25]
        self._event = 0
        self._step = 0
        self.cube_size = np.array([0.0515, 0.0515, 0.0515])
        self.cube_initial_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.target_position: np.ndarray | None = None
        self.place_approach_position: np.ndarray | None = None
        self.box_center_position: np.ndarray | None = None
        self.box_inner_half_size: np.ndarray | None = None

    def forward(self, ik_method: str) -> bool:
        if self.is_done():
            return False
        goal_orientation = self.robot.get_downward_orientation()
        if self._event == 0:
            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = np.array([cube_pos[0, 0], cube_pos[0, 1], cube_pos[0, 2] + 0.2])
        elif self._event == 1:
            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = cube_pos + np.array([0.0, 0.0, 0.1])
        elif self._event == 2:
            self.robot.close_gripper()
            goal_position = None
        elif self._event == 3:
            _, current_position, _ = self.robot.get_current_state()
            goal_position = current_position + np.array([0.0, 0.0, 0.2])
        elif self._event == 4:
            goal_position = np.asarray(self.place_approach_position, dtype=float)
        elif self._event == 5:
            goal_position = np.asarray(self.target_position, dtype=float)
        elif self._event == 6:
            self.robot.open_gripper()
            goal_position = None
        elif self._event == 7:
            goal_position = np.asarray(self.place_approach_position, dtype=float)
        else:
            goal_position = None

        if goal_position is not None:
            self.robot.set_end_effector_pose(
                position=np.asarray(goal_position, dtype=float),
                orientation=goal_orientation,
                ik_method=ik_method,
            )

        self._step += 1
        if self._step >= self.events_dt[self._event]:
            self._event += 1
            self._step = 0
        return True

    def is_done(self) -> bool:
        return self._event >= len(self.events_dt)

    def reset(self, cube_position: np.ndarray) -> None:
        self.robot.reset_to_default_pose()
        self.cube.set_world_poses(
            positions=np.asarray(cube_position, dtype=float).reshape(1, -1),
            orientations=self.cube_initial_orientation.reshape(1, -1),
        )
        self._event = 0
        self._step = 0


def as_array(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def align_camera_aperture(stage: Usd.Stage, prim_path: str, resolution: tuple[int, int]) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        return
    width, height = resolution
    if width <= 0 or height <= 0:
        return
    horizontal_attr = prim.GetAttribute("horizontalAperture")
    vertical_attr = prim.GetAttribute("verticalAperture")
    horizontal_aperture = horizontal_attr.Get()
    if horizontal_aperture is None:
        return
    expected_vertical_aperture = float(horizontal_aperture) / (float(width) / float(height))
    current_vertical_aperture = vertical_attr.Get()
    if current_vertical_aperture is None or not np.isclose(
        float(current_vertical_aperture), expected_vertical_aperture, rtol=1e-5, atol=1e-8
    ):
        vertical_attr.Set(expected_vertical_aperture)


def should_render_step(next_step: int) -> bool:
    if not args.headless or args.render_every_step:
        return True
    return args.save_data and args.save_image_interval > 0 and next_step % args.save_image_interval == 0


def monotonic_us() -> int:
    return time.monotonic_ns() // 1_000


def image_to_bgr(image: np.ndarray | None) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.shape[-1] >= 3:
        arr = arr[:, :, :3]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def depth_meters_to_uint16_mm(depth: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if depth is None or depth.size == 0:
        return np.zeros((height, width), dtype=np.uint16)
    arr = np.asarray(depth, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        return np.zeros((height, width), dtype=np.uint16)
    depth_mm = np.rint(np.clip(arr, 0.0, 65.535) / DEPTH_PNG_UNIT_M)
    return depth_mm.astype(np.uint16)


def quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float).reshape(-1)
    if q.size < 4:
        return np.zeros(3, dtype=float)
    q = q[:4]
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.zeros(3, dtype=float)
    q = q / norm
    if q[0] < 0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    xyz = q[1:4]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half <= 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(sin_half, w)
    return xyz / sin_half * angle


def pose6_from_state(position: Any, orientation: Any) -> list[float]:
    pos = as_array(position).reshape(-1)[:3]
    rotvec = quat_wxyz_to_rotvec(as_array(orientation))
    return [float(v) for v in np.concatenate([pos, rotvec])]


def camera_intrinsics_from_usd(
    stage: Usd.Stage,
    prim_path: str,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    width, height = resolution
    fx = float(width)
    fy = float(width)
    ppx = float(width) / 2.0
    ppy = float(height) / 2.0

    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and prim.IsA(UsdGeom.Camera):
        focal = prim.GetAttribute("focalLength").Get()
        horizontal_aperture = prim.GetAttribute("horizontalAperture").Get()
        vertical_aperture = prim.GetAttribute("verticalAperture").Get()
        if focal and horizontal_aperture:
            fx = float(focal) / float(horizontal_aperture) * float(width)
        if focal and vertical_aperture:
            fy = float(focal) / float(vertical_aperture) * float(height)

    return {
        "width": int(width),
        "height": int(height),
        "fx": fx,
        "fy": fy,
        "ppx": ppx,
        "ppy": ppy,
        "model": "sim_pinhole",
        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "K": [
            [fx, 0.0, ppx],
            [0.0, fy, ppy],
            [0.0, 0.0, 1.0],
        ],
    }


class RealSchemaSessionWriter:
    def __init__(
        self,
        root: Path,
        session_name: str,
        *,
        env_id: int,
        episode_id: int,
        scene_config: dict[str, np.ndarray],
        env: dict[str, Any],
        stage: Usd.Stage,
    ):
        self.root = root / session_name
        self.env_id = env_id
        self.episode_id = episode_id
        self.frame_id = 0
        self._closed = False
        self._last_robot_timestamp_us: int | None = None
        self._last_gripper_timestamp_us: int | None = None
        self._last_pressure_timestamp_us: int | None = None

        self.world_rgb_dir = self.root / "world_camera" / "rgb"
        self.world_depth_dir = self.root / "world_camera" / "depth"
        self.wrist_rgb_dir = self.root / "wrist_camera" / "rgb"
        self.wrist_depth_dir = self.root / "wrist_camera" / "depth"
        self.robot_dir = self.root / "robot_state"
        self.pressure_dir = self.root / "pressure"
        for directory in (
            self.world_rgb_dir,
            self.world_depth_dir,
            self.wrist_rgb_dir,
            self.wrist_depth_dir,
            self.robot_dir,
            self.pressure_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.frames_file = open(self.root / "frames.csv", "w", newline="", encoding="utf-8")
        self.frames_writer = csv.writer(self.frames_file)
        self.frames_writer.writerow(
            [
                "frame_id",
                "capture_monotonic_us",
                "world_rgb_save_us",
                "world_depth_save_us",
                "wrist_rgb_save_us",
                "wrist_depth_save_us",
                "save_complete_us",
                "save_queue_depth",
            ]
        )

        self.robot_file = open(self.robot_dir / "robot_state.csv", "w", newline="", encoding="utf-8")
        self.robot_writer = csv.writer(self.robot_file)
        self.robot_writer.writerow(["timestamp_us"] + [f"joint_{i + 1}" for i in range(7)] + [f"pose_{i + 1}" for i in range(6)])

        self.gripper_file = open(self.robot_dir / "gripper_state.csv", "w", newline="", encoding="utf-8")
        self.gripper_writer = csv.writer(self.gripper_file)
        self.gripper_writer.writerow(
            [
                "timestamp_us",
                "target_hz",
                "rm_plus_read_code",
                "rm_plus_read_latency_ms",
                "sys_state",
                "gripper_pos",
                "gripper_speed",
                "gripper_current",
                "gripper_force",
                "gripper_dof_state",
                "gripper_dof_err",
                "deadline_late_ms",
            ]
        )

        self.pressure_file = open(self.pressure_dir / "pressure.csv", "w", newline="", encoding="utf-8")
        self.pressure_writer = csv.writer(self.pressure_file)
        self.pressure_writer.writerow(["sensor_timestamp_us", "host_monotonic_us"] + [f"CH{i}" for i in VALID_CHANNELS])

        self._write_camera_metadata(env, stage)
        self._write_sim_metadata(scene_config, env)

    def write_sample(
        self,
        *,
        timestamp_us: int,
        world_rgb: np.ndarray | None,
        world_depth_m: np.ndarray | None,
        wrist_rgb: np.ndarray | None,
        wrist_depth_m: np.ndarray | None,
        joint_positions: np.ndarray,
        ee_position: Any,
        ee_orientation: Any,
        gripper_pos: float,
        contact: bool,
    ) -> None:
        if self._closed:
            raise RuntimeError(f"Cannot write to closed session: {self.root}")

        self.frame_id += 1
        rgb_name = f"{self.frame_id:04d}.jpg"
        depth_name = f"{self.frame_id:04d}.png"

        cv2.imwrite(str(self.world_rgb_dir / rgb_name), image_to_bgr(world_rgb), [cv2.IMWRITE_JPEG_QUALITY, 85])
        world_rgb_save_us = monotonic_us()
        world_depth_shape = (args.world_camera_height, args.world_camera_width)
        cv2.imwrite(str(self.world_depth_dir / depth_name), depth_meters_to_uint16_mm(world_depth_m, world_depth_shape))
        world_depth_save_us = monotonic_us()
        cv2.imwrite(str(self.wrist_rgb_dir / rgb_name), image_to_bgr(wrist_rgb), [cv2.IMWRITE_JPEG_QUALITY, 85])
        wrist_rgb_save_us = monotonic_us()
        wrist_depth_shape = (args.wrist_camera_height, args.wrist_camera_width)
        cv2.imwrite(str(self.wrist_depth_dir / depth_name), depth_meters_to_uint16_mm(wrist_depth_m, wrist_depth_shape))
        wrist_depth_save_us = monotonic_us()
        save_complete_us = monotonic_us()

        self.frames_writer.writerow(
            [
                self.frame_id,
                timestamp_us,
                world_rgb_save_us,
                world_depth_save_us,
                wrist_rgb_save_us,
                wrist_depth_save_us,
                save_complete_us,
                0,
            ]
        )

        joints = [float(v) for v in np.asarray(joint_positions, dtype=float).reshape(-1)[:7]]
        joints += [0.0] * max(0, 7 - len(joints))
        pose6 = pose6_from_state(ee_position, ee_orientation)
        for ts in self._sample_times(timestamp_us, "_last_robot_timestamp_us", 10_000):
            self.robot_writer.writerow([ts] + joints[:7] + pose6)

        idle_pressure = int(args.pressure_idle_value)
        pressure_value = max(0, idle_pressure - int(args.pressure_contact_value)) if contact else idle_pressure
        pressure_values = [pressure_value for _ in VALID_CHANNELS]
        for ts in self._sample_times(timestamp_us, "_last_pressure_timestamp_us", 5_000):
            self.pressure_writer.writerow([ts, ts] + pressure_values)

        for ts in self._sample_times(timestamp_us, "_last_gripper_timestamp_us", 10_000):
            self.gripper_writer.writerow(
                [
                    ts,
                    100,
                    0,
                    0.0,
                    "sim",
                    round(float(gripper_pos), 6),
                    "",
                    "",
                    int(args.pressure_contact_value) if contact else 0,
                    int(contact),
                    "",
                    0.0,
                ]
            )

    def close(self) -> None:
        if self._closed:
            return
        for file_obj in (self.frames_file, self.robot_file, self.gripper_file, self.pressure_file):
            file_obj.flush()
            file_obj.close()
        self._closed = True

    def _sample_times(self, timestamp_us: int, attr_name: str, interval_us: int) -> list[int]:
        last = getattr(self, attr_name)
        if last is None:
            setattr(self, attr_name, timestamp_us)
            return [timestamp_us]

        samples = list(range(last + interval_us, timestamp_us + 1, interval_us))
        if not samples:
            samples = [timestamp_us]
        setattr(self, attr_name, samples[-1])
        return samples

    def _write_camera_metadata(self, env: dict[str, Any], stage: Usd.Stage) -> None:
        standard_profile = standard_realsense_profile()
        standard_profile.update(
            {
                "width": args.world_camera_width,
                "height": args.world_camera_height,
                "fps": args.visual_fps,
            }
        )

        def camera_payload(name: str, prim_path: str, resolution: tuple[int, int]) -> dict[str, Any]:
            intrinsics = camera_intrinsics_from_usd(stage, prim_path, resolution)
            width, height = resolution
            return {
                "name": name,
                "available": True,
                "serial_number": f"sim_{name}_env{self.env_id}",
                "profile": {
                    **standard_realsense_profile(),
                    "width": width,
                    "height": height,
                    "fps": args.visual_fps,
                },
                "color": {"intrinsics": intrinsics},
                "depth": {
                    "intrinsics": intrinsics,
                    "aligned_to": "color",
                    "saved_pixel_intrinsics_source": "color",
                    "sensor_depth_scale_m_per_unit": DEPTH_PNG_UNIT_M,
                    "saved_png_unit": DEPTH_PNG_UNIT,
                    "saved_png_unit_m": DEPTH_PNG_UNIT_M,
                    "saved_png_dtype": "uint16",
                },
                "sim_prim_path": prim_path,
            }

        payload = {
            "schema": "dual_realsense_camera_metadata/v1",
            "standard_profile": standard_profile,
            "visual_recording_fps": args.visual_fps,
            "depth_note": "Sim depth PNG files are aligned to RGB/color pixels and saved as uint16 millimeters.",
            "cameras": {
                "world_camera": camera_payload(
                    "world_camera",
                    env["world_camera_path"],
                    (args.world_camera_width, args.world_camera_height),
                ),
                "wrist_camera": camera_payload(
                    "wrist_camera",
                    env["wrist_camera_path"],
                    (args.wrist_camera_width, args.wrist_camera_height),
                ),
            },
        }
        (self.root / CAMERA_METADATA_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_sim_metadata(self, scene_config: dict[str, np.ndarray], env: dict[str, Any]) -> None:
        payload = {
            "schema": "transvtla_isaacsim_session/v1",
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "env_path": env["env_path"],
            "robot_path": env["robot_path"],
            "cube_path": env["cube_path"],
            "world_camera_path": env["world_camera_path"],
            "wrist_camera_path": env["wrist_camera_path"],
            "scene_config": {key: np.asarray(value).tolist() for key, value in scene_config.items()},
        }
        (self.root / "sim_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class RealSchemaRunWriter:
    def __init__(self, run_root: Path, stage: Usd.Stage):
        self.run_root = run_root
        self.stage = stage
        self.sessions: dict[int, RealSchemaSessionWriter] = {}

    def start_episode(self, episode_id: int, envs: list[dict[str, Any]]) -> None:
        self.close_episode()
        for env in envs:
            env_id = int(env["env_id"])
            session_name = f"session_ep{episode_id:03d}_env{env_id}"
            self.sessions[env_id] = RealSchemaSessionWriter(
                self.run_root,
                session_name,
                env_id=env_id,
                episode_id=episode_id,
                scene_config=env["scene_config"],
                env=env,
                stage=self.stage,
            )

    def write_sample(self, env_id: int, **kwargs: Any) -> None:
        session = self.sessions.get(env_id)
        if session is None:
            raise RuntimeError(f"No real-schema session open for env {env_id}")
        session.write_sample(**kwargs)

    def close_episode(self) -> None:
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()

    def close(self) -> None:
        self.close_episode()



def _save_npy(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)


class AsyncArrayWriter:
    def __init__(self, max_workers: int, max_pending: int):
        self.max_workers = max(1, int(max_workers))
        self.max_pending = max(1, int(max_pending))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="franka_save")
        self._semaphore = threading.Semaphore(self.max_pending)
        self._futures: set[Future[Any]] = set()
        self._errors: list[BaseException] = []
        self._lock = threading.Lock()

    def save_npy(self, path: Path, array: np.ndarray) -> None:
        self._submit(_save_npy, path, np.array(array, copy=True))

    def save_npz(self, path: Path, arrays: dict[str, np.ndarray]) -> None:
        copied_arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
        self._submit(_save_npz, path, copied_arrays)

    def close(self) -> None:
        try:
            while True:
                self._raise_if_failed()
                with self._lock:
                    futures = list(self._futures)
                if not futures:
                    break
                wait(futures)
            self._raise_if_failed()
        finally:
            self._executor.shutdown(wait=True)

    def _submit(self, fn: Any, *fn_args: Any) -> None:
        self._raise_if_failed()
        self._semaphore.acquire()
        try:
            future = self._executor.submit(fn, *fn_args)
            with self._lock:
                self._futures.add(future)
            future.add_done_callback(self._on_done)
        except Exception:
            self._semaphore.release()
            raise

    def _on_done(self, future: Future[Any]) -> None:
        try:
            exc = future.exception()
            if exc is not None:
                with self._lock:
                    self._errors.append(exc)
        finally:
            with self._lock:
                self._futures.discard(future)
            self._semaphore.release()

    def _raise_if_failed(self) -> None:
        with self._lock:
            if not self._errors:
                return
            error = self._errors[0]
        raise RuntimeError("Asynchronous data save failed") from error

def world_bbox(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if bbox.IsEmpty():
        raise RuntimeError(f"Empty bbox for prim: {prim_path}")
    return np.array(bbox.GetMin(), dtype=float), np.array(bbox.GetMax(), dtype=float)


def find_descendant(stage: Usd.Stage, root_path: str, basename: str) -> str | None:
    root_prefix = root_path.rstrip("/") + "/"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(root_prefix) and path.rsplit("/", 1)[-1] == basename:
            return path
    return None


def find_descendant_any(stage: Usd.Stage, root_path: str, basenames: list[str]) -> str | None:
    for basename in basenames:
        path = find_descendant(stage, root_path, basename)
        if path is not None:
            return path
    return None


def find_camera(stage: Usd.Stage, root_path: str, preferred_names: list[str]) -> str | None:
    root_prefix = root_path.rstrip("/") + "/"
    camera_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Camera) and str(prim.GetPath()).startswith(root_prefix)
    ]
    for preferred_name in preferred_names:
        for path in camera_paths:
            if preferred_name in path.lower():
                return path
    return None


def extract_mask(frame: dict[str, Any], transparent_label: str, debug: bool = False) -> np.ndarray | None:
    """Extract semantic segmentation mask for transparent objects.

    Returns None if:
    - No semantic_segmentation data in frame
    - transparent_label not found in idToLabels (object may be fully occluded)
    """
    seg_data = frame.get("semantic_segmentation")
    if not isinstance(seg_data, dict) or "data" not in seg_data:
        if debug:
            print(f"  [DEBUG] No semantic_segmentation data in frame")
        return None
    seg_map = np.asarray(seg_data["data"])
    transparent_ids: set[int] = set()
    id_to_labels = seg_data.get("info", {}).get("idToLabels", {})

    if debug:
        print(f"  [DEBUG] idToLabels: {id_to_labels}")
        print(f"  [DEBUG] seg_map unique values: {np.unique(seg_map)}")
        print(f"  [DEBUG] seg_map shape: {seg_map.shape}")

    for id_str, label_dict in id_to_labels.items():
        if isinstance(label_dict, dict) and label_dict.get("class") == transparent_label:
            transparent_ids.add(int(id_str))
            if debug:
                print(f"  [DEBUG] Found transparent label: id={id_str}, label={label_dict}")

    if transparent_ids:
        mask = np.zeros(seg_map.shape, dtype=np.uint16)
        for transparent_id in transparent_ids:
            mask[seg_map == transparent_id] = 1
        if debug:
            print(f"  [DEBUG] Created mask with {np.sum(mask > 0)} non-zero pixels")
        return mask

    # Object fully occluded or not visible - return None instead of invalid fallback mask
    if debug:
        print(f"  [DEBUG] No transparent_ids found in idToLabels - object likely occluded, returning None")
    return None


def bbox_distance(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    gap = np.maximum(np.maximum(min_a - max_b, min_b - max_a), 0.0)
    return float(np.linalg.norm(gap))


class FrankaMultiEnvCollector:
    def __init__(self, world: World):
        self.world = world
        self.stage = world.stage
        self.rng = np.random.default_rng(args.seed)
        self.env_origins = np.zeros((args.num_envs, 3), dtype=float)
        self.envs: list[dict[str, Any]] = []
        self.controllers: list[LoadedFrontBoxFrankaPickPlace] = []
        self.world_cameras: list[Camera] = []
        self.wrist_cameras: list[Camera] = []
        self.current_episode = 0
        self.current_step = 0
        self.mask_pixel_counts: list[int] = []
        self.save_writer = AsyncArrayWriter(args.save_workers, args.max_pending_saves) if args.save_data and args.save_legacy_arrays else None
        self.real_writer: RealSchemaRunWriter | None = None
        self.data_buffer = {
            "joint_positions": [],
            "joint_velocities": [],
            "ee_positions": [],
            "ee_orientations": [],
            "target_positions": [],
            "cube_positions": [],
            "env_ids": [],
            "step_ids": [],
            "episode_ids": [],
            "success": [],
            "distances": [],
            "pressure": [],
        }

    def setup_environments(self, env_usd: Path) -> None:
        num_per_row = int(np.ceil(np.sqrt(args.num_envs)))
        define_prim("/World/envs", "Xform")
        for env_id in range(args.num_envs):
            row = env_id // num_per_row
            col = env_id % num_per_row
            origin = np.array([col * args.env_spacing, row * args.env_spacing, 0.0], dtype=float)
            self.env_origins[env_id] = origin
            env_path = f"/World/envs/env_{env_id}"
            env_prim = define_prim(env_path, "Xform")
            UsdGeom.Xformable(env_prim).AddTranslateOp().Set(Gf.Vec3d(*origin))
            add_reference_to_stage(str(env_usd), env_path)

    def discover_env_prims(self) -> None:
        for env_id in range(args.num_envs):
            env_path = f"/World/envs/env_{env_id}"
            robot_path = find_descendant(self.stage, env_path, "robot")
            cube_path = find_descendant(self.stage, env_path, "Cube")
            table_path = find_descendant(self.stage, env_path, "Table")
            left_finger_path = find_descendant_any(self.stage, env_path, ["panda_leftfinger", "leftfinger"])
            right_finger_path = find_descendant_any(self.stage, env_path, ["panda_rightfinger", "rightfinger"])
            world_camera_path = find_camera(self.stage, env_path, ["world_camera", "leftobservationcamera", "left"])
            wrist_camera_path = find_camera(self.stage, env_path, ["wrist_camera", "wrist"])
            if not all([robot_path, cube_path, table_path, left_finger_path, right_finger_path, world_camera_path, wrist_camera_path]):
                raise RuntimeError(
                    f"Failed to discover env {env_id}: robot={robot_path}, cube={cube_path}, "
                    f"table={table_path}, left_finger={left_finger_path}, right_finger={right_finger_path}, "
                    f"world_camera={world_camera_path}, wrist_camera={wrist_camera_path}"
                )
            self.envs.append(
                {
                    "env_id": env_id,
                    "env_path": env_path,
                    "origin": self.env_origins[env_id],
                    "robot_path": robot_path,
                    "cube_path": cube_path,
                    "table_path": table_path,
                    "left_finger_path": left_finger_path,
                    "right_finger_path": right_finger_path,
                    "world_camera_path": world_camera_path,
                    "wrist_camera_path": wrist_camera_path,
                    "box_bottom_path": f"{env_path}/Box/bottom",
                    "box_left_path": f"{env_path}/Box/left",
                    "box_right_path": f"{env_path}/Box/right",
                    "box_front_path": f"{env_path}/Box/front",
                    "box_back_path": f"{env_path}/Box/back",
                }
            )

    def setup_semantic_labels(self) -> None:
        for env in self.envs:
            prim = self.stage.GetPrimAtPath(env["cube_path"])
            if prim.IsValid():
                add_labels(prim, labels=[args.transparent_label], instance_name="class")

    def initialize_runtime_objects(self) -> None:
        for env in self.envs:
            robot = FrankaExperimental(robot_path=env["robot_path"], create_robot=False)
            cube = RigidPrim(paths=env["cube_path"])
            controller = LoadedFrontBoxFrankaPickPlace(robot=robot, cube=cube)
            env_info = self.infer_environment(env)
            controller.cube_size = env_info["cube_size"]
            controller.box_inner_half_size = env_info["box_inner_half_size"]
            env["info"] = env_info
            env["robot"] = robot
            env["cube"] = cube
            self.controllers.append(controller)

            world_resolution = (args.world_camera_width, args.world_camera_height)
            wrist_resolution = (args.wrist_camera_width, args.wrist_camera_height)
            align_camera_aperture(self.stage, env["world_camera_path"], world_resolution)
            align_camera_aperture(self.stage, env["wrist_camera_path"], wrist_resolution)
            self.world_cameras.append(
                Camera(
                    prim_path=env["world_camera_path"],
                    name=f"world_camera_{env['env_id']}",
                    resolution=world_resolution,
                )
            )
            self.wrist_cameras.append(
                Camera(
                    prim_path=env["wrist_camera_path"],
                    name=f"wrist_camera_{env['env_id']}",
                    resolution=wrist_resolution,
                )
            )

    def infer_environment(self, env: dict[str, Any]) -> dict[str, np.ndarray | float]:
        table_min, table_max = world_bbox(self.stage, env["table_path"])
        cube_min, cube_max = world_bbox(self.stage, env["cube_path"])
        box_bottom_min, box_bottom_max = world_bbox(self.stage, env["box_bottom_path"])
        box_left_min, box_left_max = world_bbox(self.stage, env["box_left_path"])
        box_right_min, box_right_max = world_bbox(self.stage, env["box_right_path"])
        box_front_min, box_front_max = world_bbox(self.stage, env["box_front_path"])
        box_back_min, box_back_max = world_bbox(self.stage, env["box_back_path"])
        inner_min_x = box_left_max[0]
        inner_max_x = box_right_min[0]
        inner_min_y = box_back_max[1]
        inner_max_y = box_front_min[1]
        box_center_xy = np.array([(box_bottom_min[0] + box_bottom_max[0]) / 2.0, (box_bottom_min[1] + box_bottom_max[1]) / 2.0])
        return {
            "tabletop_z": float(table_max[2]),
            "cube_size": cube_max - cube_min,
            "box_center_xy": box_center_xy,
            "box_inner_half_size": np.array([(inner_max_x - inner_min_x) / 2.0, (inner_max_y - inner_min_y) / 2.0]),
            "box_wall_top": float(max(box_left_max[2], box_right_max[2], box_front_max[2], box_back_max[2])),
        }

    def initialize_cameras_and_masks(self) -> None:
        for camera in self.world_cameras:
            camera.initialize()
            camera.add_semantic_segmentation_to_frame()
            camera.add_distance_to_image_plane_to_frame()
        for camera in self.wrist_cameras:
            camera.initialize()
            camera.add_semantic_segmentation_to_frame()
            camera.add_distance_to_image_plane_to_frame()

    def configure_output(self, data_dir: Path) -> None:
        if args.save_data:
            self.real_writer = RealSchemaRunWriter(data_dir, self.stage)

    def sample_scene_config(self, env: dict[str, Any]) -> dict[str, np.ndarray]:
        info = env["info"]
        cube_size = np.asarray(info["cube_size"], dtype=float)
        tabletop_z = float(info["tabletop_z"])
        box_center_xy = np.asarray(info["box_center_xy"], dtype=float)
        inner_half = np.asarray(info["box_inner_half_size"], dtype=float)
        origin = np.asarray(env["origin"], dtype=float)
        for _ in range(100):
            cube_xy = origin[:2] + np.array(
                [
                    self.rng.uniform(args.cube_x_min, args.cube_x_max),
                    self.rng.uniform(args.cube_y_min, args.cube_y_max),
                ]
            )
            if np.linalg.norm(cube_xy - box_center_xy) >= args.min_cube_box_distance:
                break
        else:
            raise RuntimeError(f"Could not sample cube far enough from box in env {env['env_id']}")

        target_margin = max(float(np.max(cube_size[:2])) * 0.75, 0.02)
        target_half_range = min(args.target_random_half_range, max(0.0, float(np.min(inner_half) - target_margin)))
        target_xy = box_center_xy + np.array(
            [
                self.rng.uniform(-target_half_range, target_half_range),
                self.rng.uniform(-target_half_range, target_half_range),
            ]
        )
        return {
            "cube_initial_position": np.array([cube_xy[0], cube_xy[1], tabletop_z + cube_size[2] / 2.0]),
            "target_position": np.array([target_xy[0], target_xy[1], tabletop_z + args.place_release_height]),
            "place_approach_position": np.array([target_xy[0], target_xy[1], float(info["box_wall_top"]) + args.place_approach_clearance]),
            "box_center_position": np.array([box_center_xy[0], box_center_xy[1], tabletop_z]),
        }

    def reset_episode(self) -> None:
        self.current_episode += 1
        self.current_step = 0
        for env, controller in zip(self.envs, self.controllers):
            scene_config = self.sample_scene_config(env)
            env["scene_config"] = scene_config
            controller.target_position = scene_config["target_position"]
            controller.place_approach_position = scene_config["place_approach_position"]
            controller.box_center_position = scene_config["box_center_position"]
            controller.reset(scene_config["cube_initial_position"])
        for _ in range(args.settle_steps):
            self.world.step(render=not args.headless or args.render_every_step)
        if self.real_writer is not None:
            self.real_writer.start_episode(self.current_episode, self.envs)
        print(f"Episode {self.current_episode}: reset {args.num_envs} environments")

    def capture_cameras(self, step: int = 0) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any], list[Any]]:
        world_images, wrist_images, world_masks, wrist_masks, world_depths, wrist_depths = [], [], [], [], [], []
        for env_id, (world_camera, wrist_camera) in enumerate(zip(self.world_cameras, self.wrist_cameras)):
            world_frame = world_camera.get_current_frame()
            wrist_frame = wrist_camera.get_current_frame()

            world_rgba = world_frame.get("rgb")
            if world_rgba is None:
                world_rgba = world_camera.get_rgba()
            wrist_rgba = wrist_frame.get("rgb")
            if wrist_rgba is None:
                wrist_rgba = wrist_camera.get_rgba()
            world_rgba = np.asarray(world_rgba, dtype=np.uint8) if world_rgba is not None else np.asarray([])
            wrist_rgba = np.asarray(wrist_rgba, dtype=np.uint8) if wrist_rgba is not None else np.asarray([])
            world_images.append(world_rgba[:, :, :3] if world_rgba.ndim == 3 else None)
            wrist_images.append(wrist_rgba[:, :, :3] if wrist_rgba.ndim == 3 else None)

            world_depth = world_frame.get("distance_to_image_plane")
            world_depths.append(np.asarray(world_depth, dtype=np.float32).squeeze() if world_depth is not None else None)
            wrist_depth = wrist_frame.get("distance_to_image_plane")
            wrist_depths.append(np.asarray(wrist_depth, dtype=np.float32).squeeze() if wrist_depth is not None else None)

            debug_enabled = args.debug_masks and (step <= 3 or step % 10 == 0)
            if debug_enabled:
                print(f"\n[Step {step}] Capturing cameras for env {env_id}:")

            world_masks.append(extract_mask(world_frame, args.transparent_label, debug=debug_enabled))
            wrist_masks.append(extract_mask(wrist_frame, args.transparent_label, debug=debug_enabled))
        return world_images, wrist_images, world_masks, wrist_masks, world_depths, wrist_depths

    def cube_is_inside_box(self, controller: LoadedFrontBoxFrankaPickPlace) -> bool:
        cube_position, _ = controller.cube.get_world_poses()
        cube_xy = as_array(cube_position).reshape(-1)[:2]
        box_center = np.asarray(controller.box_center_position, dtype=float)
        allowed_half = np.asarray(controller.box_inner_half_size, dtype=float) - np.max(controller.cube_size[:2]) / 2.0
        return bool(np.all(np.abs(cube_xy - box_center[:2]) <= allowed_half))

    def gripper_contacts_cube(self, env: dict[str, Any]) -> bool:
        cube_min, cube_max = world_bbox(self.stage, env["cube_path"])
        for finger_path in [env["left_finger_path"], env["right_finger_path"]]:
            finger_min, finger_max = world_bbox(self.stage, finger_path)
            if bbox_distance(finger_min, finger_max, cube_min, cube_max) <= 0.005:
                return True
        return False

    def gripper_position_proxy(self, env: dict[str, Any]) -> float:
        try:
            left_min, left_max = world_bbox(self.stage, env["left_finger_path"])
            right_min, right_max = world_bbox(self.stage, env["right_finger_path"])
        except Exception:
            return 0.0
        left_center = (left_min + left_max) / 2.0
        right_center = (right_min + right_max) / 2.0
        opening_m = float(np.linalg.norm(left_center - right_center))
        return float(np.clip(opening_m * 10_000.0, 0.0, 1000.0))

    def step(self, data_dir: Path) -> None:
        self.current_step += 1
        save_images = args.save_data and args.save_image_interval > 0 and self.current_step % args.save_image_interval == 0
        if save_images:
            world_images, wrist_images, world_masks, wrist_masks, world_depths, wrist_depths = self.capture_cameras(self.current_step)
        else:
            world_images = wrist_images = world_masks = wrist_masks = world_depths = wrist_depths = None

        for env_id, (env, controller) in enumerate(zip(self.envs, self.controllers)):
            if not controller.is_done():
                controller.forward(args.ik_method)

            dof_positions, ee_position, ee_orientation = controller.robot.get_current_state()
            try:
                joint_velocities = controller.robot.get_dof_velocities().numpy()
            except Exception:
                joint_velocities = np.zeros_like(dof_positions)
            cube_position, _ = controller.cube.get_world_poses()
            scene_config = env["scene_config"]
            distance = float(np.linalg.norm(as_array(ee_position).reshape(-1)[:3] - scene_config["target_position"]))
            success = controller.is_done() and self.cube_is_inside_box(controller)

            if args.save_data and save_images:
                contact = self.gripper_contacts_cube(env) or 2 <= controller._event <= 6
                if self.real_writer is not None:
                    self.real_writer.write_sample(
                        env_id,
                        timestamp_us=monotonic_us(),
                        world_rgb=world_images[env_id] if world_images else None,
                        world_depth_m=world_depths[env_id] if world_depths else None,
                        wrist_rgb=wrist_images[env_id] if wrist_images else None,
                        wrist_depth_m=wrist_depths[env_id] if wrist_depths else None,
                        joint_positions=as_array(dof_positions).reshape(-1),
                        ee_position=ee_position,
                        ee_orientation=ee_orientation,
                        gripper_pos=self.gripper_position_proxy(env),
                        contact=contact,
                    )

                if args.save_legacy_arrays:
                    pressure = 1 if contact else 0
                    self.data_buffer["joint_positions"].append(as_array(dof_positions).reshape(-1))
                    self.data_buffer["joint_velocities"].append(as_array(joint_velocities).reshape(-1))
                    self.data_buffer["ee_positions"].append(as_array(ee_position).reshape(-1)[:3])
                    self.data_buffer["ee_orientations"].append(as_array(ee_orientation).reshape(-1)[:4])
                    self.data_buffer["target_positions"].append(scene_config["target_position"])
                    self.data_buffer["cube_positions"].append(as_array(cube_position).reshape(-1)[:3])
                    self.data_buffer["env_ids"].append(env_id)
                    self.data_buffer["step_ids"].append(self.current_step)
                    self.data_buffer["episode_ids"].append(self.current_episode)
                    self.data_buffer["success"].append(success)
                    self.data_buffer["distances"].append(distance)
                    self.data_buffer["pressure"].append(pressure)

                stem = f"ep{self.current_episode:03d}_env{env_id}_step{self.current_step:04d}"
                if args.save_legacy_arrays and world_images and world_images[env_id] is not None:
                    assert self.save_writer is not None
                    self.save_writer.save_npy(data_dir / "world_camera" / f"{stem}.npy", world_images[env_id])
                if args.save_legacy_arrays and wrist_images and wrist_images[env_id] is not None:
                    assert self.save_writer is not None
                    self.save_writer.save_npy(data_dir / "wrist_camera" / f"{stem}.npy", wrist_images[env_id])
                if args.save_legacy_arrays and world_masks and world_masks[env_id] is not None:
                    assert self.save_writer is not None
                    self.save_writer.save_npy(data_dir / "world_camera_mask" / f"{stem}.npy", world_masks[env_id])
                    self.mask_pixel_counts.append(int(np.sum(world_masks[env_id] > 0)))
                if args.save_legacy_arrays and wrist_masks and wrist_masks[env_id] is not None:
                    assert self.save_writer is not None
                    self.save_writer.save_npy(data_dir / "wrist_camera_mask" / f"{stem}.npy", wrist_masks[env_id])
                if args.save_legacy_arrays and wrist_depths and wrist_depths[env_id] is not None:
                    assert self.save_writer is not None
                    self.save_writer.save_npy(data_dir / "wrist_camera_depth" / f"{stem}.npy", wrist_depths[env_id])

    def save_episode_data(self, data_dir: Path) -> None:
        if self.real_writer is not None:
            self.real_writer.close_episode()
        if not args.save_data or not args.save_legacy_arrays or not self.data_buffer["joint_positions"]:
            return
        episode_arrays = {
            "joint_positions": np.asarray(self.data_buffer["joint_positions"]),
            "joint_velocities": np.asarray(self.data_buffer["joint_velocities"]),
            "ee_positions": np.asarray(self.data_buffer["ee_positions"]),
            "ee_orientations": np.asarray(self.data_buffer["ee_orientations"]),
            "target_positions": np.asarray(self.data_buffer["target_positions"]),
            "cube_positions": np.asarray(self.data_buffer["cube_positions"]),
            "env_ids": np.asarray(self.data_buffer["env_ids"]),
            "step_ids": np.asarray(self.data_buffer["step_ids"]),
            "episode_ids": np.asarray(self.data_buffer["episode_ids"]),
            "success": np.asarray(self.data_buffer["success"]),
            "distances": np.asarray(self.data_buffer["distances"]),
            "pressure": np.asarray(self.data_buffer["pressure"], dtype=np.uint8),
        }
        assert self.save_writer is not None
        self.save_writer.save_npz(data_dir / f"episode_{self.current_episode:03d}.npz", episode_arrays)
        success_rate = float(np.mean(episode_arrays["success"].astype(bool))) * 100.0
        avg_mask_pixels = float(np.mean(self.mask_pixel_counts)) if self.mask_pixel_counts else 0.0
        print(
            f"Queued episode {self.current_episode:03d}: {len(self.data_buffer['joint_positions'])} samples, "
            f"success_samples={success_rate:.1f}%, avg_world_mask_pixels={avg_mask_pixels:.1f}"
        )
        for key in self.data_buffer:
            self.data_buffer[key] = []
        self.mask_pixel_counts.clear()

    def wait_for_saves(self) -> None:
        if self.real_writer is not None:
            self.real_writer.close()
        if self.save_writer is not None:
            self.save_writer.close()

    def all_done(self) -> bool:
        return all(controller.is_done() for controller in self.controllers)


def prepare_data_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"franka_env_{timestamp}"
    data_dir = (args.output_dir / run_name).resolve()
    if args.save_data:
        data_dir.mkdir(parents=True, exist_ok=True)
        if args.save_legacy_arrays:
            for subdir in ["world_camera", "wrist_camera", "world_camera_mask", "wrist_camera_mask", "wrist_camera_depth"]:
                (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    return data_dir


def write_config(data_dir: Path) -> None:
    if not args.save_data:
        return
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["env_usd"] = str(args.env_usd.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    config["data_format"] = {
        "primary": "Real-data aligned sessions under session_ep###_env#/ with world_camera/{rgb,depth}, wrist_camera/{rgb,depth}, frames.csv, camera_metadata.json, robot_state/*.csv, pressure/pressure.csv",
        "legacy_arrays": "Only written when --save-legacy-arrays is set: episode_###.npz and .npy camera/mask/depth arrays.",
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "collection_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    env_usd = args.env_usd.resolve()
    if not env_usd.exists():
        raise FileNotFoundError(f"Environment USD not found: {env_usd}")
    data_dir = prepare_data_dir()
    write_config(data_dir)
    print("=" * 72)
    print("Franka front-box multi-env camera data collection")
    print(f"  env_usd: {env_usd}")
    print(f"  num_envs: {args.num_envs}, episodes: {args.episodes}, steps_per_episode: {args.steps_per_episode}")
    print(f"  data_dir: {data_dir}")
    print("=" * 72)

    if "cuda" in args.device:
        SimulationManager.set_backend("torch")
    SimulationManager.set_physics_sim_device(args.device)
    world = World(stage_units_in_meters=1.0)
    collector = FrankaMultiEnvCollector(world)
    collector.setup_environments(env_usd)
    for _ in range(5):
        simulation_app.update()
    collector.discover_env_prims()
    collector.setup_semantic_labels()
    collector.initialize_runtime_objects()

    world.reset()
    world.play()
    collector.initialize_cameras_and_masks()
    for _ in range(max(0, args.camera_warmup_render_steps)):
        world.step(render=True)
    collector.configure_output(data_dir)

    try:
        try:
            for _ in range(args.episodes):
                collector.reset_episode()
                for _step_index in range(args.steps_per_episode):
                    world.step(render=should_render_step(collector.current_step + 1))
                    if world.is_playing():
                        collector.step(data_dir)
                    if collector.all_done():
                        break
                    if collector.current_step % 50 == 0:
                        done_count = sum(controller.is_done() for controller in collector.controllers)
                        print(f"Episode {collector.current_episode} step {collector.current_step}: done {done_count}/{args.num_envs}")
                collector.save_episode_data(data_dir)
        except KeyboardInterrupt:
            print("Interrupted by user")
            collector.save_episode_data(data_dir)
    finally:
        collector.wait_for_saves()

    print("Collection completed")
    print(f"Data saved to: {data_dir}")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
