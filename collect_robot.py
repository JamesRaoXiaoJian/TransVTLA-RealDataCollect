"""Collect synchronized data from DJI, RealSense RGB, and the robotic arm state.

Press SPACE to start/stop individual recording sessions. Each session creates a
timestamped folder containing:
    - `dji/`             : DJI Osmo Action RGB frames
    - `realsense_rgb/`   : Intel RealSense RGB frames
    - `robot_state/`     : JSON snapshots of the arm pose/state per frame

Use Q or ESC to exit at any time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyrealsense2 is required. Install it via 'pip install pyrealsense2'."
    ) from exc


DEFAULT_DJI_INDEX = 1
DEFAULT_ARM_HOST = "192.168.31.92"
DEFAULT_ARM_PORT = 8080
MAX_PREVIEW_WIDTH = 1920


class DJICamera:
    def __init__(self, index: int, width: int, height: int):
        self.index = index
        self.width = width
        self.height = height
        self.capture: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open DJI camera at index {self.index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture = cap

    def read(self) -> Optional[np.ndarray]:
        if self.capture is None:
            raise RuntimeError("DJI camera not started.")
        ok, frame = self.capture.read()
        return frame if ok else None

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class RealSenseRGB:
    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline: Optional[rs.pipeline] = None

    def start(self) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        try:
            pipeline.start(config)
        except RuntimeError as exc:
            raise RuntimeError(
                "RealSense pipeline could not start. Ensure the camera is connected, "
                "not used by another process, and the requested resolution/fps is "
                "supported."
            ) from exc
        self.pipeline = pipeline

    def read(self) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("RealSense pipeline not started.")
        frames = self.pipeline.wait_for_frames(timeout_ms=500)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Missing RealSense color frame.")
        return np.asanyarray(color_frame.get_data())

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


class RobotArmMonitor:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = None

    def connect(self) -> None:
        self.handle = self.robot.rm_create_robot_arm(self.host, self.port)
        if self.handle is None:
            raise RuntimeError("Failed to create robot arm handle.")
        print(f"机械臂ID： {self.handle.id}")

    def read_state(self) -> dict:
        if self.handle is None:
            raise RuntimeError("Robot arm not connected.")
        status = self.robot.rm_get_current_arm_state()
        if not isinstance(status, tuple) or len(status) != 2:
            return {"code": -1, "data": None}
        code, payload = status
        return {"code": code, "data": payload}

    def disconnect(self) -> None:
        if self.handle is not None:
            self.robot.rm_delete_robot_arm()
            self.handle = None


@dataclass
class SessionPaths:
    root: Path
    dji: Path
    realsense: Path
    robot_state: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DJI + RealSense + robot arm recorder.")
    parser.add_argument("--dji-index", type=int, default=DEFAULT_DJI_INDEX, help="OpenCV index for DJI camera.")
    parser.add_argument("--width", type=int, default=1280, help="Frame width for both streams.")
    parser.add_argument("--height", type=int, default=720, help="Frame height for both streams.")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate for RealSense color stream.")
    parser.add_argument("--output", type=Path, default=Path("sessions"), help="Base directory for recordings.")
    parser.add_argument(
        "--session-prefix",
        default="session",
        help="Folder prefix for each recording session under the base output directory.",
    )
    parser.add_argument("--arm-host", default=DEFAULT_ARM_HOST, help="Robot arm controller IP address.")
    parser.add_argument("--arm-port", type=int, default=DEFAULT_ARM_PORT, help="Robot arm controller port.")
    return parser


def create_session_paths(base: Path, prefix: str) -> SessionPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = base / f"{prefix}_{timestamp}"
    dji_dir = session_root / "dji"
    rs_dir = session_root / "realsense_rgb"
    robot_dir = session_root / "robot_state"

    for directory in (dji_dir, rs_dir, robot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Recording session created: {session_root}")
    return SessionPaths(root=session_root, dji=dji_dir, realsense=rs_dir, robot_state=robot_dir)


def compose_preview(
    dji_frame: np.ndarray,
    rs_frame: np.ndarray,
    status_text: str,
    joint_text: str,
    pose_text: str,
) -> np.ndarray:
    target_height = max(dji_frame.shape[0], rs_frame.shape[0])

    def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
        if frame.shape[0] == height:
            return frame
        ratio = height / frame.shape[0]
        new_width = int(frame.shape[1] * ratio)
        return cv2.resize(frame, (new_width, height))

    dji_resized = resize_to_height(dji_frame, target_height)
    rs_resized = resize_to_height(rs_frame, target_height)

    gap = 30
    top_width = dji_resized.shape[1] + rs_resized.shape[1] + gap
    top_canvas = np.zeros((target_height, top_width, 3), dtype=np.uint8)
    top_canvas[:, : dji_resized.shape[1]] = dji_resized
    top_canvas[:, dji_resized.shape[1] + gap :] = rs_resized

    info_height = 130
    info_panel = np.zeros((info_height, top_width, 3), dtype=np.uint8)
    cv2.putText(info_panel, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(info_panel, joint_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(info_panel, pose_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2, cv2.LINE_AA)

    canvas = np.vstack((top_canvas, info_panel))
    if canvas.shape[1] > MAX_PREVIEW_WIDTH:
        scale = MAX_PREVIEW_WIDTH / canvas.shape[1]
        new_size = (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale))
        canvas = cv2.resize(canvas, new_size)
    return canvas


def save_robot_state(directory: Path, frame_stem: str, state: dict) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "frame": frame_stem,
        "state": state,
    }
    state_path = directory / f"{frame_stem}.json"
    with state_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    dji = DJICamera(index=args.dji_index, width=args.width, height=args.height)
    rs_camera = RealSenseRGB(width=args.width, height=args.height, fps=args.fps)
    robot = RobotArmMonitor(host=args.arm_host, port=args.arm_port)

    print(f"Opening DJI camera at index {args.dji_index}...")
    dji.start()
    print("Starting RealSense RGB pipeline...")
    rs_camera.start()
    print(f"Connecting robot arm at {args.arm_host}:{args.arm_port}...")
    robot.connect()

    recording = False
    session_paths: Optional[SessionPaths] = None
    frame_id = 0
    latest_state_text = "Joint: N/A"
    latest_pose_text = "Pose: N/A"

    print("Press SPACE to start/stop recording sessions, Q/ESC to exit.")
    cv2.namedWindow("DJI + RealSense", cv2.WINDOW_NORMAL)

    try:
        while True:
            dji_frame = dji.read()
            if dji_frame is None:
                print("Warning: empty frame from DJI camera.")
                continue

            try:
                rs_frame = rs_camera.read()
            except RuntimeError as err:
                print(f"Warning: RealSense frame skipped ({err}).")
                continue

            try:
                state = robot.read_state()
            except RuntimeError as err:
                state = {"code": -1, "error": str(err)}

            payload = state.get("data") if isinstance(state, dict) else None
            joints = payload.get("joint") if isinstance(payload, dict) else None
            pose = payload.get("pose") if isinstance(payload, dict) else None
            latest_state_text = "Joint: " + (" | ".join(f"{val:.1f}" for val in joints) if joints else "N/A")
            latest_pose_text = "Pose: " + (" | ".join(f"{val:.3f}" for val in pose) if pose else "N/A")

            status = "REC" if recording else "IDLE"
            preview = compose_preview(
                dji_frame,
                rs_frame,
                f"Status: {status} | Frames: {frame_id if recording else 0}",
                latest_state_text,
                latest_pose_text,
            )
            cv2.imshow("DJI + RealSense", preview)

            if recording and session_paths is not None:
                frame_index = frame_id + 1
                frame_stem = f"{frame_index:04d}"
                image_name = f"{frame_stem}.jpg"

                cv2.imwrite(str(session_paths.dji / image_name), dji_frame)
                cv2.imwrite(str(session_paths.realsense / image_name), rs_frame)

                save_robot_state(session_paths.robot_state, frame_stem, state)

                frame_id += 1

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Exiting capture loop.")
                break
            if key == ord(" "):
                recording = not recording
                if recording:
                    session_paths = create_session_paths(args.output, args.session_prefix)
                    frame_id = 0
                else:
                    session_paths = None
                    frame_id = 0
                    print("Recording paused. Press SPACE to start a new session.")
    finally:
        dji.stop()
        rs_camera.stop()
        robot.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    main()
