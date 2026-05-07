"""Collect synchronized data from DJI, RealSense RGB, robotic arm state, and pressure UDP data.

Press SPACE to start/stop individual recording sessions. Each session creates a
timestamped folder containing:
    - `dji/`             : DJI Osmo Action RGB frames
    - `realsense_rgb/`   : Intel RealSense RGB frames
    - `robot_state/`     : JSON snapshots of the arm pose/state per frame
    - `pressure/`        : CSV pressure samples (`left/right`)

Use Q or ESC to exit at any time.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


DEFAULT_DJI_INDEX = 1
DEFAULT_ARM_HOST = "192.168.31.92"
DEFAULT_ARM_PORT = 8080
MAX_PREVIEW_WIDTH = 1920

DEFAULT_PRESSURE_LOCAL_PORT = 4321
DEFAULT_PRESSURE_REMOTE_IP = "192.168.31.164"
DEFAULT_PRESSURE_REMOTE_PORT = 2222
PRESSURE_PACKET_FORMAT = "<Q64h"
PRESSURE_PACKET_SIZE = struct.calcsize(PRESSURE_PACKET_FORMAT)
PRESSURE_BUFFER_SIZE = PRESSURE_PACKET_SIZE
PRESSURE_BATCH_SIZE = 100
PRESSURE_FLUSH_INTERVAL_S = 0.1


class DJICamera:
    def __init__(self, index: int, width: int, height: int):
        self.index = index
        self.width = width
        self.height = height
        self.capture: Optional[cv2.VideoCapture] = None
        self.available = False
        self.last_warn_time = 0.0

    def start(self) -> None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.available = False
            self.capture = None
            print(
                f"Warning: Unable to open DJI camera at index {self.index}. "
                "Using zero-filled frames."
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture = cap
        self.available = True

    def read(self) -> np.ndarray:
        if self.capture is None:
            return self._zero_frame()

        ok, frame = self.capture.read()
        if ok and frame is not None:
            return frame

        now = time.time()
        if (now - self.last_warn_time) >= 1.0:
            print("Warning: DJI frame read failed. Using zero-filled frame.")
            self.last_warn_time = now
        return self._zero_frame()

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.available = False

    def _zero_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class RealSenseRGB:
    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline: Optional[object] = None
        self.available = False
        self.last_warn_time = 0.0

    def start(self) -> None:
        if rs is None:
            self.pipeline = None
            self.available = False
            print("Warning: pyrealsense2 not installed. Using zero-filled RealSense frames.")
            return

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        try:
            pipeline.start(config)
        except RuntimeError as exc:
            self.pipeline = None
            self.available = False
            print(
                "Warning: RealSense pipeline could not start. "
                "Using zero-filled frames. Details: "
                f"{exc}"
            )
            return
        self.pipeline = pipeline
        self.available = True

    def read(self) -> np.ndarray:
        if self.pipeline is None:
            return self._zero_frame()

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Missing RealSense color frame.")
            return np.asanyarray(color_frame.get_data())
        except Exception as exc:
            now = time.time()
            if (now - self.last_warn_time) >= 1.0:
                print(f"Warning: RealSense frame read failed. Using zero-filled frame. Details: {exc}")
                self.last_warn_time = now
            return self._zero_frame()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.available = False

    def _zero_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


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


class PressureCollector:
    def __init__(
        self,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        timeout_s: float = 1.0,
        batch_size: int = PRESSURE_BATCH_SIZE,
        flush_interval_s: float = PRESSURE_FLUSH_INTERVAL_S,
    ):
        self.local_port = local_port
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.timeout_s = timeout_s
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s

        self.sock: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False

        self.recording = False
        self.csv_file = None
        self.csv_writer: Optional[csv.writer] = None
        self.row_buffer: list[list[int]] = []

        self.latest_timestamp_us: Optional[int] = None
        self.latest_values: list[int] = [0] * 64

        self.last_flush_time = time.time()

        self.lock = threading.Lock()

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.local_port))
        sock.settimeout(self.timeout_s)
        sock.sendto(b"HELLO", (self.remote_ip, self.remote_port))

        self.sock = sock
        self.running = True
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

        print(f"Pressure UDP listening on {self.local_port}")
        print(f"Pressure handshake sent to {self.remote_ip}:{self.remote_port}")

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

        self.stop_session()

        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def start_session(self, session_root: Path) -> None:
        pressure_dir = session_root / "pressure"
        pressure_dir.mkdir(parents=True, exist_ok=True)
        csv_path = pressure_dir / "pressure.csv"

        with self.lock:
            self._stop_session_locked()
            self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            headers = ["timestamp_us"] + [f"CH{i+1}" for i in range(64)]
            self.csv_writer.writerow(headers)
            self.row_buffer = []
            self.last_flush_time = time.time()
            self.recording = True

        print(f"Pressure session file: {csv_path}")

    def stop_session(self) -> None:
        with self.lock:
            self._stop_session_locked()

    def _stop_session_locked(self) -> None:
        self._flush_locked(force=True)
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        self.csv_writer = None
        self.recording = False

    def get_latest_values(self) -> list[int]:
        with self.lock:
            return list(self.latest_values)

    def _recv_loop(self) -> None:
        while self.running:
            if self.sock is None:
                break

            try:
                data, addr = self.sock.recvfrom(PRESSURE_BUFFER_SIZE)
            except socket.timeout:
                with self.lock:
                    self._flush_locked(force=False)
                continue
            except OSError:
                break

            if len(data) < PRESSURE_PACKET_SIZE:
                print(f"[Pressure {addr}] Packet too small: {len(data)} bytes")
                continue

            timestamp_us, *values = struct.unpack(PRESSURE_PACKET_FORMAT, data)

            now = time.time()
            with self.lock:
                self.latest_timestamp_us = timestamp_us
                self.latest_values = list(values)

                if self.recording and self.csv_writer is not None:
                    row = [timestamp_us] + self.latest_values
                    self.row_buffer.append(row)
                    self._flush_locked(force=False)

    def _flush_locked(self, force: bool) -> None:
        if not self.recording or self.csv_writer is None or self.csv_file is None:
            return
        if not self.row_buffer:
            return

        now = time.time()
        should_flush = force or len(self.row_buffer) >= self.batch_size or (
            now - self.last_flush_time
        ) >= self.flush_interval_s

        if not should_flush:
            return

        self.csv_writer.writerows(self.row_buffer)
        self.csv_file.flush()
        self.row_buffer.clear()
        self.last_flush_time = now


@dataclass
class SessionPaths:
    root: Path
    dji: Path
    realsense: Path
    robot_state: Path
    pressure: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DJI + RealSense + robot arm + pressure recorder.")
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

    parser.add_argument("--pressure-local-port", type=int, default=DEFAULT_PRESSURE_LOCAL_PORT)
    parser.add_argument("--pressure-remote-ip", default=DEFAULT_PRESSURE_REMOTE_IP)
    parser.add_argument("--pressure-remote-port", type=int, default=DEFAULT_PRESSURE_REMOTE_PORT)

    return parser


def create_session_paths(base: Path, prefix: str) -> SessionPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = base / f"{prefix}_{timestamp}"
    dji_dir = session_root / "dji"
    rs_dir = session_root / "realsense_rgb"
    robot_dir = session_root / "robot_state"
    pressure_dir = session_root / "pressure"

    for directory in (dji_dir, rs_dir, robot_dir, pressure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Recording session created: {session_root}")
    return SessionPaths(root=session_root, dji=dji_dir, realsense=rs_dir, robot_state=robot_dir, pressure=pressure_dir)


def draw_pressure_dashboard(canvas: np.ndarray, x: int, y: int, frame_w: int, values: list[int]) -> None:
    if not values or len(values) < 64:
        return

    LEFT_CHANNEL = 19
    RIGHT_CHANNEL = 18
    LEFT_MATRIX_CHANNELS = [[1, 16, 15], [14, 13, 12], [11, 10, 9]]
    RIGHT_MATRIX_CHANNELS = [[17, 32, 31], [30, 29, 28], [27, 26, 25]]

    def get_val(ch: int) -> int:
        return values[ch - 1] if 0 <= ch - 1 < len(values) else 0

    left_val = get_val(LEFT_CHANNEL)
    right_val = get_val(RIGHT_CHANNEL)
    left_mat = [[get_val(ch) for ch in row] for row in LEFT_MATRIX_CHANNELS]
    right_mat = [[get_val(ch) for ch in row] for row in RIGHT_MATRIX_CHANNELS]

    all_vals = [abs(left_val), abs(right_val)] + [abs(v) for row in left_mat for v in row] + [abs(v) for row in right_mat for v in row]
    peak = float(max(1, max(all_vals)))

    def get_color(val: int) -> tuple[int, int, int]:
        ratio = min(1.0, abs(val) / peak)
        b = int(40 + (255 - 40) * ratio)
        g = int(40 + (100 - 40) * ratio)
        r = int(40 + (50 - 40) * ratio)
        return (b, g, r)

    def draw_text_centered(img, text, cx, cy, font_scale=0.6, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        th = 2 if font_scale > 0.5 else 1
        sz, _ = cv2.getTextSize(text, font, font_scale, th)
        cv2.putText(img, text, (int(cx - sz[0] / 2), int(cy + sz[1] / 2)), font, font_scale, color, th, cv2.LINE_AA)

    col_w = frame_w // 3

    # ==========================
    # Column 1: Stacked L/R Bars
    # ==========================
    def draw_bar(bx: int, by: int, max_w: int, label: str, value: int) -> None:
        bar_h = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        th = 2
        label_size, _ = cv2.getTextSize(label, font, font_scale, th)
        label_w = label_size[0]
        label_h = label_size[1]
        
        baseline_y = by + (bar_h + label_h) // 2 - 2
        cv2.putText(canvas, label, (bx + 10, baseline_y), font, font_scale, (240, 240, 240), th, cv2.LINE_AA)

        b_left = bx + 10 + label_w + 15
        b_right = bx + max_w - 20
        b_w = max(20, b_right - b_left)

        ratio = min(1.0, abs(value) / peak)
        fill_w = int(b_w * ratio)
        if fill_w > 0:
            cv2.rectangle(canvas, (b_left, by), (b_left + fill_w, by + bar_h), get_color(value), -1)
        cv2.rectangle(canvas, (b_left, by), (b_right, by + bar_h), (100, 100, 100), 2)
        draw_text_centered(canvas, f"{value:d}", b_left + b_w // 2, by + bar_h // 2, font_scale=0.7)

    bar_start_y = y + 60
    draw_bar(x, bar_start_y, col_w, "Left", left_val)
    draw_bar(x, bar_start_y + 80, col_w, "Right", right_val)

    # ==========================
    # Column 2 & 3: Matrices
    # ==========================
    def draw_matrix(mx: int, label: str, matrix: list[list[int]]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        th = 2
        label_size, _ = cv2.getTextSize(label, font, font_scale, th)
        label_w = label_size[0]
        label_h = label_size[1]
        
        matrix_area_w = col_w - 30 - label_w
        cell_gap = 8
        cell_s = min(86, (matrix_area_w - cell_gap * 2) // 3)
        matrix_side = cell_s * 3 + cell_gap * 2
        
        matrix_y = y + 20
        label_y = matrix_y + matrix_side // 2 + label_h // 2
        
        cv2.putText(canvas, label, (mx + 10, label_y), font, font_scale, (240, 240, 240), th, cv2.LINE_AA)
        
        matrix_x = mx + 20 + label_w

        for row_i in range(3):
            for col_i in range(3):
                cx = matrix_x + col_i * cell_s
                cy = matrix_y + row_i * cell_s
                val = matrix[row_i][col_i]
                inner = cell_s - cell_gap
                cv2.rectangle(canvas, (cx, cy), (cx + inner, cy + inner), get_color(val), -1)
                cv2.rectangle(canvas, (cx, cy), (cx + inner, cy + inner), (100, 100, 100), 1)
                draw_text_centered(canvas, str(val), cx + inner // 2, cy + inner // 2, font_scale=0.55)

    draw_matrix(x + col_w, "Left Matrix", left_mat)
    draw_matrix(x + col_w * 2, "Right Matrix", right_mat)


def compose_preview(
    dji_frame: np.ndarray,
    rs_frame: np.ndarray,
    status_text: str,
    joint_text: str,
    pose_text: str,
    pressure_values: list[int],
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

    info_height = 420
    info_panel = np.zeros((info_height, top_width, 3), dtype=np.uint8)
    
    # Left side details
    cv2.putText(info_panel, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(info_panel, joint_text, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(info_panel, pose_text, (20, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)

    # Dashboard placement inside info_panel
    # Split the info panel width to layout the pressure views
    # Assuming large screen layout. We give the dashboard right half or whole width under text.
    draw_pressure_dashboard(info_panel, 20, 140, top_width - 40, pressure_values)

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
    pressure = PressureCollector(
        local_port=args.pressure_local_port,
        remote_ip=args.pressure_remote_ip,
        remote_port=args.pressure_remote_port,
    )

    print(f"Opening DJI camera at index {args.dji_index}...")
    dji.start()
    print("Starting RealSense RGB pipeline...")
    rs_camera.start()
    print(f"Connecting robot arm at {args.arm_host}:{args.arm_port}...")
    robot.connect()
    print("Starting pressure collector...")
    pressure.start()

    recording = False
    session_paths: Optional[SessionPaths] = None
    frame_id = 0
    latest_state_text = "Joint: N/A"
    latest_pose_text = "Pose: N/A"

    print("Press SPACE to start/stop recording sessions, Q/ESC to exit.")
    cv2.namedWindow("DJI + RealSense", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DJI + RealSense", 1920, 1080)

    try:
        while True:
            dji_frame = dji.read()
            rs_frame = rs_camera.read()

            try:
                state = robot.read_state()
            except RuntimeError as err:
                state = {"code": -1, "error": str(err)}

            payload = state.get("data") if isinstance(state, dict) else None
            joints = payload.get("joint") if isinstance(payload, dict) else None
            pose = payload.get("pose") if isinstance(payload, dict) else None
            latest_state_text = "Joint: " + (" | ".join(f"{val:.1f}" for val in joints) if joints else "N/A")
            latest_pose_text = "Pose: " + (" | ".join(f"{val:.3f}" for val in pose) if pose else "N/A")
            latest_pressure_values = pressure.get_latest_values()

            status = "REC" if recording else "IDLE"
            preview = compose_preview(
                dji_frame,
                rs_frame,
                f"Status: {status} | Frames: {frame_id if recording else 0}",
                latest_state_text,
                latest_pose_text,
                latest_pressure_values,
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
                    pressure.start_session(session_paths.root)
                    frame_id = 0
                else:
                    pressure.stop_session()
                    session_paths = None
                    frame_id = 0
                    print("Recording paused. Press SPACE to start a new session.")
    finally:
        dji.stop()
        rs_camera.stop()
        robot.disconnect()
        pressure.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    main()
