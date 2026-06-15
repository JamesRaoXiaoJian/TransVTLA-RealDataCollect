"""Collect synchronized data from two RealSense RGB-D cameras, robot, pressure, and gripper state.

Sampling rates:
    - Visual (world RealSense + wrist RealSense): 30Hz
    - Tactile/Pressure: 200Hz
    - Robot Arm State: 100Hz
    - Gripper RM Plus State: 100Hz target

Press SPACE to start/stop individual recording sessions. Each session creates a
timestamped folder containing:
    - `world_camera/rgb/`   : world RealSense RGB frames (30Hz)
    - `world_camera/depth/` : world RealSense depth frames (30Hz, 16-bit PNG)
    - `wrist_camera/rgb/`   : wrist RealSense RGB frames (30Hz)
    - `wrist_camera/depth/` : wrist RealSense depth frames (30Hz, 16-bit PNG)
    - `robot_state/`     : CSV robot arm state + gripper state
    - `pressure/`        : CSV pressure samples at 200Hz

Use Q or ESC to exit at any time.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from timestamp_utils import get_timestamp_us
from realsense_standard import (
    CAMERA_METADATA_FILE,
    STANDARD_RS_FPS,
    STANDARD_RS_HEIGHT,
    STANDARD_RS_WIDTH,
    standard_realsense_profile,
)
from session_schema import (
    WORLD_CAMERA,
    WRIST_CAMERA,
    camera_depth_dir,
    camera_rgb_dir,
)

from collectors import (
    GripperStateCollector,
    PressureCollector,
    RealSenseRGB,
    RobotArmCollector,
)
from collectors.pressure import (
    DEFAULT_PRESSURE_LOCAL_PORT,
    DEFAULT_PRESSURE_REMOTE_IP,
    DEFAULT_PRESSURE_REMOTE_PORT,
)
from collectors.robot_arm import DEFAULT_ARM_HOST, DEFAULT_ARM_PORT

VISUAL_FPS = 30
VISUAL_INTERVAL_S = 1.0 / VISUAL_FPS
FRAME_SAVE_QUEUE_SIZE = 180

# Pressure channel mapping (from Channel Mapping.txt)
from channel_config import (
    LEFT_CHANNEL, RIGHT_CHANNEL,
    LEFT_MATRIX_CHANNELS, RIGHT_MATRIX_CHANNELS,
    INTERPOLATE_CHANNELS,
)


@dataclass
class SessionPaths:
    root: Path
    world_rgb: Path
    world_depth: Path
    wrist_rgb: Path
    wrist_depth: Path
    pressure: Path
    camera_metadata: Path


@dataclass
class FrameSaveTask:
    frame_id: int
    capture_us: int
    world_rgb: np.ndarray
    world_depth: np.ndarray
    wrist_rgb: np.ndarray
    wrist_depth: np.ndarray


class FrameSaveWorker:
    """Background image/metadata writer so preview never blocks on disk I/O."""

    def __init__(self, session_paths: SessionPaths, queue_size: int = FRAME_SAVE_QUEUE_SIZE):
        self.session_paths = session_paths
        self.queue: queue.Queue[FrameSaveTask | None] = queue.Queue(maxsize=queue_size)
        self.thread: threading.Thread | None = None
        self.running = False
        self.dropped_tasks = 0
        self.failed_writes = 0
        self.saved_frames = 0
        self.max_queue_depth = 0
        self.frames_file = None
        self.frames_writer: csv.writer | None = None

    def start(self) -> None:
        frames_path = self.session_paths.root / "frames.csv"
        self.frames_file = open(frames_path, "w", newline="", encoding="utf-8")
        self.frames_writer = csv.writer(self.frames_file)
        self.frames_writer.writerow([
            "frame_id", "capture_monotonic_us",
            "world_rgb_save_us", "world_depth_save_us",
            "wrist_rgb_save_us", "wrist_depth_save_us",
            "save_complete_us", "save_queue_depth",
        ])
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def enqueue(self, task: FrameSaveTask) -> bool:
        try:
            self.queue.put_nowait(task)
        except queue.Full:
            self.dropped_tasks += 1
            return False
        self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())
        return True

    def stop(self) -> None:
        if self.running:
            self.running = False
            self.queue.put(None)
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            self.thread = None
        if self.frames_file is not None:
            self.frames_file.flush()
            self.frames_file.close()
            self.frames_file = None
        self.frames_writer = None

    def _run(self) -> None:
        while True:
            task = self.queue.get()
            try:
                if task is None:
                    break
                self._save_task(task)
            finally:
                self.queue.task_done()

    def _save_task(self, task: FrameSaveTask) -> None:
        image_name = f"{task.frame_id:04d}.jpg"
        depth_name = f"{task.frame_id:04d}.png"
        try:
            ok = cv2.imwrite(
                str(self.session_paths.world_rgb / image_name),
                task.world_rgb,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            world_rgb_save_us = get_timestamp_us()
            ok = cv2.imwrite(str(self.session_paths.world_depth / depth_name), task.world_depth) and ok
            world_depth_save_us = get_timestamp_us()
            ok = cv2.imwrite(
                str(self.session_paths.wrist_rgb / image_name),
                task.wrist_rgb,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            ) and ok
            wrist_rgb_save_us = get_timestamp_us()
            ok = cv2.imwrite(str(self.session_paths.wrist_depth / depth_name), task.wrist_depth) and ok
            wrist_depth_save_us = get_timestamp_us()
            save_complete_us = get_timestamp_us()
            if not ok:
                self.failed_writes += 1
        except Exception:
            self.failed_writes += 1
            return

        if self.frames_writer is not None:
            self.frames_writer.writerow([
                task.frame_id, task.capture_us,
                world_rgb_save_us, world_depth_save_us,
                wrist_rgb_save_us, wrist_depth_save_us,
                save_complete_us, self.queue.qsize(),
            ])
            if task.frame_id % VISUAL_FPS == 0 and self.frames_file is not None:
                self.frames_file.flush()
        self.saved_frames += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual RealSense + robot arm + pressure + gripper recorder.")
    parser.add_argument("--world-serial", default=None, help="Serial number for the world RealSense camera.")
    parser.add_argument("--wrist-serial", default=None, help="Serial number for the wrist RealSense camera.")
    parser.add_argument("--width", type=int, default=STANDARD_RS_WIDTH, help="Standard RealSense RGB/depth width.")
    parser.add_argument("--height", type=int, default=STANDARD_RS_HEIGHT, help="Standard RealSense RGB/depth height.")
    parser.add_argument("--rs-fps", type=int, default=STANDARD_RS_FPS, help="Standard RealSense camera FPS.")
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

    parser.add_argument("--disable-gripper", action="store_true", help="Disable RM Plus gripper state logging.")

    return parser


def create_session_paths(base: Path, prefix: str) -> SessionPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = base / f"{prefix}_{timestamp}"
    world_rgb_dir = camera_rgb_dir(session_root, WORLD_CAMERA)
    world_depth_dir = camera_depth_dir(session_root, WORLD_CAMERA)
    wrist_rgb_dir = camera_rgb_dir(session_root, WRIST_CAMERA)
    wrist_depth_dir = camera_depth_dir(session_root, WRIST_CAMERA)
    pressure_dir = session_root / "pressure"

    for directory in (world_rgb_dir, world_depth_dir, wrist_rgb_dir, wrist_depth_dir, pressure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Recording session created: {session_root}")
    return SessionPaths(
        root=session_root,
        world_rgb=world_rgb_dir,
        world_depth=world_depth_dir,
        wrist_rgb=wrist_rgb_dir,
        wrist_depth=wrist_depth_dir,
        pressure=pressure_dir,
        camera_metadata=session_root / CAMERA_METADATA_FILE,
    )


# ---------------------------------------------------------------------------
# PySide6 helpers (reused from data_viwer.py pattern)
# ---------------------------------------------------------------------------


def _bgr_to_qimage(img: np.ndarray) -> QtGui.QImage:
    if img is None:
        return QtGui.QImage()
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        return qimg.copy()
    rgb = img[:, :, ::-1].copy()
    h, w, _ = rgb.shape
    bytes_per_line = 3 * w
    qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    return qimg.copy()


def _depth_to_preview_bgr(depth: np.ndarray) -> np.ndarray:
    if depth is None or depth.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    valid = depth[depth > 0]
    if valid.size:
        scale_max = float(np.percentile(valid, 95))
    else:
        scale_max = 1.0
    scale_max = max(scale_max, 1.0)
    normalized = np.clip(depth.astype(np.float32) * (255.0 / scale_max), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


class ImageLabel(QtWidgets.QLabel):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._pixmap: QtGui.QPixmap | None = None
        self.setMinimumSize(1, 1)

    def set_image(self, img: np.ndarray) -> None:
        qimg = _bgr_to_qimage(img)
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled()


# ---------------------------------------------------------------------------
# Pressure dashboard widget (pure QPainter, no cv2)
# ---------------------------------------------------------------------------


def _pressure_color(ratio: float) -> QtGui.QColor:
    ratio = min(1.0, max(0.0, ratio))
    b = int(40 + (255 - 40) * ratio)
    g = int(40 + (100 - 40) * ratio)
    r = int(40 + (50 - 40) * ratio)
    return QtGui.QColor(r, g, b)


class PressureDashboard(QtWidgets.QWidget):
    """Left: state info + L/R bar charts. Right: two 3x3 heat-matrices + colorbar."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._values: list[int] = [0] * 64
        self._state_text = ""
        self.setMinimumHeight(260)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtCore.Qt.white)
        self.setPalette(pal)

    def set_values(self, values: list[int]) -> None:
        if len(values) >= 64:
            self._values = list(values)
            # 对 INTERPOLATE_CHANNELS 中标记的异常通道，用相邻通道均值替代
            for bad_ch, adjacent_chs in INTERPOLATE_CHANNELS.items():
                if 0 <= bad_ch - 1 < len(self._values):
                    adj_vals = [self._values[c - 1] for c in adjacent_chs if 0 <= c - 1 < len(self._values)]
                    if adj_vals:
                        self._values[bad_ch - 1] = int(sum(adj_vals) / len(adj_vals))
            self.update()

    def set_state_info(self, text: str) -> None:
        self._state_text = text
        self.update()

    @staticmethod
    def _get_val(values: list[int], ch: int) -> int:
        return values[ch - 1] if 0 <= ch - 1 < len(values) else 0

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()

        vals = self._values
        left_val = self._get_val(vals, LEFT_CHANNEL)
        right_val = self._get_val(vals, RIGHT_CHANNEL)
        left_mat = [[self._get_val(vals, ch) for ch in row] for row in LEFT_MATRIX_CHANNELS]
        right_mat = [[self._get_val(vals, ch) for ch in row] for row in RIGHT_MATRIX_CHANNELS]

        all_abs = [abs(left_val), abs(right_val)]
        all_abs += [abs(v) for row in left_mat for v in row]
        all_abs += [abs(v) for row in right_mat for v in row]
        peak = float(max(1, max(all_abs)))

        font = p.font()
        font.setPointSize(10)
        p.setFont(font)
        fm = QtGui.QFontMetrics(font)

        col_w = w // 3

        # --- Matrices geometry (compute first for alignment) ---
        mat_cell_gap = 8
        mat_avail_w = col_w - 30
        mat_cell_s = min(80, (mat_avail_w - mat_cell_gap * 2) // 3)
        mat_inner = mat_cell_s - mat_cell_gap
        mat_side = mat_cell_s * 3 + mat_cell_gap * 2
        matrix_y = 10

        # --- Column 1: state text + L/R bars, spread across matrix height ---
        line_h = fm.height() + 6
        state_lines = [l for l in self._state_text.splitlines() if l]
        bar_h = 36
        bar_gap = 20
        text_block_h = len(state_lines) * line_h
        bars_block_h = bar_h * 2 + bar_gap
        remaining = mat_side - text_block_h - bars_block_h
        gap_text_bars = max(12, remaining // 2)
        text_y = matrix_y
        bar_top = matrix_y + text_block_h + gap_text_bars

        p.setPen(QtCore.Qt.black)
        for i, line in enumerate(state_lines):
            p.drawText(10, text_y + fm.ascent() + i * line_h, line)

        def draw_bar(bx: int, by: int, max_w: int, label: str, value: int) -> None:
            label_rect = fm.boundingRect(label)
            baseline_y = by + (bar_h + label_rect.height()) // 2
            p.setPen(QtCore.Qt.black)
            p.drawText(bx + 10, baseline_y, label)

            b_left = bx + 10 + label_rect.width() + 15
            b_right = bx + max_w - 20
            b_w = max(20, b_right - b_left)

            ratio = abs(value) / peak
            fill_w = int(b_w * ratio)
            color = _pressure_color(ratio)
            if fill_w > 0:
                p.fillRect(b_left, by, fill_w, bar_h, color)
            p.setPen(QtGui.QColor(100, 100, 100))
            p.drawRect(b_left, by, b_w, bar_h)

            val_text = f"{value:d}"
            val_rect = fm.boundingRect(val_text)
            p.setPen(QtCore.Qt.black)
            p.drawText(
                b_left + (b_w - val_rect.width()) // 2,
                by + (bar_h + val_rect.height()) // 2,
                val_text,
            )

        draw_bar(0, bar_top, col_w, "Left", left_val)
        draw_bar(0, bar_top + bar_h + bar_gap, col_w, "Right", right_val)

        # --- Column 2 & 3: heat matrices ---
        def draw_matrix(mx: int, matrix: list[list[int]]) -> None:
            for row_i in range(3):
                for col_i in range(3):
                    cx = mx + col_i * mat_cell_s
                    cy = matrix_y + row_i * mat_cell_s
                    val = matrix[row_i][col_i]
                    ratio = abs(val) / peak
                    color = _pressure_color(ratio)
                    p.fillRect(cx, cy, mat_inner, mat_inner, color)
                    p.setPen(QtGui.QColor(100, 100, 100))
                    p.drawRect(cx, cy, mat_inner, mat_inner)

                    val_text = str(val)
                    val_rect = fm.boundingRect(val_text)
                    p.setPen(QtCore.Qt.black)
                    p.drawText(
                        cx + (mat_inner - val_rect.width()) // 2,
                        cy + (mat_inner + val_rect.height()) // 2,
                        val_text,
                    )

        draw_matrix(col_w, left_mat)
        draw_matrix(col_w * 2, right_mat)

        p.end()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.setWindowTitle("Data Collector")
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtCore.Qt.white)
        self.setPalette(pal)

        world_serial, wrist_serial, devices = RealSenseRGB.resolve_serial_pair(
            args.world_serial, args.wrist_serial,
        )
        if devices:
            print("Detected RealSense devices:")
            for dev in devices:
                print(f"  - {dev['name']} serial={dev['serial']}")
        if world_serial is None:
            print("Warning: no world RealSense serial resolved; world camera will use fallback frames.")
        if wrist_serial is None:
            print("Warning: no wrist RealSense serial resolved; wrist camera will use fallback frames.")

        self.world_camera = RealSenseRGB(
            width=args.width, height=args.height, fps=args.rs_fps,
            enable_depth=True, enable_filters=False,
            serial_number=world_serial, name="world_camera",
            enabled=(world_serial is not None or not devices),
        )
        self.wrist_camera = RealSenseRGB(
            width=args.width, height=args.height, fps=args.rs_fps,
            enable_depth=True, enable_filters=False,
            serial_number=wrist_serial, name="wrist_camera",
            enabled=(wrist_serial is not None or not devices),
        )
        self.robot = RobotArmCollector(host=args.arm_host, port=args.arm_port)
        self.gripper: GripperStateCollector | None = None
        if not args.disable_gripper:
            self.gripper = GripperStateCollector(host=args.arm_host, port=args.arm_port)
        self.pressure = PressureCollector(
            local_port=args.pressure_local_port,
            remote_ip=args.pressure_remote_ip,
            remote_port=args.pressure_remote_port,
        )

        self.recording = False
        self.session_paths: SessionPaths | None = None
        self.frame_id = 0
        self.save_worker: FrameSaveWorker | None = None

        self._build_ui()
        self._start_collectors()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(int(VISUAL_INTERVAL_S * 1000))

    # ---- UI ----

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        root.setAutoFillBackground(True)
        pal = root.palette()
        pal.setColor(QtGui.QPalette.Window, QtCore.Qt.white)
        root.setPalette(pal)
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Camera row
        cam_grid = QtWidgets.QGridLayout()
        cam_grid.setSpacing(8)
        self.world_rgb_label = ImageLabel()
        self.world_depth_label = ImageLabel()
        self.wrist_rgb_label = ImageLabel()
        self.wrist_depth_label = ImageLabel()
        cam_grid.addWidget(self.world_rgb_label, 0, 0)
        cam_grid.addWidget(self.world_depth_label, 0, 1)
        cam_grid.addWidget(self.wrist_rgb_label, 1, 0)
        cam_grid.addWidget(self.wrist_depth_label, 1, 1)
        layout.addLayout(cam_grid, stretch=5)

        # Status label
        self.status_label = QtWidgets.QLabel("Status: IDLE | Frames: 0")
        font = self.status_label.font()
        font.setPointSize(11)
        self.status_label.setFont(font)
        self.status_label.setStyleSheet("color: #000000;")
        layout.addWidget(self.status_label)

        # Pressure dashboard (includes state text + bars + matrices)
        self.pressure_dashboard = PressureDashboard()
        layout.addWidget(self.pressure_dashboard, stretch=2)

    # ---- Collector lifecycle ----

    def _start_collectors(self) -> None:
        print("Starting world RealSense RGB-D pipeline...")
        self.world_camera.start()
        print("Starting wrist RealSense RGB-D pipeline...")
        self.wrist_camera.start()
        print(f"Connecting robot arm at {self.args.arm_host}:{self.args.arm_port}...")
        self.robot.connect()
        print("Starting robot arm collector...")
        self.robot.start()
        if self.gripper is not None:
            print("Connecting gripper RM Plus state collector...")
            self.gripper.connect()
            print("Starting gripper RM Plus state collector...")
            self.gripper.start()
        print("Starting pressure collector...")
        self.pressure.start()
        print("Press SPACE to start/stop recording, Q/ESC to exit.")

    def _stop_collectors(self) -> None:
        self.world_camera.stop()
        self.wrist_camera.stop()
        self.robot.stop()
        if self.gripper is not None:
            self.gripper.stop()
        self.pressure.stop()

    # ---- Recording toggle ----

    def _start_recording(self) -> None:
        self.session_paths = create_session_paths(self.args.output, self.args.session_prefix)
        self._write_camera_metadata(self.session_paths)
        self.pressure.start_session(self.session_paths.root)
        self.robot.start_session(self.session_paths.root)
        if self.gripper is not None:
            self.gripper.start_session(self.session_paths.root)

        self.save_worker = FrameSaveWorker(self.session_paths)
        self.save_worker.start()

        self.frame_id = 0
        self.recording = True
        print(f"Recording started. Session: {self.session_paths.root.name}")

    def _write_camera_metadata(self, session_paths: SessionPaths) -> None:
        payload = {
            "schema": "dual_realsense_camera_metadata/v1",
            "standard_profile": standard_realsense_profile(),
            "visual_recording_fps": VISUAL_FPS,
            "depth_note": "Depth PNG files are aligned to RGB/color pixels and saved as uint16 millimeters.",
            "cameras": {
                WORLD_CAMERA: self.world_camera.get_metadata(),
                WRIST_CAMERA: self.wrist_camera.get_metadata(),
            },
        }
        with open(session_paths.camera_metadata, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Camera metadata saved: {session_paths.camera_metadata}")

    def _stop_recording(self) -> None:
        self.pressure.stop_session()
        self.robot.stop_session()
        if self.gripper is not None:
            self.gripper.stop_session()

        saved = dropped = failed = max_q = 0
        if self.save_worker is not None:
            self.save_worker.stop()
            saved = self.save_worker.saved_frames
            dropped = self.save_worker.dropped_tasks
            failed = self.save_worker.failed_writes
            max_q = self.save_worker.max_queue_depth
            self.save_worker = None

        self.session_paths = None
        self.frame_id = 0
        self.recording = False
        print(
            "Recording paused. "
            f"saved_frames={saved}, dropped_save_tasks={dropped}, "
            f"failed_writes={failed}, max_save_queue={max_q}. "
            "Press SPACE to start a new session."
        )

    # ---- Timer callback (30Hz) ----

    def _on_timer(self) -> None:
        world_rgb = self.world_camera.read()
        world_depth = self.world_camera.read_depth()
        wrist_rgb = self.wrist_camera.read()
        wrist_depth = self.wrist_camera.read_depth()

        self.world_rgb_label.set_image(world_rgb)
        self.world_depth_label.set_image(_depth_to_preview_bgr(world_depth))
        self.wrist_rgb_label.set_image(wrist_rgb)
        self.wrist_depth_label.set_image(_depth_to_preview_bgr(wrist_depth))

        # Status
        status = "REC" if self.recording else "IDLE"
        queue_text = ""
        if self.save_worker is not None:
            queue_text = (
                f" | SaveQ: {self.save_worker.queue.qsize()}"
                f" | Drop: {self.save_worker.dropped_tasks}"
            )
        self.status_label.setText(
            f"Status: {status} | Frames: {self.frame_id if self.recording else 0}{queue_text}"
        )

        # Robot + gripper state → dashboard
        joints = self.robot.get_latest_joints()
        pose = self.robot.get_latest_pose()
        joint_text = "Joint: " + (" | ".join(f"{v:.1f}" for v in joints) if joints else "N/A")
        pose_text = "Pose:  " + (" | ".join(f"{v:.3f}" for v in pose) if pose else "N/A")

        if self.gripper is not None:
            gs = self.gripper.get_latest_state()
            pos = gs.get("gripper_pos")
            code = gs.get("code")
            latency = gs.get("latency_ms")
            if pos is None:
                gripper_text = f"Gripper: code={code} pos=N/A"
            else:
                gripper_text = f"Gripper: code={code} pos={pos} read={latency:.1f}ms"
        else:
            gripper_text = "Gripper: disabled"

        self.pressure_dashboard.set_state_info(f"{joint_text}\n{pose_text}\n{gripper_text}")
        self.pressure_dashboard.set_values(self.pressure.get_latest_values())

        # Save frames asynchronously. The UI thread only packages the latest synchronized sample.
        if self.recording and self.session_paths is not None and self.save_worker is not None:
            self.frame_id += 1
            capture_us = get_timestamp_us()
            self.save_worker.enqueue(
                FrameSaveTask(
                    frame_id=self.frame_id,
                    capture_us=capture_us,
                    world_rgb=world_rgb,
                    world_depth=world_depth,
                    wrist_rgb=wrist_rgb,
                    wrist_depth=wrist_depth,
                )
            )

    # ---- Keyboard ----

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            if self.recording:
                self._stop_recording()
            else:
                self._start_recording()
            return
        if key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
            return
        super().keyPressEvent(event)

    # ---- Cleanup ----

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._timer.stop()
        if self.recording:
            self._stop_recording()
        self._stop_collectors()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(args)
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
