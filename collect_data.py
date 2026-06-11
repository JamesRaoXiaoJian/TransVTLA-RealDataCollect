"""Collect synchronized data from DJI, RealSense RGB, robot, pressure, and gripper state.

Sampling rates:
    - Visual (DJI + RealSense): 20Hz
    - Tactile/Pressure: 200Hz
    - Robot Arm State: 100Hz
    - Gripper RM Plus State: 200Hz target

Press SPACE to start/stop individual recording sessions. Each session creates a
timestamped folder containing:
    - `dji/`             : DJI Osmo Action RGB frames (20Hz)
    - `realsense_rgb/`   : Intel RealSense RGB frames (20Hz)
    - `realsense_depth/` : Intel RealSense depth frames (20Hz, 16-bit PNG)
    - `robot_state/`     : CSV robot arm state + gripper state
    - `pressure/`        : CSV pressure samples at 200Hz

Use Q or ESC to exit at any time.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from timestamp_utils import get_timestamp_us

from collectors import (
    DJICamera,
    GripperStateCollector,
    PressureCollector,
    RealSenseRGB,
    RobotArmCollector,
)
from collectors.dji_camera import DEFAULT_DJI_INDEX
from collectors.pressure import (
    DEFAULT_PRESSURE_LOCAL_PORT,
    DEFAULT_PRESSURE_REMOTE_IP,
    DEFAULT_PRESSURE_REMOTE_PORT,
)
from collectors.robot_arm import DEFAULT_ARM_HOST, DEFAULT_ARM_PORT

VISUAL_FPS = 20
VISUAL_INTERVAL_S = 1.0 / VISUAL_FPS

# Pressure channel mapping (from Channel Mapping.txt)
from channel_config import (
    LEFT_CHANNEL, RIGHT_CHANNEL,
    LEFT_MATRIX_CHANNELS, RIGHT_MATRIX_CHANNELS,
)


@dataclass
class SessionPaths:
    root: Path
    dji: Path
    realsense: Path
    realsense_depth: Path
    pressure: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DJI + RealSense + robot arm + pressure + gripper recorder.")
    parser.add_argument("--dji-index", type=int, default=DEFAULT_DJI_INDEX, help="OpenCV index for DJI camera.")
    parser.add_argument("--width", type=int, default=1280, help="Frame width for both streams.")
    parser.add_argument("--height", type=int, default=720, help="Frame height for both streams.")
    parser.add_argument("--rs-fps", type=int, default=30, help="RealSense camera FPS (supports 30/60/90).")
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
    dji_dir = session_root / "dji"
    rs_dir = session_root / "realsense_rgb"
    rs_depth_dir = session_root / "realsense_depth"
    pressure_dir = session_root / "pressure"

    for directory in (dji_dir, rs_dir, rs_depth_dir, pressure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Recording session created: {session_root}")
    return SessionPaths(root=session_root, dji=dji_dir, realsense=rs_dir, realsense_depth=rs_depth_dir, pressure=pressure_dir)


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

        self.dji = DJICamera(index=args.dji_index, width=args.width, height=args.height)
        # RealSense 使用优化配置：后台线程 + 关闭实时深度滤波
        self.rs_camera = RealSenseRGB(
            width=848, height=480, fps=args.rs_fps,
            enable_depth=True, enable_filters=False,
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
        self.frames_file = None
        self.frames_writer = None

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
        cam_row = QtWidgets.QHBoxLayout()
        cam_row.setSpacing(10)
        self.dji_label = ImageLabel()
        self.rs_label = ImageLabel()
        cam_row.addWidget(self.dji_label, stretch=1)
        cam_row.addWidget(self.rs_label, stretch=1)
        layout.addLayout(cam_row, stretch=5)

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
        print(f"Opening DJI camera at index {self.args.dji_index}...")
        self.dji.start()
        print("Starting RealSense RGB pipeline...")
        self.rs_camera.start()
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
        self.dji.stop()
        self.rs_camera.stop()
        self.robot.stop()
        if self.gripper is not None:
            self.gripper.stop()
        self.pressure.stop()

    # ---- Recording toggle ----

    def _start_recording(self) -> None:
        self.session_paths = create_session_paths(self.args.output, self.args.session_prefix)
        self.pressure.start_session(self.session_paths.root)
        self.robot.start_session(self.session_paths.root)
        if self.gripper is not None:
            self.gripper.start_session(self.session_paths.root)

        # 创建 frames.csv（帧元数据）
        frames_path = self.session_paths.root / "frames.csv"
        self.frames_file = open(frames_path, "w", newline="", encoding="utf-8")
        self.frames_writer = csv.writer(self.frames_file)
        self.frames_writer.writerow([
            "frame_id", "capture_monotonic_us",
            "dji_save_us", "realsense_save_us", "depth_save_us",
        ])

        self.frame_id = 0
        self.recording = True
        print(f"Recording started. Session: {self.session_paths.root.name}")

    def _stop_recording(self) -> None:
        self.pressure.stop_session()
        self.robot.stop_session()
        if self.gripper is not None:
            self.gripper.stop_session()

        # 关闭 frames.csv
        if hasattr(self, 'frames_file') and self.frames_file is not None:
            self.frames_file.close()
            self.frames_file = None
            self.frames_writer = None

        self.session_paths = None
        self.frame_id = 0
        self.recording = False
        print("Recording paused. Press SPACE to start a new session.")

    # ---- Timer callback (20Hz) ----

    def _on_timer(self) -> None:
        dji_frame = self.dji.read()
        rs_frame = self.rs_camera.read()

        self.dji_label.set_image(dji_frame)
        self.rs_label.set_image(rs_frame)

        # Status
        status = "REC" if self.recording else "IDLE"
        self.status_label.setText(f"Status: {status} | Frames: {self.frame_id if self.recording else 0}")

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

        # Save frames
        if self.recording and self.session_paths is not None:
            self.frame_id += 1
            image_name = f"{self.frame_id:04d}.jpg"
            depth_name = f"{self.frame_id:04d}.png"

            # 使用 cv2.imwrite 保存 JPEG (Q85)，替代 QImage.save（更快）
            capture_us = get_timestamp_us()
            cv2.imwrite(
                str(self.session_paths.dji / image_name), dji_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            dji_save_us = get_timestamp_us()

            cv2.imwrite(
                str(self.session_paths.realsense / image_name), rs_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            rs_save_us = get_timestamp_us()

            depth_frame = self.rs_camera.read_depth()
            cv2.imwrite(str(self.session_paths.realsense_depth / depth_name), depth_frame)
            depth_save_us = get_timestamp_us()

            # 记录帧元数据
            if self.frames_writer is not None:
                self.frames_writer.writerow([
                    self.frame_id, capture_us, dji_save_us, rs_save_us, depth_save_us,
                ])
                if self.frame_id % 10 == 0:
                    self.frames_file.flush()

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
