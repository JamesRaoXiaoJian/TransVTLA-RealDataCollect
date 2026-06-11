"""Offline viewer for synchronized DJI, RealSense, robot state, and pressure data.

Recursively search for session directories under the given base path.
PySide6 GUI with a session list, frame slider, and composed view.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

# Pressure channel mapping (from Channel Mapping.txt)
from channel_config import (
    LEFT_CHANNEL, RIGHT_CHANNEL,
    LEFT_MATRIX_CHANNELS, RIGHT_MATRIX_CHANNELS,
)

BG_COLOR = QtCore.Qt.black
CELL_SIZE = 90
CELL_GAP = 8


# --- Session discovery ---


def find_sessions(base: Path) -> List[Path]:
    if not base.exists():
        return []
    results: List[Path] = []
    for dji_dir in base.rglob("dji"):
        session_dir = dji_dir.parent
        if (session_dir / "realsense_rgb").is_dir():
            results.append(session_dir)
    results.sort(key=lambda p: str(p))
    return results


def common_frame_stems(dji_dir: Path, rs_dir: Path) -> List[str]:
    dji = {p.stem for p in dji_dir.glob("*.jpg")}
    rs = {p.stem for p in rs_dir.glob("*.jpg")}
    return sorted(dji & rs)


# --- Data loaders ---


def load_robot_state_csv(state_dir: Path) -> list[list[float]]:
    csv_path = state_dir / "robot_state.csv"
    if not csv_path.exists():
        return []
    rows: list[list[float]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                vals = [float(x) for x in row[1:14]]
                if len(vals) == 13:
                    rows.append(vals)
            except (ValueError, IndexError):
                pass
    return rows


def load_pressure_csv(pressure_dir: Path) -> list[list[int]]:
    csv_path = pressure_dir / "pressure.csv"
    if not csv_path.exists():
        return []
    rows: list[list[int]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                vals = [int(x) for x in row[1:65]]
                if len(vals) == 64:
                    rows.append(vals)
            except (ValueError, IndexError):
                pass
    return rows


def load_gripper_state_csv(state_dir: Path) -> list[list]:
    """Load gripper_state.csv -> list of [code, pos, latency_ms] per row."""
    csv_path = state_dir / "gripper_state.csv"
    if not csv_path.exists():
        return []
    rows: list[list] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                code = int(row[2]) if row[2] else -1
                pos = int(row[5]) if row[5] else None
                latency = float(row[3]) if row[3] else None
                rows.append([code, pos, latency])
            except (ValueError, IndexError):
                pass
    return rows


def format_robot_state(
    joints: Optional[list[float]],
    pose: Optional[list[float]],
    gripper_pos: Optional[int] = None,
    gripper_code: Optional[int] = None,
    gripper_latency: Optional[float] = None,
) -> str:
    j_str = "Joint: " + " | ".join(f"{v:.1f}" for v in joints) if joints else "Joint: N/A"
    p_str = "Pose:  " + " | ".join(f"{v:.3f}" for v in pose) if pose else "Pose:  N/A"
    if gripper_pos is None:
        g_str = "Gripper: N/A"
    else:
        g_str = f"Gripper: code={gripper_code} pos={gripper_pos}"
        if gripper_latency is not None:
            g_str += f" read={gripper_latency:.1f}ms"
    return f"{j_str}\n{p_str}\n{g_str}"


# --- Pressure dashboard widget ---


def _pressure_color(ratio: float) -> QtGui.QColor:
    ratio = min(1.0, max(0.0, ratio))
    b = int(60 + 195 * ratio)
    g = int(30 + 180 * ratio)
    r = int(20 + 60 * ratio)
    return QtGui.QColor(r, g, b)


class PressureDashboard(QtWidgets.QWidget):
    """Left: state info + L/R bar charts. Right: two 3x3 heat-matrices + colorbar."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._values: list[int] = [0] * 64
        self._state_text = ""
        self.setMinimumHeight(320)

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

        margin = 20
        fm = p.fontMetrics()

        left_w = int((w - margin * 2) * 0.4)
        right_x = margin + left_w
        right_w = w - margin * 2 - left_w

        # --- Matrices geometry (compute first for alignment) ---
        mat_side = CELL_SIZE * 3 + CELL_GAP * 2
        colorbar_w = 20
        mat_gap = 36
        cb_gap = 24
        total_mat_area = mat_side * 2 + mat_gap + colorbar_w + cb_gap
        mat_origin_x = right_x + (right_w - total_mat_area) // 2
        mat_y = 10

        # --- State text + L/R bars, spread across matrix height ---
        line_h = fm.height() + 6
        state_lines = [l for l in self._state_text.splitlines() if l]
        bar_h = 32
        bar_gap = 16
        text_block_h = len(state_lines) * line_h
        bars_block_h = bar_h * 2 + bar_gap
        remaining = mat_side - text_block_h - bars_block_h
        gap_text_bars = max(12, remaining // 2)
        text_y = mat_y
        bar_top = mat_y + text_block_h + gap_text_bars

        p.setPen(QtGui.QColor(30, 30, 30))
        for i, line in enumerate(state_lines):
            p.drawText(margin, text_y + fm.ascent() + i * line_h, line)

        for i, (label, value) in enumerate([("Left", left_val), ("Right", right_val)]):
            bx = margin
            by = bar_top + i * (bar_h + bar_gap)
            p.setPen(QtCore.Qt.black)
            p.drawText(bx, by - 6, label)

            ratio = abs(value) / peak
            fill_w = int(left_w * ratio)
            if fill_w > 0:
                p.fillRect(bx, by, fill_w, bar_h, _pressure_color(ratio))
            p.setPen(QtGui.QColor(160, 160, 160))
            p.drawRect(bx, by, left_w, bar_h)

            val_text = f"{value:d}"
            val_rect = fm.boundingRect(val_text)
            text_color = QtCore.Qt.white if ratio > 0.45 else QtCore.Qt.black
            p.setPen(text_color)
            p.drawText(
                bx + (left_w - val_rect.width()) // 2,
                by + (bar_h + val_rect.height()) // 2,
                val_text,
            )

        # --- Heat matrices ---
        def draw_matrix(mx: int, my: int, matrix: list[list[int]]) -> None:
            for ri in range(3):
                for ci in range(3):
                    cx = mx + ci * (CELL_SIZE + CELL_GAP)
                    cy = my + ri * (CELL_SIZE + CELL_GAP)
                    val = matrix[ri][ci]
                    ratio = abs(val) / peak
                    color = _pressure_color(ratio)
                    p.fillRect(cx, cy, CELL_SIZE, CELL_SIZE, color)
                    p.setPen(QtGui.QColor(50, 50, 50))
                    p.drawRect(cx, cy, CELL_SIZE, CELL_SIZE)

                    val_text = str(val)
                    val_rect = fm.boundingRect(val_text)
                    text_color = QtCore.Qt.white if ratio > 0.35 else QtCore.Qt.black
                    p.setPen(text_color)
                    p.drawText(
                        cx + (CELL_SIZE - val_rect.width()) // 2,
                        cy + (CELL_SIZE + val_rect.height()) // 2,
                        val_text,
                    )

        draw_matrix(mat_origin_x, mat_y, left_mat)
        draw_matrix(mat_origin_x + mat_side + mat_gap, mat_y, right_mat)

        # --- Colorbar ---
        cb_x = mat_origin_x + mat_side * 2 + mat_gap + cb_gap
        for row in range(mat_side):
            r = 1.0 - row / max(mat_side - 1, 1)
            p.setPen(_pressure_color(r))
            p.drawLine(cb_x, mat_y + row, cb_x + colorbar_w, mat_y + row)
        p.setPen(QtGui.QColor(140, 140, 140))
        p.drawRect(cb_x, mat_y, colorbar_w, mat_side)

        steps = [0.0, 0.25, 0.5, 0.75, 1.0]
        p.setPen(QtCore.Qt.black)
        for frac in steps:
            ty = mat_y + int(frac * mat_side)
            val = peak * (1.0 - frac)
            p.drawText(cb_x + colorbar_w + 8, ty + 6, f"{val:.0f}")
        p.drawText(cb_x + colorbar_w // 2 - 16, mat_y - 10, "Peak")

        p.end()


# --- Image display widget ---


class ImageLabel(QtWidgets.QLabel):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 180)
        self._pixmap = QtGui.QPixmap()

    def set_pixmap(self, pm: QtGui.QPixmap) -> None:
        self._pixmap = pm
        self._update_scaled()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)


# --- Session data ---


class SessionData:
    def __init__(self, session_path: Path, base: Path):
        self.session = session_path
        self.base = base

        self.dji_dir = session_path / "dji"
        self.rs_dir = session_path / "realsense_rgb"
        self.state_dir = session_path / "robot_state"
        self.pressure_dir = session_path / "pressure"

        if not self.dji_dir.exists() or not self.rs_dir.exists():
            raise FileNotFoundError("Session missing dji/ or realsense_rgb/ directory.")

        self.stems = common_frame_stems(self.dji_dir, self.rs_dir)
        if not self.stems:
            raise FileNotFoundError("No overlapping frame names between DJI and RealSense.")

        self.robot_state_data = load_robot_state_csv(self.state_dir)
        self.pressure_data = load_pressure_csv(self.pressure_dir)
        self.gripper_data = load_gripper_state_csv(self.state_dir)

    def label(self) -> str:
        return str(self.session.relative_to(self.base))

    def _get_robot_state(self, frame_idx: int) -> tuple[Optional[list[float]], Optional[list[float]]]:
        if not self.robot_state_data:
            return None, None
        ratio = len(self.robot_state_data) / max(len(self.stems), 1)
        idx = min(int(frame_idx * ratio), len(self.robot_state_data) - 1)
        row = self.robot_state_data[idx]
        return row[:7], row[7:13]

    def _get_pressure_values(self, frame_idx: int) -> list[int]:
        if not self.pressure_data:
            return []
        ratio = len(self.pressure_data) / max(len(self.stems), 1)
        idx = min(int(frame_idx * ratio), len(self.pressure_data) - 1)
        return self.pressure_data[idx]

    def _get_gripper_state(self, frame_idx: int) -> tuple[Optional[int], Optional[int], Optional[float]]:
        """Returns (code, pos, latency_ms) or (None, None, None)."""
        if not self.gripper_data:
            return None, None, None
        ratio = len(self.gripper_data) / max(len(self.stems), 1)
        idx = min(int(frame_idx * ratio), len(self.gripper_data) - 1)
        row = self.gripper_data[idx]
        return row[0], row[1], row[2]

    def get_frame_data(
        self, frame_idx: int, session_index: int, total_sessions: int,
    ) -> tuple[QtGui.QPixmap, QtGui.QPixmap, str, str, list[int]]:
        stem = self.stems[frame_idx]
        dji_pm = QtGui.QPixmap(str(self.dji_dir / f"{stem}.jpg"))
        rs_pm = QtGui.QPixmap(str(self.rs_dir / f"{stem}.jpg"))

        joints, pose = self._get_robot_state(frame_idx)
        g_code, g_pos, g_latency = self._get_gripper_state(frame_idx)
        state_text = format_robot_state(joints, pose, g_pos, g_code, g_latency)
        pressure_values = self._get_pressure_values(frame_idx)

        session_label = self.session.parent.name + "/" + self.session.name
        header = f"[{session_index + 1}/{total_sessions}] {session_label}  |  Frame {stem}"
        return dji_pm, rs_pm, header, state_text, pressure_values


# --- Main window ---


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sessions: List[Path], base: Path, start_index: int) -> None:
        super().__init__()
        self.setWindowTitle("Session Viewer")

        self.base = base
        self.sessions = sessions
        self.current_index = -1
        self.session_data: Optional[SessionData] = None

        self._build_ui()
        self._populate_sessions()
        self.session_list.setVisible(False)
        self.list_visible = False
        if sessions:
            self.session_list.setCurrentRow(max(0, min(start_index, len(sessions) - 1)))

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        main_layout = QtWidgets.QHBoxLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)

        self.session_list = QtWidgets.QListWidget()
        self.session_list.setMinimumWidth(320)
        self.session_list.currentRowChanged.connect(self._on_session_selected)
        main_layout.addWidget(self.session_list, 0)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setSpacing(14)

        self.header_label = QtWidgets.QLabel("No session selected")
        self.header_label.setStyleSheet("font-weight: 600; font-size: 14px;")
        right_layout.addWidget(self.header_label)

        # Two cameras side by side
        cam_row = QtWidgets.QHBoxLayout()
        cam_row.setSpacing(10)
        self.dji_label = ImageLabel()
        self.rs_label = ImageLabel()
        cam_row.addWidget(self.dji_label, stretch=1)
        cam_row.addWidget(self.rs_label, stretch=1)
        right_layout.addLayout(cam_row, stretch=5)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        right_layout.addWidget(self.frame_slider)

        # Pressure dashboard (includes state text + bars + matrices)
        self.pressure_dashboard = PressureDashboard()
        right_layout.addWidget(self.pressure_dashboard, stretch=2)

        self.status_label = QtWidgets.QLabel("Press A to toggle session list | ESC to exit")
        right_layout.addWidget(self.status_label)

        main_layout.addWidget(right, 1)

    def _populate_sessions(self) -> None:
        self.session_list.clear()
        for s in self.sessions:
            self.session_list.addItem(str(s.relative_to(self.base)))

    def _on_session_selected(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        try:
            self.session_data = SessionData(self.sessions[index], self.base)
        except FileNotFoundError as exc:
            QtWidgets.QMessageBox.warning(self, "Session error", str(exc))
            return

        self.current_index = index
        self.frame_slider.setRange(0, max(len(self.session_data.stems) - 1, 0))
        self.frame_slider.setValue(0)
        self._render_frame()
        print(f"Loaded session: {self.sessions[index].name}")

    def _on_frame_changed(self, value: int) -> None:
        if not self.session_data:
            return
        self._render_frame()

    def _render_frame(self) -> None:
        if not self.session_data:
            return
        idx = int(self.frame_slider.value())
        dji_pm, rs_pm, header, state_text, pressure_values = self.session_data.get_frame_data(
            idx, self.current_index, len(self.sessions),
        )
        self.dji_label.set_pixmap(dji_pm)
        self.rs_label.set_pixmap(rs_pm)
        self.header_label.setText(header)
        self.pressure_dashboard.set_state_info(state_text)
        self.pressure_dashboard.set_values(pressure_values)
        self.status_label.setText(f"Frame {idx + 1}/{len(self.session_data.stems)} | Press A to toggle session list | ESC to exit")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self.close()
            return
        if key == QtCore.Qt.Key_A:
            self.list_visible = not self.list_visible
            self.session_list.setVisible(self.list_visible)
            if self.list_visible:
                self.session_list.setFocus()
            return
        if key == QtCore.Qt.Key_Left:
            self.frame_slider.setValue(max(self.frame_slider.value() - 1, self.frame_slider.minimum()))
            return
        if key == QtCore.Qt.Key_Right:
            self.frame_slider.setValue(min(self.frame_slider.value() + 1, self.frame_slider.maximum()))
            return
        if key in (QtCore.Qt.Key_N,):
            new_idx = (self.current_index + 1) % len(self.sessions)
            self.session_list.setCurrentRow(new_idx)
            return
        if key in (QtCore.Qt.Key_M,):
            new_idx = (self.current_index - 1) % len(self.sessions)
            self.session_list.setCurrentRow(new_idx)
            return
        if key == QtCore.Qt.Key_B:
            self.session_list.setFocus()
            return
        super().keyPressEvent(event)


# --- Main ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session data viewer")
    parser.add_argument("--sessions", type=Path, default=Path("sessions"),
                        help="Base directory to recursively search for sessions")
    parser.add_argument("--session", type=str,
                        help="Specific session folder name (skips to that session directly)")
    return parser.parse_args()


def _print_hint() -> None:
    print("Keys: Left/Right=frame, N/M=session, A=toggle list, B=focus list, ESC=exit")


def main() -> None:
    args = parse_args()
    sessions = find_sessions(args.sessions)
    if not sessions:
        raise SystemExit(f"No sessions found under {args.sessions}.")

    start_idx = 0
    if args.session:
        for i, s in enumerate(sessions):
            if s.name == args.session:
                start_idx = i
                break
        else:
            raise SystemExit(f"Session '{args.session}' not found under {args.sessions}.")

    _print_hint()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(sessions, args.sessions, start_idx)
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
