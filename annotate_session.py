"""PySide6 session viewer with task/instruction annotations.

Recursively search for session directories under the given base path.
Annotations are stored as JSON inside each session folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# Pressure channel mapping
LEFT_CHANNEL = 51
RIGHT_CHANNEL = 50
LEFT_MATRIX_CHANNELS = [[63, 60, 57], [64, 61, 58], [49, 62, 59]]
RIGHT_MATRIX_CHANNELS = [[47, 44, 41], [48, 45, 42], [33, 46, 43]]

ANNOTATION_FILE = "annotations.json"


# --- Session discovery ---


def find_sessions(base: Path) -> List[Path]:
    """Recursively find all directories that contain dji/ and realsense_rgb/."""
    if not base.exists():
        return []
    results: List[Path] = []
    for dji_dir in base.rglob("dji"):
        session_dir = dji_dir.parent
        if (session_dir / "realsense_rgb").is_dir():
            results.append(session_dir)
    results.sort(key=lambda p: str(p))
    return results


# --- Annotations ---


def load_annotations(session_dir: Path) -> dict:
    path = session_dir / ANNOTATION_FILE
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def save_annotations(session_dir: Path, task: str, instruction: str) -> None:
    path = session_dir / ANNOTATION_FILE
    payload = {
        "task": task.strip(),
        "instruction": instruction.strip(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def common_frame_stems(dji_dir: Path, rs_dir: Path) -> List[str]:
    dji = {p.stem for p in dji_dir.glob("*.jpg")}
    rs = {p.stem for p in rs_dir.glob("*.jpg")}
    return sorted(dji & rs)


# --- Data loaders ---


def load_robot_state_csv(state_dir: Path) -> list[list[float]]:
    """Load robot_state.csv -> list of [joint*7, pose*6] per row."""
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
    """Load pressure.csv -> list of 64 int values per row."""
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


def format_robot_state(joints: Optional[list[float]], pose: Optional[list[float]]) -> str:
    j_str = "Joint: " + " | ".join(f"{v:.1f}" for v in joints) if joints else "Joint: N/A"
    p_str = "Pose:  " + " | ".join(f"{v:.3f}" for v in pose) if pose else "Pose:  N/A"
    return f"{j_str}\n{p_str}"


# --- Drawing helpers ---


def draw_text_centered(
    img: np.ndarray, text: str, cx: int, cy: int,
    font_scale: float = 0.6, color=(255, 255, 255), thickness: int = -1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = thickness if thickness > 0 else (2 if font_scale > 0.5 else 1)
    sz, _ = cv2.getTextSize(text, font, font_scale, th)
    cv2.putText(img, text, (int(cx - sz[0] / 2), int(cy + sz[1] / 2)),
                font, font_scale, color, th, cv2.LINE_AA)


def pressure_color(ratio: float, peak: float) -> tuple[int, int, int]:
    """Single blue hue: dark navy at 0, bright blue-white at peak (BGR)."""
    r = min(1.0, max(0.0, ratio))
    b = int(60 + 195 * r)
    g = int(30 + 180 * r)
    return (b, g, int(20 + 60 * r))


def draw_colorbar(canvas: np.ndarray, x: int, y: int, w: int, h: int, peak: float) -> None:
    """Vertical colorbar: dark at bottom (0), bright at top (peak)."""
    for row in range(h):
        ratio = 1.0 - row / max(h - 1, 1)
        color = pressure_color(ratio, 1.0)
        cv2.line(canvas, (x, y + row), (x + w, y + row), color, 1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (140, 140, 140), 1)

    steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    for frac in steps:
        ty = y + int(frac * h)
        val = peak * (1.0 - frac)
        cv2.putText(canvas, f"{val:.0f}", (x + w + 8, ty + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Peak", (x + w // 2 - 16, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA)


# --- Pressure dashboard ---

CELL_SIZE = 90
CELL_GAP = 8

BG_COLOR = (18, 18, 18)


def draw_pressure_dashboard(
    canvas: np.ndarray,
    left_x: int, left_w: int,
    right_x: int, right_w: int,
    bar_centers: list[int], total_h: int,
    values: list[int],
) -> None:
    """Draw left column bars and right area matrices + colorbar."""
    if not values or len(values) < 64:
        return

    def get_val(ch: int) -> int:
        return values[ch - 1] if 0 <= ch - 1 < len(values) else 0

    left_val = get_val(LEFT_CHANNEL)
    right_val = get_val(RIGHT_CHANNEL)
    left_mat = [[get_val(ch) for ch in row] for row in LEFT_MATRIX_CHANNELS]
    right_mat = [[get_val(ch) for ch in row] for row in RIGHT_MATRIX_CHANNELS]

    all_abs = [abs(left_val), abs(right_val)]
    all_abs += [abs(v) for row in left_mat for v in row]
    all_abs += [abs(v) for row in right_mat for v in row]
    peak = float(max(1, max(all_abs)))

    #  Left column: two horizontal bars stacked vertically, left-aligned 
    bar_h = 32

    for label, value, cy in [("Left", left_val, bar_centers[0]),
                              ("Right", right_val, bar_centers[1])]:
        bx = left_x
        by = cy - bar_h // 2
        cv2.putText(canvas, label, (bx, by - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA)
        ratio = min(1.0, abs(value) / peak)
        fill_w = int(left_w * ratio)
        if fill_w > 0:
            cv2.rectangle(canvas, (bx, by), (bx + fill_w, by + bar_h),
                          pressure_color(ratio, 1.0), -1)
        cv2.rectangle(canvas, (bx, by), (bx + left_w, by + bar_h), (160, 160, 160), 1)
        text_c = (255, 255, 255) if ratio > 0.45 else (30, 30, 30)
        draw_text_centered(canvas, f"{value:d}", bx + left_w // 2, by + bar_h // 2,
                   font_scale=0.7, color=text_c, thickness=1)

    #  Right area: two 3x3 matrices side by side + colorbar (no labels) 
    mat_w = CELL_SIZE * 3 + CELL_GAP * 2
    mat_h = mat_w
    colorbar_w = 20
    mat_gap = 36
    cb_gap = 24
    total_mat_area = mat_w * 2 + mat_gap + colorbar_w + cb_gap
    mat_origin_x = right_x + (right_w - total_mat_area) // 2
    mat_y = (total_h - mat_h) // 2

    def draw_matrix(mx: int, my: int, matrix: list[list[int]]) -> None:
        for ri in range(3):
            for ci in range(3):
                cx = mx + ci * (CELL_SIZE + CELL_GAP)
                cy = my + ri * (CELL_SIZE + CELL_GAP)
                val = matrix[ri][ci]
                ratio = min(1.0, abs(val) / peak)
                color = pressure_color(ratio, 1.0)
                cv2.rectangle(canvas, (cx, cy), (cx + CELL_SIZE, cy + CELL_SIZE), color, -1)
                cv2.rectangle(canvas, (cx, cy), (cx + CELL_SIZE, cy + CELL_SIZE), (50, 50, 50), 2)
                text_c = (255, 255, 255) if ratio > 0.35 else (30, 30, 30)
                draw_text_centered(canvas, str(val), cx + CELL_SIZE // 2, cy + CELL_SIZE // 2,
                                   font_scale=0.7, color=text_c, thickness=1)

    draw_matrix(mat_origin_x, mat_y, left_mat)
    draw_matrix(mat_origin_x + mat_w + mat_gap, mat_y, right_mat)

    cb_x = mat_origin_x + mat_w * 2 + mat_gap + cb_gap
    draw_colorbar(canvas, cb_x, mat_y, colorbar_w, mat_h, peak)


# --- Canvas composition ---


def compose_view_canvas(
    dji_img: np.ndarray,
    rs_img: np.ndarray,
    frame_label: str,
    state_text: str,
    pressure_values: list[int],
) -> np.ndarray:
    height = max(dji_img.shape[0], rs_img.shape[0])

    def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    left = resize_to_height(dji_img, height)
    right = resize_to_height(rs_img, height)

    gap = 6
    width = left.shape[1] + right.shape[1] + gap
    top = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    top[:, :left.shape[1]] = left
    top[:, left.shape[1] + gap:] = right

    # Info panel: left 40% (info+bars), right 60% (matrices)
    margin = 12
    state_lines = state_text.splitlines()

    mat_h = CELL_SIZE * 3 + CELL_GAP * 2

    left_w = int((width - margin * 2) * 0.4)
    right_w = width - margin * 2 - left_w
    left_x = margin
    right_x = left_x + left_w

    info_height = mat_h + 40
    info = np.full((info_height, width, 3), (245, 245, 245), dtype=np.uint8)

    text_y = 24
    cv2.putText(info, frame_label, (left_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 60, 140), 2, cv2.LINE_AA)
    line_h = 30
    for i, line in enumerate(state_lines):
        y_pos = text_y + line_h * (i + 1)
        cv2.putText(info, line, (left_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)

    bar_h = 32
    bar_gap = 12
    bar_area_h = bar_h * 2 + bar_gap
    text_block_h = text_y + line_h * (len(state_lines) + 1)
    bar_top = max(info_height - bar_area_h - 8, text_block_h + 12)
    bar_positions = [bar_top + bar_h // 2, bar_top + bar_h + bar_gap + bar_h // 2]
    draw_pressure_dashboard(info, left_x, left_w, right_x, right_w, bar_positions, info_height, pressure_values)

    canvas = np.vstack((top, info))
    return canvas


def _bgr_to_qimage(img: np.ndarray) -> QtGui.QImage:
    if img is None:
        return QtGui.QImage()
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        return qimg.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    bytes_per_line = 3 * w
    qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    return qimg.copy()


class ImageLabel(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self._pixmap = QtGui.QPixmap()

    def set_image(self, img: np.ndarray) -> None:
        self._pixmap = QtGui.QPixmap.fromImage(_bgr_to_qimage(img))
        self._update_scaled()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)


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

        ann = load_annotations(self.session)
        self.task = str(ann.get("task", ""))
        self.instruction = str(ann.get("instruction", ""))

    def label(self) -> str:
        return session_list_label(self.session, self.base)

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

    def frame_canvas(self, frame_idx: int, session_index: int, total_sessions: int) -> np.ndarray:
        stem = self.stems[frame_idx]
        dji_img = cv2.imread(str(self.dji_dir / f"{stem}.jpg"))
        rs_img = cv2.imread(str(self.rs_dir / f"{stem}.jpg"))
        if dji_img is None or rs_img is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Missing frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank

        joints, pose = self._get_robot_state(frame_idx)
        state_text = format_robot_state(joints, pose)
        pressure_values = self._get_pressure_values(frame_idx)

        session_label = self.session.parent.name + "/" + self.session.name
        header = f"[{session_index + 1}/{total_sessions}] {session_label}  |  Frame {stem}"
        return compose_view_canvas(dji_img, rs_img, header, state_text, pressure_values)


def session_list_label(session: Path, base: Path) -> str:
    rel = session.relative_to(base)
    ann = load_annotations(session)
    has_ann = bool(ann.get("task") or ann.get("instruction"))
    return f"{rel}{' *' if has_ann else ''}"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sessions: List[Path], base: Path, start_index: int) -> None:
        super().__init__()
        self.setWindowTitle("Session Annotator")
        self.resize(1400, 900)

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
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        self.session_list = QtWidgets.QListWidget()
        self.session_list.setMinimumWidth(320)
        self.session_list.currentRowChanged.connect(self._on_session_selected)
        main_layout.addWidget(self.session_list, 0)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setSpacing(10)

        self.frame_label = QtWidgets.QLabel("No session selected")
        self.frame_label.setStyleSheet("font-weight: 600;")
        right_layout.addWidget(self.frame_label)

        self.image_label = ImageLabel()
        right_layout.addWidget(self.image_label, 1)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        right_layout.addWidget(self.frame_slider)

        form = QtWidgets.QFormLayout()
        self.task_input = QtWidgets.QLineEdit()
        self.instruction_input = QtWidgets.QLineEdit()
        form.addRow("Task", self.task_input)
        form.addRow("Instruction", self.instruction_input)
        right_layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self._save_annotations)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch(1)
        right_layout.addLayout(btn_row)

        self.status_label = QtWidgets.QLabel("Press A to toggle session list")
        right_layout.addWidget(self.status_label)

        main_layout.addWidget(right, 1)

    def _populate_sessions(self) -> None:
        self.session_list.clear()
        for s in self.sessions:
            self.session_list.addItem(session_list_label(s, self.base))

    def _on_session_selected(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        try:
            self.session_data = SessionData(self.sessions[index], self.base)
        except FileNotFoundError as exc:
            QtWidgets.QMessageBox.warning(self, "Session error", str(exc))
            return

        self.current_index = index
        self.task_input.setText(self.session_data.task)
        self.instruction_input.setText(self.session_data.instruction)
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
        canvas = self.session_data.frame_canvas(idx, self.current_index, len(self.sessions))
        self.image_label.set_image(canvas)
        self.frame_label.setText(f"Frame {idx + 1}/{len(self.session_data.stems)}")

    def _save_annotations(self) -> None:
        if not self.session_data:
            return
        self.session_data.task = self.task_input.text().strip()
        self.session_data.instruction = self.instruction_input.text().strip()
        save_annotations(self.session_data.session, self.session_data.task, self.session_data.instruction)
        self.session_list.item(self.current_index).setText(self.session_data.label())
        path = self.session_data.session / ANNOTATION_FILE
        self.status_label.setText(f"Saved: {path}")
        print(f"Saved annotations to {path}")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
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
    parser = argparse.ArgumentParser(description="Session data viewer with annotations")
    parser.add_argument("--sessions", type=Path, default=Path("sessions"),
                        help="Base directory to recursively search for sessions")
    parser.add_argument("--session", type=str,
                        help="Specific session folder name (skips to that session directly)")
    return parser.parse_args()


def _print_hint() -> None:
    print("Keys: Left/Right=frame, N/M=session, A=toggle list, B=focus list")


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
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
