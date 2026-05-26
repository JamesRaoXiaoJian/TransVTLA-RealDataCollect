"""Offline viewer for synchronized DJI, RealSense, robot state, and pressure data.

Recursively search for session directories under the given base path.
Shows a GUI session selector; use ↑/↓/Enter or click to pick a session.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Pressure channel mapping
LEFT_CHANNEL = 51
RIGHT_CHANNEL = 50
LEFT_MATRIX_CHANNELS = [[63, 60, 57], [64, 61, 58], [49, 62, 59]]
RIGHT_MATRIX_CHANNELS = [[47, 44, 41], [48, 45, 42], [33, 46, 43]]


# ─── Session discovery ───────────────────────────────────────────────────────


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


# ─── Session selector GUI ────────────────────────────────────────────────────

ROW_H = 38
VISIBLE_ROWS = 30
WIN_NAME = "Select Session"


class SessionSelector:
    """GUI list for picking a session. Returns index or -1 if cancelled."""

    def __init__(self, sessions: List[Path], base: Path):
        self.sessions = sessions
        self.base = base
        self.hover = 0
        self.selected = -1
        self.scroll_offset = 0

        # Pre-compute display labels and frame counts
        self.labels: list[str] = []
        self.frame_counts: list[int] = []
        for s in sessions:
            rel = s.relative_to(base)
            fc = len(list((s / "dji").glob("*.jpg"))) if (s / "dji").is_dir() else 0
            self.labels.append(str(rel))
            self.frame_counts.append(fc)

        self.total = len(sessions)

    def _clamp_scroll(self) -> None:
        max_scroll = max(0, self.total - VISIBLE_ROWS)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            row = y // ROW_H
            if 0 <= row < min(VISIBLE_ROWS, self.total - self.scroll_offset):
                self.hover = self.scroll_offset + row
        elif event == cv2.EVENT_LBUTTONDOWN:
            row = y // ROW_H
            if 0 <= row < min(VISIBLE_ROWS, self.total - self.scroll_offset):
                self.selected = self.scroll_offset + row
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.scroll_offset -= 3
            else:
                self.scroll_offset += 3
            self._clamp_scroll()

    def run(self) -> int:
        """Show selector and return chosen session index, or -1 on ESC."""
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN_NAME, self._on_mouse)

        while True:
            canvas = self._render()
            cv2.imshow(WIN_NAME, canvas)
            key = cv2.waitKey(30) & 0xFFFF
            ch = key & 0xFF

            if ch == 27:  # ESC
                cv2.destroyWindow(WIN_NAME)
                return -1
            if ch == 13 or ch == 10:  # Enter
                cv2.destroyWindow(WIN_NAME)
                return self.hover
            if key in (81, 2424832):  # Up arrow
                self.hover = max(0, self.hover - 1)
                if self.hover < self.scroll_offset:
                    self.scroll_offset = self.hover
            if key in (83, 2555904):  # Down arrow
                self.hover = min(self.total - 1, self.hover + 1)
                if self.hover >= self.scroll_offset + VISIBLE_ROWS:
                    self.scroll_offset = self.hover - VISIBLE_ROWS + 1

            if self.selected >= 0:
                cv2.destroyWindow(WIN_NAME)
                return self.selected

        cv2.destroyWindow(WIN_NAME)
        return -1

    def _render(self) -> np.ndarray:
        n_visible = min(VISIBLE_ROWS, self.total - self.scroll_offset)
        canvas_w = 900
        canvas = np.full((n_visible * ROW_H, canvas_w, 3), 22, dtype=np.uint8)

        for i in range(n_visible):
            idx = self.scroll_offset + i
            y0 = i * ROW_H
            y1 = y0 + ROW_H

            # Background
            if idx == self.hover:
                canvas[y0:y1, :] = (50, 50, 50)
            if idx % 2 == 1:
                canvas[y0:y1, :] = canvas[y0:y1, :].astype(np.int32) + 6
                canvas[y0:y1, :] = np.clip(canvas[y0:y1, :], 0, 255).astype(np.uint8)

            # Index
            cv2.putText(canvas, f"[{idx:3d}]", (12, y0 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
            # Label
            cv2.putText(canvas, self.labels[idx], (70, y0 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
            # Frame count
            fc_text = f"{self.frame_counts[idx]} frames"
            sz, _ = cv2.getTextSize(fc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, fc_text, (canvas_w - sz[0] - 16, y0 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

        # Scroll indicator
        if self.total > VISIBLE_ROWS:
            bar_h = max(20, int(VISIBLE_ROWS * ROW_H * VISIBLE_ROWS / self.total))
            bar_y = int(self.scroll_offset / max(1, self.total - VISIBLE_ROWS) * (VISIBLE_ROWS * ROW_H - bar_h))
            cv2.rectangle(canvas, (canvas_w - 8, bar_y), (canvas_w - 2, bar_y + bar_h), (100, 100, 100), -1)

        # Footer hint
        hint = f"{self.total} sessions  |  ↑↓ navigate  Enter select  ESC cancel  Mouse scroll"
        cv2.putText(canvas, hint, (12, n_visible * ROW_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)

        return canvas


def common_frame_stems(dji_dir: Path, rs_dir: Path) -> List[str]:
    dji = {p.stem for p in dji_dir.glob("*.jpg")}
    rs = {p.stem for p in rs_dir.glob("*.jpg")}
    return sorted(dji & rs)


# ─── Data loaders ────────────────────────────────────────────────────────────


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


# ─── Drawing helpers ─────────────────────────────────────────────────────────


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
        cv2.putText(canvas, f"{val:.0f}", (x + w + 8, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Peak", (x + w // 2 - 14, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)


# ─── Pressure dashboard ──────────────────────────────────────────────────────

CELL_SIZE = 80
CELL_GAP = 12

BG_COLOR = (24, 24, 24)


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

    # ── Left column: two horizontal bars stacked vertically, left-aligned ──
    bar_h = 28

    for label, value, cy in [("Left", left_val, bar_centers[0]),
                              ("Right", right_val, bar_centers[1])]:
        bx = left_x
        by = cy - bar_h // 2
        cv2.putText(canvas, label, (bx, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
        ratio = min(1.0, abs(value) / peak)
        fill_w = int(left_w * ratio)
        if fill_w > 0:
            cv2.rectangle(canvas, (bx, by), (bx + fill_w, by + bar_h),
                          pressure_color(ratio, 1.0), -1)
        cv2.rectangle(canvas, (bx, by), (bx + left_w, by + bar_h), (160, 160, 160), 1)
        text_c = (255, 255, 255) if ratio > 0.45 else (40, 40, 40)
        draw_text_centered(canvas, f"{value:d}", bx + left_w // 2, by + bar_h // 2,
                           font_scale=0.6, color=text_c, thickness=1)

    # ── Right area: two 3x3 matrices side by side + colorbar (no labels) ──
    mat_w = CELL_SIZE * 3 + CELL_GAP * 2
    mat_h = mat_w
    colorbar_w = 20
    mat_gap = 50
    cb_gap = 40
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
                cv2.rectangle(canvas, (cx, cy), (cx + CELL_SIZE, cy + CELL_SIZE), (60, 60, 60), 2)
                text_c = (255, 255, 255) if ratio > 0.4 else (40, 40, 40)
                draw_text_centered(canvas, str(val), cx + CELL_SIZE // 2, cy + CELL_SIZE // 2,
                                   font_scale=0.6, color=text_c, thickness=1)

    draw_matrix(mat_origin_x, mat_y, left_mat)
    draw_matrix(mat_origin_x + mat_w + mat_gap, mat_y, right_mat)

    cb_x = mat_origin_x + mat_w * 2 + mat_gap + cb_gap
    draw_colorbar(canvas, cb_x, mat_y, colorbar_w, mat_h, peak)


# ─── Canvas composition ──────────────────────────────────────────────────────


def compose_canvas(
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

    gap = 16
    width = left.shape[1] + right.shape[1] + gap
    top = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    top[:, :left.shape[1]] = left
    top[:, left.shape[1] + gap:] = right

    # ── Info panel: left 40% (info+bars), right 60% (matrices) ──
    margin = 20
    state_lines = state_text.splitlines()

    mat_h = CELL_SIZE * 3 + CELL_GAP * 2

    left_w = int((width - margin * 2) * 0.4)
    right_w = width - margin * 2 - left_w
    left_x = margin
    right_x = left_x + left_w

    # All left column items: frame_label, joint, pose, Left, Right
    # The bars are drawn by draw_pressure_dashboard; we only draw text here
    n_text_items = 1 + len(state_lines)  # frame_label + joint + pose
    n_total = n_text_items + 2  # + Left bar + Right bar
    info_height = mat_h + 40

    info = np.full((info_height, width, 3), (255, 255, 255), dtype=np.uint8)

    # Evenly distribute text items across full height (bars will fill remaining slots)
    item_spacing = info_height // (n_total + 1)
    for i, line in enumerate([frame_label] + state_lines):
        y_pos = item_spacing * (i + 1)
        if i == 0:
            cv2.putText(info, line, (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 100, 200), 2, cv2.LINE_AA)
        else:
            cv2.putText(info, line, (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)

    # Dashboard: left column = bars at evenly spaced positions, right area = matrices
    bar_positions = [item_spacing * (n_text_items + 1), item_spacing * (n_text_items + 2)]
    draw_pressure_dashboard(info, left_x, left_w, right_x, right_w, bar_positions, info_height, pressure_values)

    canvas = np.vstack((top, info))
    return canvas


# ─── Session viewer ───────────────────────────────────────────────────────────


class SessionViewer:
    def __init__(self, session_path: Path, session_index: int, total_sessions: int):
        self.session = session_path
        self.session_index = session_index
        self.total_sessions = total_sessions

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

        self.index = 0
        self.window = "Session Viewer"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("frame", self.window, 0, max(len(self.stems) - 1, 1), self._on_trackbar)

    def _on_trackbar(self, value: int) -> None:
        self.index = int(np.clip(value, 0, len(self.stems) - 1))
        self.render()

    def step(self, delta: int) -> None:
        self.index = int(np.clip(self.index + delta, 0, len(self.stems) - 1))
        cv2.setTrackbarPos("frame", self.window, self.index)
        self.render()

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

    def render(self) -> None:
        stem = self.stems[self.index]
        dji_img = cv2.imread(str(self.dji_dir / f"{stem}.jpg"))
        rs_img = cv2.imread(str(self.rs_dir / f"{stem}.jpg"))
        if dji_img is None or rs_img is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Missing frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow(self.window, blank)
            return

        joints, pose = self._get_robot_state(self.index)
        state_text = format_robot_state(joints, pose)
        pressure_values = self._get_pressure_values(self.index)

        session_label = self.session.parent.name + "/" + self.session.name
        header = f"[{self.session_index + 1}/{self.total_sessions}] {session_label}  |  Frame {stem}"
        canvas = compose_canvas(dji_img, rs_img, header, state_text, pressure_values)
        cv2.imshow(self.window, canvas)


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session data viewer")
    parser.add_argument("--sessions", type=Path, default=Path("sessions"),
                        help="Base directory to recursively search for sessions")
    parser.add_argument("--session", type=str,
                        help="Specific session folder name (skips to that session directly)")
    return parser.parse_args()


def _print_hint() -> None:
    print("\n" + "=" * 50)
    print("  ← →   前/后帧")
    print("  N / M  下一个/上一个 session")
    print("  B      返回选择列表")
    print("  Q/ESC  退出")
    print("=" * 50 + "\n")


def _open_viewer(sessions: List[Path], idx: int) -> int:
    """Open viewer for session at idx.
    Returns: -1=quit, 0=selector, positive=new session index to open.
    """
    viewer = SessionViewer(sessions[idx], idx, len(sessions))
    viewer.render()
    _print_hint()

    while True:
        key = cv2.waitKey(50) & 0xFFFF
        ch = key & 0xFF

        # Window closed via X button
        try:
            if cv2.getWindowProperty(viewer.window, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return -1
        except cv2.error:
            return -1

        if ch in (ord("q"), 27):
            cv2.destroyWindow(viewer.window)
            return -1
        if key in (81, 2424832):
            viewer.step(-1)
        if key in (83, 2555904):
            viewer.step(1)
        if ch == ord("b") or ch == ord("B"):
            cv2.destroyWindow(viewer.window)
            return 0
        if ch == ord("n") or ch == ord("N"):
            cv2.destroyWindow(viewer.window)
            new_idx = (idx + 1) % len(sessions)
            print(f"→ 切换到 session [{new_idx + 1}/{len(sessions)}]: {sessions[new_idx].name}")
            return new_idx
        if ch == ord("m") or ch == ord("M"):
            cv2.destroyWindow(viewer.window)
            new_idx = (idx - 1) % len(sessions)
            print(f"→ 切换到 session [{new_idx + 1}/{len(sessions)}]: {sessions[new_idx].name}")
            return new_idx


def main() -> None:
    args = parse_args()
    sessions = find_sessions(args.sessions)
    if not sessions:
        raise SystemExit(f"No sessions found under {args.sessions}.")

    # If --session specified, skip selector
    if args.session:
        start_idx = None
        for i, s in enumerate(sessions):
            if s.name == args.session:
                start_idx = i
                break
        if start_idx is None:
            raise SystemExit(f"Session '{args.session}' not found under {args.sessions}.")
        idx = start_idx
        while True:
            action = _open_viewer(sessions, idx)
            if action < 0:
                cv2.destroyAllWindows()
                return
            if action == 0:
                break  # fall through to selector
            idx = action

    # Main loop: selector -> viewer -> selector -> ...
    while True:
        selector = SessionSelector(sessions, args.sessions)
        idx = selector.run()
        if idx < 0:
            break
        # Viewer loop: stays until quit (-1) or back to selector (0)
        while True:
            action = _open_viewer(sessions, idx)
            if action < 0:
                cv2.destroyAllWindows()
                return
            if action == 0:
                break
            idx = action  # N/M session switch

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
