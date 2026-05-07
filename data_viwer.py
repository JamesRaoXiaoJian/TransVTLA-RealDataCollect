"""Offline viewer for synchronized DJI, RealSense, and robot state data.

Select a session directory under `sessions/` and scrub through the captured
frames using an OpenCV trackbar or arrow keys.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def list_sessions(base: Path) -> List[Path]:
	if not base.exists():
		return []
	return sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)


def common_frame_stems(dji_dir: Path, rs_dir: Path) -> List[str]:
	dji = {p.stem for p in dji_dir.glob("*.jpg")}
	rs = {p.stem for p in rs_dir.glob("*.jpg")}
	stems = sorted(dji & rs)
	return stems


def load_robot_state(state_dir: Path, stem: str) -> str:
	path = state_dir / f"{stem}.json"
	if not path.exists():
		return "State: N/A"
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return "State: Invalid JSON"
	payload = data.get("state", {}).get("data") if isinstance(data, dict) else None
	if not isinstance(payload, dict):
		return "State: Missing"
	joints = payload.get("joint")
	pose = payload.get("pose")
	joint_txt = "Joint: " + (" | ".join(f"{val:.1f}" for val in joints) if joints else "N/A")
	pose_txt = "Pose: " + (" | ".join(f"{val:.3f}" for val in pose) if pose else "N/A")
	return f"{joint_txt}\n{pose_txt}"


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


def compose_canvas(
	dji_img: np.ndarray,
	rs_img: np.ndarray,
	frame_label: str,
	state_lines: str,
	pressure_values: list[int],
	max_width: int = 3840,
) -> np.ndarray:
	height = max(dji_img.shape[0], rs_img.shape[0])

	def resize_to_height(img: np.ndarray) -> np.ndarray:
		if img.shape[0] == height:
			return img
		ratio = height / img.shape[0]
		new_size = (int(img.shape[1] * ratio), height)
		return cv2.resize(img, new_size)

	left = resize_to_height(dji_img)
	right = resize_to_height(rs_img)

	gap = 40
	width = left.shape[1] + right.shape[1] + gap
	top = np.zeros((height, width, 3), dtype=np.uint8)
	top[:, : left.shape[1]] = left
	top[:, left.shape[1] + gap :] = right

	info_height = 420
	info = np.zeros((info_height, width, 3), dtype=np.uint8)
	cv2.putText(info, frame_label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
	for i, line in enumerate(state_lines.splitlines(), start=1):
		cv2.putText(info, line, (20, 45 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)

	draw_pressure_dashboard(info, 20, 140, width - 40, pressure_values)

	canvas = np.vstack((top, info))
	if canvas.shape[1] > max_width:
		scale = max_width / canvas.shape[1]
		new_size = (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale))
		canvas = cv2.resize(canvas, new_size)
	return canvas


class SessionViewer:
	def __init__(self, session_path: Path):
		self.session = session_path
		self.dji_dir = session_path / "dji"
		self.rs_dir = session_path / "realsense_rgb"
		self.state_dir = session_path / "robot_state"
		self.pressure_dir = session_path / "pressure"
		if not self.dji_dir.exists() or not self.rs_dir.exists():
			raise FileNotFoundError("Session missing dji or realsense_rgb directory.")
		self.stems = common_frame_stems(self.dji_dir, self.rs_dir)
		if not self.stems:
			raise FileNotFoundError("No overlapping frame names between DJI and RealSense.")

		self.pressure_data = []
		pressure_csv = self.pressure_dir / "pressure.csv"
		if pressure_csv.exists():
			with open(pressure_csv, "r", encoding="utf-8") as f:
				reader = csv.reader(f)
				header = next(reader, None)
				for row in reader:
					try:
						self.pressure_data.append([int(x) for x in row[1:65]])
					except (ValueError, IndexError):
						pass

		self.index = 0
		self.window = "Session Viewer"
		cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
		cv2.createTrackbar("frame", self.window, 0, len(self.stems) - 1, self.on_trackbar)

	def on_trackbar(self, value: int) -> None:
		self.index = int(np.clip(value, 0, len(self.stems) - 1))
		self.render()

	def next(self, step: int) -> None:
		self.index = int(np.clip(self.index + step, 0, len(self.stems) - 1))
		cv2.setTrackbarPos("frame", self.window, self.index)
		self.render()

	def render(self) -> None:
		stem = self.stems[self.index]
		dji_img = cv2.imread(str(self.dji_dir / f"{stem}.jpg"))
		rs_img = cv2.imread(str(self.rs_dir / f"{stem}.jpg"))
		if dji_img is None or rs_img is None:
			blank = np.zeros((480, 640, 3), dtype=np.uint8)
			cv2.putText(blank, "Missing frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
			cv2.imshow(self.window, blank)
			return
		state_lines = load_robot_state(self.state_dir, stem)
		
		pressure_values = []
		if self.pressure_data:
			idx = int(self.index * len(self.pressure_data) / max(1, len(self.stems)))
			idx = min(idx, len(self.pressure_data) - 1)
			pressure_values = self.pressure_data[idx]

		canvas = compose_canvas(dji_img, rs_img, f"Frame {stem}", state_lines, pressure_values)
		cv2.imshow(self.window, canvas)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Session data viewer")
	parser.add_argument("--sessions", type=Path, default=Path("sessions"), help="Base directory containing sessions")
	parser.add_argument("--session", type=str, help="Specific session folder name", nargs="?")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	sessions = list_sessions(args.sessions)
	if not sessions:
		raise SystemExit("No sessions found.")

	session_path: Optional[Path] = None
	if args.session:
		candidate = args.sessions / args.session
		if candidate.exists():
			session_path = candidate
		else:
			raise SystemExit(f"Session '{args.session}' not found under {args.sessions}.")
	else:
		session_path = sessions[0]

	viewer = SessionViewer(session_path)
	viewer.render()

	print(f"Viewing session: {session_path}")
	print("Use trackbar or ←/→ keys to navigate, Q/ESC to exit.")

	while True:
		key = cv2.waitKey(50)
		if key in (ord("q"), 27):
			break
		if key in (81, 2424832):  # left arrow
			viewer.next(-1)
		if key in (83, 2555904):  # right arrow
			viewer.next(1)

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

