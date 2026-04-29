"""Offline viewer for synchronized DJI, RealSense, and robot state data.

Select a session directory under `sessions/` and scrub through the captured
frames using an OpenCV trackbar or arrow keys.
"""

from __future__ import annotations

import argparse
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


def compose_canvas(
	dji_img: np.ndarray,
	rs_img: np.ndarray,
	frame_label: str,
	state_lines: str,
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

	info_height = 140
	info = np.zeros((info_height, width, 3), dtype=np.uint8)
	cv2.putText(info, frame_label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
	for i, line in enumerate(state_lines.splitlines(), start=1):
		cv2.putText(info, line, (20, 45 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)

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
		if not self.dji_dir.exists() or not self.rs_dir.exists():
			raise FileNotFoundError("Session missing dji or realsense_rgb directory.")
		self.stems = common_frame_stems(self.dji_dir, self.rs_dir)
		if not self.stems:
			raise FileNotFoundError("No overlapping frame names between DJI and RealSense.")
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
		canvas = compose_canvas(dji_img, rs_img, f"Frame {stem}", state_lines)
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

