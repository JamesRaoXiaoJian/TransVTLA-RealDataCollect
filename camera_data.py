"""Preview and capture data from a DJI Osmo Action webcam feed and
an Intel RealSense D435 depth camera.

Two workflows are provided:
1. Preview mode: quickly bring up both sensors for a visual check.
2. Capture mode: store a small batch of synchronized frames for validation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
	import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"pyrealsense2 is required for RealSense capture. Install with "
		"'pip install pyrealsense2'."
	) from exc


def probe_cameras(max_index: int = 10) -> List[int]:
	"""Return indices that open successfully via OpenCV."""

	indices: List[int] = []
	for idx in range(max_index):
		capture = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
		if capture.isOpened():
			indices.append(idx)
		capture.release()
	return indices


class DJICameraRecorder:
	def __init__(self, index: int, width: int = 1920, height: int = 1080):
		self.index = index
		self.width = width
		self.height = height
		self.capture: Optional[cv2.VideoCapture] = None

	def open(self) -> None:
		capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
		if not capture.isOpened():
			raise RuntimeError(f"Failed to open DJI camera at index {self.index}.")
		capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
		capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
		self.capture = capture

	def read(self) -> Optional[np.ndarray]:
		if self.capture is None:
			raise RuntimeError("DJI camera not opened.")
		ok, frame = self.capture.read()
		return frame if ok else None

	def release(self) -> None:
		if self.capture is not None:
			self.capture.release()
			self.capture = None


class RealSenseRecorder:
	def __init__(self, fps: int = 30, width: int = 1280, height: int = 720):
		self.fps = fps
		self.width = width
		self.height = height
		self.pipeline: Optional[rs.pipeline] = None
		self.align: Optional[rs.align] = None

	def start(self) -> None:
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
		config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
		profile = pipeline.start(config)
		depth_sensor = profile.get_device().first_depth_sensor()
		if depth_sensor and depth_sensor.supports(rs.option.enable_auto_exposure):
			depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
		self.pipeline = pipeline
		self.align = rs.align(rs.stream.color)

	def fetch(self, timeout_ms: int = 500) -> Tuple[np.ndarray, np.ndarray]:
		if self.pipeline is None or self.align is None:
			raise RuntimeError("RealSense pipeline not started.")
		frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
		aligned_frames = self.align.process(frames)
		color_frame = aligned_frames.get_color_frame()
		depth_frame = aligned_frames.get_depth_frame()
		if not color_frame or not depth_frame:
			raise RuntimeError("Incomplete frameset from RealSense.")
		color = np.asanyarray(color_frame.get_data())
		depth = np.asanyarray(depth_frame.get_data())
		return color, depth

	def stop(self) -> None:
		if self.pipeline is not None:
			self.pipeline.stop()
			self.pipeline = None
			self.align = None


@dataclass
class OutputDirs:
	dji: Path
	rs_color: Path
	rs_depth: Path
	rs_depth_preview: Optional[Path]


def prepare_output_dirs(base: Path, depth_preview: bool) -> OutputDirs:
	dji_dir = base / "dji"
	rs_color_dir = base / "realsense" / "color"
	rs_depth_dir = base / "realsense" / "depth_raw"
	rs_depth_preview_dir = base / "realsense" / "depth_color" if depth_preview else None

	for directory in (dji_dir, rs_color_dir, rs_depth_dir, rs_depth_preview_dir):
		if directory is not None:
			directory.mkdir(parents=True, exist_ok=True)

	return OutputDirs(
		dji=dji_dir,
		rs_color=rs_color_dir,
		rs_depth=rs_depth_dir,
		rs_depth_preview=rs_depth_preview_dir,
	)


def write_frames(
	timestamp: str,
	dji_frame: Optional[np.ndarray],
	rs_color: Optional[np.ndarray],
	rs_depth: Optional[np.ndarray],
	outputs: OutputDirs,
	depth_preview: bool,
) -> None:
	if dji_frame is not None:
		cv2.imwrite(str(outputs.dji / f"dji_{timestamp}.jpg"), dji_frame)
	if rs_color is not None:
		cv2.imwrite(str(outputs.rs_color / f"rs_color_{timestamp}.jpg"), rs_color)
	if rs_depth is not None:
		depth_path = outputs.rs_depth / f"rs_depth_{timestamp}.png"
		cv2.imwrite(str(depth_path), rs_depth)
		if depth_preview and outputs.rs_depth_preview is not None:
			scaled = cv2.convertScaleAbs(rs_depth, alpha=0.03)
			colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
			preview_path = outputs.rs_depth_preview / f"rs_depth_color_{timestamp}.jpg"
			cv2.imwrite(str(preview_path), colored)


def preview_cameras(
	dji_index: int,
	duration: float,
	depth_preview: bool,
	fps: int,
	width: int,
	height: int,
) -> None:
	realsense = RealSenseRecorder(width=width, height=height, fps=fps)
	dji = DJICameraRecorder(index=dji_index, width=width, height=height)

	print("Starting RealSense preview pipeline...")
	realsense.start()
	print("Opening DJI preview stream...")
	dji.open()

	frame_interval = max(1.0 / max(fps, 1), 0.01)
	last_preview = 0.0
	start_time = time.time()
	print(f"Previewing for up to {duration} seconds. Press 'q' to quit early.")

	try:
		while time.time() - start_time <= duration:
			dji_frame = dji.read()
			rs_color, rs_depth = realsense.fetch()
			now = time.time()

			if now - last_preview >= frame_interval:
				if dji_frame is not None:
					cv2.imshow("DJI Preview", dji_frame)
				cv2.imshow("RealSense Color Preview", rs_color)
				if depth_preview:
					scaled = cv2.convertScaleAbs(rs_depth, alpha=0.03)
					colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
					cv2.imshow("RealSense Depth Preview", colored)

				if cv2.waitKey(1) & 0xFF == ord("q"):
					print("Preview stopped by user.")
					break
				last_preview = now
	except KeyboardInterrupt:
		print("Preview interrupted by user.")
	finally:
		dji.release()
		realsense.stop()
		cv2.destroyAllWindows()


def capture_samples(
	dji_index: int,
	output_dir: Path,
	frame_count: int,
	depth_preview: bool,
	duration: float,
	fps: int,
	width: int,
	height: int,
	show_preview: bool,
) -> None:
	outputs = prepare_output_dirs(output_dir, depth_preview=depth_preview)
	realsense = RealSenseRecorder(width=width, height=height, fps=fps)
	dji = DJICameraRecorder(index=dji_index, width=width, height=height)

	print("Starting RealSense capture pipeline...")
	realsense.start()
	print("Opening DJI capture stream...")
	dji.open()

	captured = 0
	start_time = time.time()
	print(f"Capturing {frame_count} synchronized frame sets into {output_dir}.")

	try:
		while captured < frame_count and (time.time() - start_time) <= duration:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
			dji_frame = dji.read()
			rs_color, rs_depth = realsense.fetch()

			write_frames(
				timestamp=timestamp,
				dji_frame=dji_frame,
				rs_color=rs_color,
				rs_depth=rs_depth,
				outputs=outputs,
				depth_preview=depth_preview,
			)

			captured += 1
			print(f"Captured frame set {captured}/{frame_count}.")

			if show_preview:
				if dji_frame is not None:
					cv2.imshow("DJI Capture Preview", dji_frame)
				cv2.imshow("RealSense Color Capture", rs_color)
				if depth_preview:
					scaled = cv2.convertScaleAbs(rs_depth, alpha=0.03)
					colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
					cv2.imshow("RealSense Depth Capture", colored)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					print("Capture stopped by user input.")
					break

		if captured < frame_count:
			print(
				"Capture stopped because duration limit was reached before "
				f"collecting {frame_count} frames."
			)
	except KeyboardInterrupt:
		print("Capture interrupted by user.")
	finally:
		dji.release()
		realsense.stop()
		cv2.destroyAllWindows()
		print(f"Frames saved under {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Preview or capture data from DJI and RealSense cameras.")
	parser.add_argument("--dji-index", type=int, default=0, help="OpenCV device index for DJI webcam stream.")
	parser.add_argument("--duration", type=float, default=15.0, help="Max duration for preview/capture in seconds.")
	parser.add_argument("--output", type=Path, default=Path("output"), help="Directory to store captured data.")
	parser.add_argument("--mode", choices=["preview", "capture"], default="preview", help="Workflow to execute.")
	parser.add_argument("--frame-count", type=int, default=5, help="Frame sets to save in capture mode.")
	parser.add_argument("--list-cameras", action="store_true", help="List available OpenCV camera indices and exit.")
	parser.add_argument("--depth-preview", action="store_true", help="Show/save RealSense depth colormap.")
	parser.add_argument("--fps", type=int, default=15, help="RealSense FPS and preview refresh cap.")
	parser.add_argument("--width", type=int, default=1280, help="Frame width for both sensors.")
	parser.add_argument("--height", type=int, default=720, help="Frame height for both sensors.")
	parser.add_argument(
		"--max-probe-index",
		type=int,
		default=8,
		help="Upper bound (exclusive) when probing available cameras.",
	)
	parser.add_argument(
		"--no-display",
		action="store_true",
		help="Skip OpenCV windows during capture mode (still available in preview mode).",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	if args.list_cameras:
		indices = probe_cameras(max_index=args.max_probe_index)
		if indices:
			print("Detected OpenCV camera indices:", ", ".join(map(str, indices)))
		else:
			print("No cameras detected via OpenCV within the probed range.")
		return

	if args.mode == "preview":
		preview_cameras(
			dji_index=args.dji_index,
			duration=args.duration,
			depth_preview=args.depth_preview,
			fps=args.fps,
			width=args.width,
			height=args.height,
		)
		return

	capture_samples(
		dji_index=args.dji_index,
		output_dir=args.output.resolve(),
		frame_count=args.frame_count,
		depth_preview=args.depth_preview,
		duration=args.duration,
		fps=args.fps,
		width=args.width,
		height=args.height,
		show_preview=not args.no_display,
	)


if __name__ == "__main__":  # pragma: no cover
	main()

