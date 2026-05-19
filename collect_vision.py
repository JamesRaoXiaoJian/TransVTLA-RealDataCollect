"""Interactive data capture for DJI Osmo Action (UVC) and RealSense D435 RGB.

Press SPACE to start/stop recording sessions. Each resumed run is stored
in its own timestamped folder containing synchronized RGB frames from both
cameras. Press Q or ESC to exit the program.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyrealsense2 is required. Install it via 'pip install pyrealsense2'."
    ) from exc


DEFAULT_DJI_INDEX = 1  # Detected external DJI Osmo Action camera index on this host
MAX_PREVIEW_WIDTH = 1920  # Downscale preview if the combined width exceeds this value


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
        pipeline.start(config)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive DJI + RealSense capture tool.")
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
    return parser


def create_session_dirs(base: Path, prefix: str) -> Tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = base / f"{prefix}_{timestamp}"
    dji_dir = session_root / "dji"
    rs_dir = session_root / "realsense_rgb"
    dji_dir.mkdir(parents=True, exist_ok=True)
    rs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recording session created: {session_root}")
    return dji_dir, rs_dir


def stack_frames(left: np.ndarray, right: np.ndarray, max_width: int = MAX_PREVIEW_WIDTH) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        new_width = int(right.shape[1] * scale)
        right = cv2.resize(right, (new_width, left.shape[0]))
    combined = np.hstack((left, right))
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        new_size = (int(combined.shape[1] * scale), int(combined.shape[0] * scale))
        combined = cv2.resize(combined, new_size)
    return combined


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    dji = DJICamera(index=args.dji_index, width=args.width, height=args.height)
    rs_camera = RealSenseRGB(width=args.width, height=args.height, fps=args.fps)

    print(f"Opening DJI camera at index {args.dji_index}...")
    dji.start()
    print("Starting RealSense RGB pipeline...")
    rs_camera.start()

    recording = False
    dji_dir: Optional[Path] = None
    rs_dir: Optional[Path] = None
    frame_id = 0

    print("Press SPACE to start/stop recording sessions, Q/ESC to exit.")

    cv2.namedWindow("DJI + RealSense", cv2.WINDOW_NORMAL)

    try:
        while True:
            dji_frame = dji.read()
            try:
                rs_frame = rs_camera.read()
            except RuntimeError as err:
                print(f"Warning: RealSense frame skipped ({err}).")
                continue

            if dji_frame is None:
                print("Warning: empty frame from DJI camera.")
                continue

            combined = stack_frames(dji_frame, rs_frame)
            status = "REC" if recording else "IDLE"
            cv2.putText(
                combined,
                f"Status: {status} | Frames: {frame_id if recording else 0}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if recording else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("DJI + RealSense", combined)

            if recording and dji_dir and rs_dir:
                frame_name = f"{frame_id + 1:04d}.jpg"
                cv2.imwrite(str(dji_dir / frame_name), dji_frame)
                cv2.imwrite(str(rs_dir / frame_name), rs_frame)
                frame_id += 1

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Exiting capture loop.")
                break
            if key == ord(" "):
                recording = not recording
                if recording:
                    dji_dir, rs_dir = create_session_dirs(args.output, args.session_prefix)
                    frame_id = 0
                else:
                    dji_dir = None
                    rs_dir = None
                    frame_id = 0
                    print("Recording paused. Press SPACE to start a new session.")
    finally:
        dji.stop()
        rs_camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    main()
