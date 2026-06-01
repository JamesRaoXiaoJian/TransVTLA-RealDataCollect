from __future__ import annotations

import time
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


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
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
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

    def read_depth(self) -> np.ndarray:
        if self.pipeline is None:
            return self._zero_depth()

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError("Missing RealSense depth frame.")
            return np.asanyarray(depth_frame.get_data())
        except Exception as exc:
            now = time.time()
            if (now - self.last_warn_time) >= 1.0:
                print(f"Warning: RealSense depth read failed. Using zero-filled frame. Details: {exc}")
                self.last_warn_time = now
            return self._zero_depth()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.available = False

    def _zero_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _zero_depth(self) -> np.ndarray:
        return np.zeros((self.height, self.width), dtype=np.uint16)
