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
        self.align: Optional[object] = None
        self.spatial_filter: Optional[object] = None
        self.temporal_filter: Optional[object] = None
        self.hole_filling_filter: Optional[object] = None
        self.available = False
        self.last_warn_time = 0.0
        self._last_color: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None

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
        self.align = rs.align(rs.stream.color)
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        self.available = True

    def _grab_frames(self) -> None:
        """Grab one aligned frame pair, apply post-processing, cache results."""
        if self.pipeline is None:
            return
        frames = self.pipeline.wait_for_frames(timeout_ms=500)
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Missing RealSense frame.")
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.hole_filling_filter.process(depth_frame)
        self._last_color = np.asanyarray(color_frame.get_data())
        self._last_depth = np.asanyarray(depth_frame.get_data())

    def read(self) -> np.ndarray:
        if self.pipeline is None:
            return self._zero_frame()
        try:
            self._grab_frames()
            return self._last_color
        except Exception as exc:
            now = time.time()
            if (now - self.last_warn_time) >= 1.0:
                print(f"Warning: RealSense frame read failed. Using zero-filled frame. Details: {exc}")
                self.last_warn_time = now
            return self._zero_frame()

    def read_depth(self) -> np.ndarray:
        if self._last_depth is not None:
            return self._last_depth
        return self._zero_depth()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.align = None
        self.spatial_filter = None
        self.temporal_filter = None
        self.hole_filling_filter = None
        self._last_color = None
        self._last_depth = None
        self.available = False

    def _zero_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _zero_depth(self) -> np.ndarray:
        return np.zeros((self.height, self.width), dtype=np.uint16)
