from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


class RealSenseRGB:
    """RealSense 相机采集器（后台线程版本）。

    优化点：
    - 后台线程持续采集，read() 非阻塞
    - 深度滤波可选（默认关闭，离线处理）
    - 关闭自动曝光，保证训练数据一致性
    - 默认 848x480@30fps
    """

    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        enable_filters: bool = False,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.enable_filters = enable_filters

        self.pipeline: Optional[object] = None
        self.align: Optional[object] = None
        self.available = False
        self.last_warn_time = 0.0

        self._last_color: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if rs is None:
            self.pipeline = None
            self.available = False
            print("Warning: pyrealsense2 not installed. Using zero-filled RealSense frames.")
            return

        pipeline = rs.pipeline()
        config = rs.config()

        # 配置 color 流
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )

        # 配置 depth 流（可选）
        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )

        try:
            pipeline_profile = pipeline.start(config)
        except RuntimeError as exc:
            self.pipeline = None
            self.available = False
            print(
                "Warning: RealSense pipeline could not start. "
                "Using zero-filled frames. Details: "
                f"{exc}"
            )
            return

        # 调整传感器设置
        device = pipeline_profile.get_device()

        # 关闭自动曝光（训练数据一致性）
        try:
            depth_sensor = device.first_depth_sensor()
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        except Exception:
            pass

        try:
            color_sensor = device.first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            # 手动曝光（根据环境光调整）
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, 8.0)  # 8ms
        except Exception:
            pass

        self.pipeline = pipeline
        if self.enable_depth:
            self.align = rs.align(rs.stream.color)

        # 启动后台采集线程
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.available = True

        print(f"RealSense: {self.width}x{self.height} @{self.fps}fps, depth={self.enable_depth}")

    def _capture_loop(self) -> None:
        """后台线程：持续采集帧，缓存最新帧。"""
        while self._running:
            if self.pipeline is None:
                break
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                if self.align:
                    frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_data = np.asanyarray(color_frame.get_data())

                depth_data = None
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        # 深度滤波可选（默认关闭，离线处理）
                        if self.enable_filters:
                            depth_frame = rs.spatial_filter().process(depth_frame)
                            depth_frame = rs.temporal_filter().process(depth_frame)
                            depth_frame = rs.hole_filling_filter().process(depth_frame)
                        depth_data = np.asanyarray(depth_frame.get_data())

                with self._lock:
                    self._last_color = color_data
                    self._last_depth = depth_data

            except Exception as e:
                if self._running:
                    time.sleep(0.01)

    def read(self) -> np.ndarray:
        """非阻塞读取最新帧。"""
        with self._lock:
            if self._last_color is not None:
                return self._last_color.copy()
        return self._zero_frame()

    def read_depth(self) -> np.ndarray:
        """非阻塞读取最新深度帧。"""
        with self._lock:
            if self._last_depth is not None:
                return self._last_depth.copy()
        return self._zero_depth()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.align = None
        self._last_color = None
        self._last_depth = None
        self.available = False

    def _zero_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _zero_depth(self) -> np.ndarray:
        return np.zeros((self.height, self.width), dtype=np.uint16)
