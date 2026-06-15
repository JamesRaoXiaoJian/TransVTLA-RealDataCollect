from __future__ import annotations

import copy
import threading
import time
from typing import Optional

import numpy as np
from realsense_standard import (
    DEPTH_PNG_UNIT,
    DEPTH_PNG_UNIT_M,
    STANDARD_RS_FPS,
    STANDARD_RS_HEIGHT,
    STANDARD_RS_WIDTH,
    standard_realsense_profile,
)

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


class RealSenseRGB:
    """RealSense RGB-D camera collector.

    The class name is kept for backward compatibility with older scripts.
    """

    def __init__(
        self,
        width: int = STANDARD_RS_WIDTH,
        height: int = STANDARD_RS_HEIGHT,
        fps: int = STANDARD_RS_FPS,
        enable_depth: bool = True,
        enable_filters: bool = False,
        serial_number: str | None = None,
        name: str = "RealSense",
        enabled: bool = True,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.enable_filters = enable_filters
        self.serial_number = serial_number
        self.name = name
        self.enabled = enabled

        self.pipeline: Optional[object] = None
        self.align: Optional[object] = None
        self.available = False
        self.last_warn_time = 0.0
        self.depth_scale_m = DEPTH_PNG_UNIT_M
        self.metadata: dict = self._build_metadata(available=False)

        self._last_color: Optional[np.ndarray] = None
        self._last_depth_raw: Optional[np.ndarray] = None
        self._frame_count = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @staticmethod
    def list_devices() -> list[dict[str, str]]:
        if rs is None:
            return []
        devices: list[dict[str, str]] = []
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                serial = dev.get_info(rs.camera_info.serial_number)
                name = dev.get_info(rs.camera_info.name)
                devices.append({"serial": serial, "name": name})
        except Exception:
            return []
        return devices

    @staticmethod
    def resolve_serial_pair(
        world_serial: str | None = None,
        wrist_serial: str | None = None,
    ) -> tuple[str | None, str | None, list[dict[str, str]]]:
        devices = RealSenseRGB.list_devices()
        serials = [dev["serial"] for dev in devices]

        if world_serial and wrist_serial and world_serial == wrist_serial:
            raise ValueError("world and wrist RealSense serial numbers must be different.")

        if world_serial is None:
            world_serial = next((serial for serial in serials if serial != wrist_serial), None)
        if wrist_serial is None:
            wrist_serial = next((serial for serial in serials if serial != world_serial), None)

        return world_serial, wrist_serial, devices

    def start(self) -> None:
        if not self.enabled:
            self.pipeline = None
            self.available = False
            self.metadata = self._build_metadata(available=False, note="disabled")
            print(f"Warning: {self.name} disabled. Using zero-filled RealSense frames.")
            return

        if rs is None:
            self.pipeline = None
            self.available = False
            self.metadata = self._build_metadata(available=False, note="pyrealsense2 not installed")
            print(f"Warning: pyrealsense2 not installed. Using zero-filled {self.name} frames.")
            return

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)

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
            self.metadata = self._build_metadata(available=False, note=str(exc))
            print(
                f"Warning: {self.name} pipeline could not start. "
                "Using zero-filled frames. Details: "
                f"{exc}"
            )
            return

        # 调整传感器设置
        device = pipeline_profile.get_device()
        self.metadata = self._metadata_from_profile(pipeline_profile, device)

        # 关闭自动曝光（训练数据一致性）
        try:
            depth_sensor = device.first_depth_sensor()
            self.depth_scale_m = float(depth_sensor.get_depth_scale())
            self.metadata["depth"]["sensor_depth_scale_m_per_unit"] = self.depth_scale_m
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

        serial_text = f", serial={self.serial_number}" if self.serial_number else ""
        print(f"{self.name}: {self.width}x{self.height} @{self.fps}fps, depth={self.enable_depth}{serial_text}")

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

                depth_data_raw = None
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        continue
                    # 深度滤波可选（默认关闭，离线处理）
                    if self.enable_filters:
                        depth_frame = rs.spatial_filter().process(depth_frame)
                        depth_frame = rs.temporal_filter().process(depth_frame)
                        depth_frame = rs.hole_filling_filter().process(depth_frame)
                    depth_data_raw = np.asanyarray(depth_frame.get_data())

                with self._lock:
                    self._last_color = color_data
                    self._last_depth_raw = depth_data_raw
                    self._frame_count += 1

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
        """Read latest aligned depth frame as uint16 millimeters."""
        raw = self.read_depth_raw()
        if self.depth_scale_m == DEPTH_PNG_UNIT_M:
            return raw
        depth_m = raw.astype(np.float32) * float(self.depth_scale_m)
        depth_mm = np.rint(depth_m / DEPTH_PNG_UNIT_M)
        return np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    def read_depth_raw(self) -> np.ndarray:
        """Read latest aligned RealSense z16 depth frame in sensor units."""
        with self._lock:
            if self._last_depth_raw is not None:
                return self._last_depth_raw.copy()
        return self._zero_depth()

    def get_frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def get_metadata(self) -> dict:
        with self._lock:
            return copy.deepcopy(self.metadata)

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
        self._last_depth_raw = None
        self._frame_count = 0
        self.available = False

    def _zero_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _zero_depth(self) -> np.ndarray:
        return np.zeros((self.height, self.width), dtype=np.uint16)

    def _build_metadata(self, available: bool, note: str | None = None) -> dict:
        metadata = {
            "name": self.name,
            "available": available,
            "serial_number": self.serial_number,
            "profile": {
                **standard_realsense_profile(),
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
            },
            "color": {"intrinsics": None},
            "depth": {
                "intrinsics": None,
                "aligned_to": "color" if self.enable_depth else None,
                "saved_pixel_intrinsics_source": "color" if self.enable_depth else None,
                "sensor_depth_scale_m_per_unit": self.depth_scale_m,
                "saved_png_unit": DEPTH_PNG_UNIT,
                "saved_png_unit_m": DEPTH_PNG_UNIT_M,
                "saved_png_dtype": "uint16",
            },
        }
        if note:
            metadata["note"] = note
        return metadata

    @staticmethod
    def _intrinsics_to_dict(intr) -> dict:
        return {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "model": str(intr.model),
            "coeffs": [float(v) for v in intr.coeffs],
            "K": [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
        }

    def _metadata_from_profile(self, pipeline_profile, device) -> dict:
        metadata = self._build_metadata(available=True)
        try:
            metadata["device"] = {
                "name": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
            }
            metadata["serial_number"] = metadata["device"]["serial_number"]
        except Exception:
            pass

        try:
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            color_intr = color_stream.as_video_stream_profile().get_intrinsics()
            metadata["color"]["intrinsics"] = self._intrinsics_to_dict(color_intr)
        except Exception:
            pass

        if self.enable_depth:
            try:
                depth_stream = pipeline_profile.get_stream(rs.stream.depth)
                depth_intr = depth_stream.as_video_stream_profile().get_intrinsics()
                metadata["depth"]["intrinsics"] = self._intrinsics_to_dict(depth_intr)
            except Exception:
                pass

        return metadata
