from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

DEFAULT_DJI_INDEX = 0


class DJICamera:
    """DJI camera with threaded capture to prevent UI blocking and frame tearing."""

    def __init__(self, index: int, width: int, height: int):
        self.index = index
        self.width = width
        self.height = height
        self.capture: Optional[cv2.VideoCapture] = None
        self.available = False
        self.last_warn_time = 0.0

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        # 使用 V4L2 后端 (Linux)，回退到默认后端
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            self.available = False
            self.capture = None
            print(
                f"Warning: Unable to open DJI camera at index {self.index}. "
                "Using zero-filled frames."
            )
            return

        # 设置像素格式 (YUYV 用于 USB 3.0，MJPEG 用于 USB 2.0)
        # 先尝试 YUYV，如果失败则回退到 MJPEG
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc != cv2.VideoWriter_fourcc(*'YUYV'):
            # YUYV 不支持，尝试 MJPEG
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # 设置分辨率和帧率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 最小化缓冲区，获取最新帧
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.capture = cap
        self.available = True

        # 打印实际配置
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fmt = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"DJI camera: {fmt} {actual_w}x{actual_h} @{actual_fps:.0f}fps")

        # Warmup: discard first few frames to let camera stabilize
        for _ in range(5):
            cap.grab()

        # Start background capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        """Background thread: continuously grabs frames at camera native rate."""
        while self._running:
            if self.capture is None:
                break
            ok = self.capture.grab()
            if not ok:
                time.sleep(0.01)
                continue
            ok, frame = self.capture.retrieve()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame.copy()

    def read(self) -> np.ndarray:
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
        return self._zero_frame()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.available = False

    def _zero_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
