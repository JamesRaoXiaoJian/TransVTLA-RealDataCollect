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
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.available = False
            self.capture = None
            print(
                f"Warning: Unable to open DJI camera at index {self.index}. "
                "Using zero-filled frames."
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Reduce internal buffer to 1 to always get the latest frame
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = cap
        self.available = True

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
