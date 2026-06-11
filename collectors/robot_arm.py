from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Optional

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from timestamp_utils import get_timestamp_us

DEFAULT_ARM_HOST = "192.168.31.92"
DEFAULT_ARM_PORT = 8080
ROBOT_ARM_FPS = 100  # 网线直连后可达 ~100-200 Hz
ROBOT_ARM_INTERVAL_S = 0.0  # 不额外 sleep，让 SDK 调用耗时决定节奏
ROBOT_ARM_BATCH_SIZE = 100
ROBOT_ARM_FLUSH_INTERVAL_S = 0.1


class RobotArmCollector:
    """Collect robot arm state at 200Hz in a separate thread."""

    def __init__(
        self,
        host: str,
        port: int,
        interval_s: float = ROBOT_ARM_INTERVAL_S,
        batch_size: int = ROBOT_ARM_BATCH_SIZE,
        flush_interval_s: float = ROBOT_ARM_FLUSH_INTERVAL_S,
    ):
        self.host = host
        self.port = port
        self.interval_s = interval_s
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s

        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = None
        self.thread: Optional[threading.Thread] = None
        self.running = False

        self.recording = False
        self.csv_file = None
        self.csv_writer: Optional[csv.writer] = None
        self.row_buffer: list[list] = []
        self.last_flush_time = time.time()

        self.latest_state: dict = {"code": -1, "data": None}
        self.latest_joints: Optional[list[float]] = None
        self.latest_pose: Optional[list[float]] = None

        self.lock = threading.Lock()

    def connect(self) -> None:
        self.handle = self.robot.rm_create_robot_arm(self.host, self.port)
        if self.handle is None:
            raise RuntimeError("Failed to create robot arm handle.")
        print(f"机械臂ID： {self.handle.id}")

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        print(f"Robot arm collector started at {ROBOT_ARM_FPS}Hz")

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        self.stop_session()
        if self.handle is not None:
            self.robot.rm_delete_robot_arm()
            self.handle = None

    def start_session(self, session_root: Path) -> None:
        robot_dir = session_root / "robot_state"
        robot_dir.mkdir(parents=True, exist_ok=True)
        csv_path = robot_dir / "robot_state.csv"

        with self.lock:
            self._stop_session_locked()
            self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            headers = ["timestamp_us"]
            headers += [f"joint_{i+1}" for i in range(7)]
            headers += [f"pose_{i+1}" for i in range(6)]
            self.csv_writer.writerow(headers)
            self.row_buffer = []
            self.last_flush_time = time.time()
            self.recording = True

        print(f"Robot arm session file: {csv_path}")

    def stop_session(self) -> None:
        with self.lock:
            self._stop_session_locked()

    def _stop_session_locked(self) -> None:
        self._flush_locked(force=True)
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        self.csv_writer = None
        self.recording = False

    def get_latest_state(self) -> dict:
        with self.lock:
            return dict(self.latest_state)

    def get_latest_joints(self) -> Optional[list[float]]:
        with self.lock:
            return list(self.latest_joints) if self.latest_joints else None

    def get_latest_pose(self) -> Optional[list[float]]:
        with self.lock:
            return list(self.latest_pose) if self.latest_pose else None

    def _poll_loop(self) -> None:
        """轮询机械臂状态。网线直连时单次调用 ~1-5ms，可达 ~100-200 Hz。"""
        while self.running:
            if self.handle is None:
                time.sleep(0.1)
                continue

            try:
                status = self.robot.rm_get_current_arm_state()
                if not isinstance(status, tuple) or len(status) != 2:
                    state = {"code": -1, "data": None}
                else:
                    code, payload = status
                    state = {"code": code, "data": payload}
            except Exception:
                state = {"code": -1, "data": None}

            # 使用高精度单调时间戳
            timestamp_us = get_timestamp_us()

            payload = state.get("data") if isinstance(state, dict) else None
            joints = payload.get("joint") if isinstance(payload, dict) else None
            pose = payload.get("pose") if isinstance(payload, dict) else None

            with self.lock:
                self.latest_state = state
                self.latest_joints = list(joints) if joints else None
                self.latest_pose = list(pose) if pose else None

                if self.recording and self.csv_writer is not None:
                    row = [timestamp_us]
                    row += list(joints) if joints else [0] * 7
                    row += list(pose) if pose else [0] * 6
                    self.row_buffer.append(row)
                    self._flush_locked(force=False)

            # 不 sleep，让 SDK 调用耗时决定节奏

    def _flush_locked(self, force: bool) -> None:
        if not self.recording or self.csv_writer is None or self.csv_file is None:
            return
        if not self.row_buffer:
            return

        now = time.time()
        should_flush = force or len(self.row_buffer) >= self.batch_size or (
            now - self.last_flush_time
        ) >= self.flush_interval_s

        if not should_flush:
            return

        self.csv_writer.writerows(self.row_buffer)
        self.csv_file.flush()
        self.row_buffer.clear()
        self.last_flush_time = now
