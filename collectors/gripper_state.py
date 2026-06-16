from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Optional

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from collectors.robot_arm import DEFAULT_ARM_HOST, DEFAULT_ARM_PORT
from timestamp_utils import get_timestamp_us

GRIPPER_FPS = 120  # 目标频率（SDK 实测 ~122Hz 自由运行）
GRIPPER_INTERVAL_S = 0.0  # 不限速：与机械臂共享 SDK 连接，自由运行靠时间戳对齐
GRIPPER_BATCH_SIZE = 100
GRIPPER_FLUSH_INTERVAL_S = 0.1


class GripperStateCollector:
    """Collect gripper state through RM Plus into robot_state/gripper_state.csv.

    RM Plus returns real-time end-tool state. For this gripper, pos[0] is the
    gripper opening value requested by the user.
    """

    def __init__(
        self,
        host: str,
        port: int,
        interval_s: float = GRIPPER_INTERVAL_S,
        batch_size: int = GRIPPER_BATCH_SIZE,
        flush_interval_s: float = GRIPPER_FLUSH_INTERVAL_S,
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

        self.latest_state: dict = {
            "code": -1,
            "gripper_pos": None,
            "latency_ms": None,
        }
        self.lock = threading.Lock()

    def connect(self) -> None:
        self.handle = self.robot.rm_create_robot_arm(self.host, self.port)
        if self.handle is None or getattr(self.handle, "id", -1) < 0:
            raise RuntimeError("Failed to create gripper RM Plus robot arm handle.")
        print(f"Gripper RM Plus handle ID: {self.handle.id}")

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        print(f"Gripper state collector started at target {GRIPPER_FPS}Hz")

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
        csv_path = robot_dir / "gripper_state.csv"

        with self.lock:
            self._stop_session_locked()
            self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            headers = [
                "timestamp_us",
                "target_hz",
                "rm_plus_read_code",
                "rm_plus_read_latency_ms",
                "sys_state",
                "gripper_pos",
                "gripper_speed",
                "gripper_current",
                "gripper_force",
                "gripper_dof_state",
                "gripper_dof_err",
                "deadline_late_ms",
            ]
            self.csv_writer.writerow(headers)
            self.row_buffer = []
            self.last_flush_time = time.time()
            self.recording = True

        print(f"Gripper state session file: {csv_path}")

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

    @staticmethod
    def _first(values: object) -> Optional[int]:
        if isinstance(values, list) and values:
            return values[0]
        return None

    def _poll_loop(self) -> None:
        """轮询夹爪状态。SDK 自由运行 ~122Hz，靠时间戳对齐到 30Hz 视觉帧。"""
        last_timestamp_us = 0  # 去重：跳过相同时间戳

        while self.running:
            if self.handle is None:
                time.sleep(0.1)
                continue

            read_start = time.perf_counter()
            try:
                code, data = self.robot.rm_get_rm_plus_state_info()
            except Exception:
                code, data = -1, {}
            read_latency_ms = (time.perf_counter() - read_start) * 1000.0

            payload = data if code == 0 and isinstance(data, dict) else {}
            gripper_pos = self._first(payload.get("pos"))
            gripper_speed = self._first(payload.get("speed"))
            gripper_current = self._first(payload.get("current"))
            gripper_force = self._first(payload.get("force"))
            gripper_dof_state = self._first(payload.get("dof_state"))
            gripper_dof_err = self._first(payload.get("dof_err"))
            sys_state = payload.get("sys_state", "")

            # 使用高精度单调时间戳
            timestamp_us = get_timestamp_us()

            # 去重：跳过相同时间戳
            if timestamp_us == last_timestamp_us:
                continue
            last_timestamp_us = timestamp_us

            with self.lock:
                self.latest_state = {
                    "code": code,
                    "gripper_pos": gripper_pos,
                    "sys_state": sys_state,
                    "latency_ms": read_latency_ms,
                }

                if self.recording and self.csv_writer is not None:
                    row = [
                        timestamp_us,
                        GRIPPER_FPS,
                        code,
                        read_latency_ms,
                        sys_state,
                        gripper_pos if gripper_pos is not None else "",
                        gripper_speed if gripper_speed is not None else "",
                        gripper_current if gripper_current is not None else "",
                        gripper_force if gripper_force is not None else "",
                        gripper_dof_state if gripper_dof_state is not None else "",
                        gripper_dof_err if gripper_dof_err is not None else "",
                        0.0,  # late_ms 不再适用
                    ]
                    self.row_buffer.append(row)
                    self._flush_locked(force=False)

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
