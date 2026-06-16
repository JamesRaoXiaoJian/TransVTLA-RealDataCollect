from __future__ import annotations

import csv
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Optional

from timestamp_utils import get_timestamp_us
from channel_config import PRESSURE_VALUE_COLUMNS, VALID_CHANNELS

TACTILE_FPS = 200

DEFAULT_PRESSURE_LOCAL_PORT = 4321
DEFAULT_PRESSURE_REMOTE_IP = "192.168.31.164"
DEFAULT_PRESSURE_REMOTE_PORT = 2222
RAW_PRESSURE_CHANNEL_COUNT = 64
RAW_PRESSURE_PACKET_FORMAT = f"<Q{RAW_PRESSURE_CHANNEL_COUNT}h"
RAW_PRESSURE_PACKET_SIZE = struct.calcsize(RAW_PRESSURE_PACKET_FORMAT)
PRESSURE_BUFFER_SIZE = RAW_PRESSURE_PACKET_SIZE
PRESSURE_BATCH_SIZE = 100
PRESSURE_FLUSH_INTERVAL_S = 0.1


class PressureCollector:
    def __init__(
        self,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        timeout_s: float = 1.0,
        batch_size: int = PRESSURE_BATCH_SIZE,
        flush_interval_s: float = PRESSURE_FLUSH_INTERVAL_S,
    ):
        self.local_port = local_port
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.timeout_s = timeout_s
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s

        self.sock: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False

        self.recording = False
        self.csv_file = None
        self.csv_writer: Optional[csv.writer] = None
        self.row_buffer: list[list[int]] = []

        self.latest_timestamp_us: Optional[int] = None
        self.latest_values: list[int] = [0] * len(VALID_CHANNELS)

        self.last_flush_time = time.time()

        self.lock = threading.Lock()

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.local_port))
        sock.settimeout(self.timeout_s)
        sock.sendto(b"HELLO", (self.remote_ip, self.remote_port))

        self.sock = sock
        self.running = True
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

        print(f"Pressure UDP listening on {self.local_port}")
        print(f"Pressure handshake sent to {self.remote_ip}:{self.remote_port}")

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

        self.stop_session()

        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def start_session(self, session_root: Path) -> None:
        pressure_dir = session_root / "pressure"
        pressure_dir.mkdir(parents=True, exist_ok=True)
        csv_path = pressure_dir / "pressure.csv"

        with self.lock:
            self._stop_session_locked()
            self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            # 双时间戳：sensor_timestamp_us（传感器时钟）+ host_monotonic_us（主机单调时钟）。
            # 标准数据只保存建模使用的 20 个有效触觉通道。
            headers = ["sensor_timestamp_us", "host_monotonic_us"] + PRESSURE_VALUE_COLUMNS
            self.csv_writer.writerow(headers)
            self.row_buffer = []
            self.last_flush_time = time.time()
            self.recording = True

        print(f"Pressure session file: {csv_path}")

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

    def get_latest_values(self) -> list[int]:
        with self.lock:
            return list(self.latest_values)

    def _recv_loop(self) -> None:
        while self.running:
            if self.sock is None:
                break

            try:
                data, addr = self.sock.recvfrom(PRESSURE_BUFFER_SIZE)
            except socket.timeout:
                with self.lock:
                    self._flush_locked(force=False)
                continue
            except OSError:
                break

            if len(data) < RAW_PRESSURE_PACKET_SIZE:
                print(f"[Pressure {addr}] Packet too small: {len(data)} bytes")
                continue

            sensor_timestamp_us, *raw_values = struct.unpack(RAW_PRESSURE_PACKET_FORMAT, data)

            # 记录主机单调时间戳（用于时钟域对齐）
            host_monotonic_us = get_timestamp_us()
            selected_values = [raw_values[ch - 1] for ch in VALID_CHANNELS]

            with self.lock:
                self.latest_timestamp_us = sensor_timestamp_us
                self.latest_values = selected_values

                if self.recording and self.csv_writer is not None:
                    row = [sensor_timestamp_us, host_monotonic_us] + selected_values
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
