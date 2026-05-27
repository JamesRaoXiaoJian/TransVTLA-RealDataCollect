from __future__ import annotations

import csv
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Optional

TACTILE_FPS = 200

DEFAULT_PRESSURE_LOCAL_PORT = 4321
DEFAULT_PRESSURE_REMOTE_IP = "192.168.31.164"
DEFAULT_PRESSURE_REMOTE_PORT = 2222
PRESSURE_PACKET_FORMAT = "<Q64h"
PRESSURE_PACKET_SIZE = struct.calcsize(PRESSURE_PACKET_FORMAT)
PRESSURE_BUFFER_SIZE = PRESSURE_PACKET_SIZE
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
        self.latest_values: list[int] = [0] * 64

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
            headers = ["timestamp_us"] + [f"CH{i+1}" for i in range(64)]
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

            if len(data) < PRESSURE_PACKET_SIZE:
                print(f"[Pressure {addr}] Packet too small: {len(data)} bytes")
                continue

            timestamp_us, *values = struct.unpack(PRESSURE_PACKET_FORMAT, data)

            now = time.time()
            with self.lock:
                self.latest_timestamp_us = timestamp_us
                self.latest_values = list(values)

                if self.recording and self.csv_writer is not None:
                    row = [timestamp_us] + self.latest_values
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
