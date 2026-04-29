"""herong_9_pressure_data.py

Listen for the pressure UDP packets and refresh-display the 64 channels
in-place in the terminal (no accumulating new lines).

Usage:
    python herong_9_pressure_data.py

The script expects the packet format <Q64h> (uint64 timestamp_us + 64 int16).
It does a single HELLO handshake to the remote device and then prints a
fixed area that is updated for every received packet. Ctrl+C stops the
program and cleans up the socket.
"""

from __future__ import annotations

import socket
import struct
import sys
import time
import os
from datetime import datetime

# UDP / packet configuration
UDP_PORT = 4321
BUFFER_SIZE = 136
PACKET_FORMAT = "<Q64h"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# Remote device for handshake (adjust if needed)
REMOTE_IP = "192.168.31.164"
REMOTE_PORT = 2222

ROWS = 4
COLS = 8
DISPLAY_LINES = ROWS + 1  # header + rows

# Try enable VT processing on Windows consoles so ANSI escape sequences work
def enable_vt_mode() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        hStdOut = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode)):
            ENABLE_VT_PROCESSING = 0x0004
            new_mode = mode.value | ENABLE_VT_PROCESSING
            kernel32.SetConsoleMode(hStdOut, new_mode)
    except Exception:
        # best-effort; if it fails, Windows may still support ANSI or fallback will be used
        pass


def format_grid(values: list[int]) -> list[str]:
    """Return list of strings: header + ROWS text rows for display."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    header = f"Timestamp: {ts}"
    rows = []
    for r in range(ROWS):
        start = r * COLS
        block = values[start : start + COLS]
        # format each value as signed int, width 6
        vals = " ".join(f"{v:6d}" for v in block)
        rows.append(f"CH{start+1:02d}-{start+COLS:02d}: {vals}")
    return [header] + rows


def move_cursor_up(n: int) -> None:
    # ANSI escape: move cursor up n lines
    sys.stdout.write(f"\x1b[{n}A")


def main() -> None:
    enable_vt_mode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)

    # handshake
    try:
        sock.sendto(b"HELLO", (REMOTE_IP, REMOTE_PORT))
    except Exception:
        pass

    print(f"Sent handshake to {REMOTE_IP}:{REMOTE_PORT}")
    print(f"Listening on UDP port {UDP_PORT}... (Ctrl+C to quit)")
    # print initial blank grid
    zeros = [0] * (ROWS * COLS)
    for line in format_grid(zeros):
        print(line)
    sys.stdout.flush()

    first_draw = True

    try:
        while True:
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                # no packet, just continue (do not print anything)
                continue

            if len(data) < PACKET_SIZE:
                # ignore short packets
                continue

            timestamp_us, *values = struct.unpack(PACKET_FORMAT, data)
            # ensure we have 64 values
            if len(values) < ROWS * COLS:
                values = values + [0] * (ROWS * COLS - len(values))

            # prepare display lines
            lines = format_grid(values)

            if not first_draw:
                # move cursor up to overwrite previous grid
                move_cursor_up(DISPLAY_LINES)
            else:
                first_draw = False

            # write all lines and flush without adding extra blank lines
            for line in lines:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
