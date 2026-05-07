"""herong_9_pressure_data.py

Listen for the pressure UDP packets and render a live terminal + matplotlib
dashboard for the selected channels.

Usage:
    python herong_9_pressure_data.py

The script expects the packet format <Q64h> (uint64 timestamp_us + 64 int16).
It does a single HELLO handshake to the remote device and then updates a
fixed terminal block and a matplotlib window for every received packet.
Ctrl+C stops the program and cleans up the socket.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import os
import socket
import struct
import sys

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# UDP / packet configuration
UDP_PORT = 4321
PACKET_FORMAT = "<Q64h"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
BUFFER_SIZE = 136
PACKET_VALUE_COUNT = 64

# Remote device for handshake (adjust if needed)
REMOTE_IP = "192.168.31.164"
REMOTE_PORT = 2222

UNIT_LABEL = "a.u."
COLORMAP_NAME = "Blues"
SOCKET_TIMEOUT_S = 1.0

LEFT_CHANNEL = 18
RIGHT_CHANNEL = 19
LEFT_MATRIX_CHANNELS = [
    [17, 32, 31],
    [30, 29, 28],
    [27, 26, 25],
]
RIGHT_MATRIX_CHANNELS = [
    [1, 16, 15],
    [14, 13, 12],
    [11, 10, 9],
]


@dataclass
class DashboardState:
    fig: Any
    left_ax: Any
    right_ax: Any
    left_matrix_ax: Any
    right_matrix_ax: Any
    left_bar: Any
    right_bar: Any
    left_bar_text: Any
    right_bar_text: Any
    left_image: Any
    right_image: Any
    left_matrix_texts: list[list[Any]]
    right_matrix_texts: list[list[Any]]
    color_mappable: ScalarMappable
    colorbar: Any


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
            enable_vt_processing = 0x0004
            new_mode = mode.value | enable_vt_processing
            kernel32.SetConsoleMode(hStdOut, new_mode)
    except Exception:
        # best-effort; if it fails, Windows may still support ANSI or fallback will be used
        pass


def get_channel_value(values: list[int], channel_number: int) -> int:
    """Return a 1-based channel value, or 0 if the index is not available."""
    index = channel_number - 1
    if 0 <= index < len(values):
        return values[index]
    return 0


def extract_matrix(values: list[int], channel_layout: list[list[int]]) -> list[list[int]]:
    """Map a 3x3 channel layout to a 3x3 value layout."""
    return [[get_channel_value(values, channel) for channel in row] for row in channel_layout]


def format_matrix_rows(values_matrix: list[list[int]], channels_matrix: list[list[int]]) -> list[str]:
    """Format one matrix as three aligned terminal rows."""
    rows = []
    for channel_row, value_row in zip(channels_matrix, values_matrix):
        parts = [f"CH{channel:02d}: {value:6d}" for channel, value in zip(channel_row, value_row)]
        rows.append("  " + " | ".join(parts))
    return rows


def format_terminal_snapshot(values: list[int]) -> list[str]:
    """Return the terminal block for the selected channels only."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    left_value = get_channel_value(values, LEFT_CHANNEL)
    right_value = get_channel_value(values, RIGHT_CHANNEL)
    left_matrix_values = extract_matrix(values, LEFT_MATRIX_CHANNELS)
    right_matrix_values = extract_matrix(values, RIGHT_MATRIX_CHANNELS)

    lines = [
        f"Timestamp: {ts}",
        f"Unit: {UNIT_LABEL}",
        f"Left (CH{LEFT_CHANNEL:02d}): {left_value:6d}",
        f"Right (CH{RIGHT_CHANNEL:02d}): {right_value:6d}",
        "Left Matrix:",
    ]
    lines.extend(format_matrix_rows(left_matrix_values, LEFT_MATRIX_CHANNELS))
    lines.append("Right Matrix:")
    lines.extend(format_matrix_rows(right_matrix_values, RIGHT_MATRIX_CHANNELS))
    return lines


def move_cursor_up(n: int) -> None:
    # ANSI escape: move cursor up n lines
    sys.stdout.write(f"\x1b[{n}A")


def clear_and_write_lines(lines: list[str]) -> None:
    """Overwrite a previously printed terminal block in place."""
    for line in lines:
        sys.stdout.write("\x1b[2K" + line + "\n")
    sys.stdout.flush()


def flatten_matrix(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def create_dashboard() -> DashboardState:
    """Build the live matplotlib dashboard."""
    plt.ion()
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    manager = fig.canvas.manager
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("Pressure Monitor")

    grid = fig.add_gridspec(2, 2, height_ratios=(0.9, 1.1))
    left_ax = fig.add_subplot(grid[0, 0])
    right_ax = fig.add_subplot(grid[0, 1])
    left_matrix_ax = fig.add_subplot(grid[1, 0])
    right_matrix_ax = fig.add_subplot(grid[1, 1])

    color_mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=COLORMAP_NAME)
    color_mappable.set_array([])

    def setup_bar_axis(ax: Any, title: str) -> None:
        ax.set_title(title)
        ax.set_xlabel(f"Signal Value ({UNIT_LABEL})")
        ax.set_yticks([])
        ax.set_ylim(-0.75, 0.75)
        ax.grid(True, axis="x", linestyle="--", alpha=0.25)
        ax.axvline(0, color="#4c4c4c", linewidth=0.8, alpha=0.4)

    def setup_matrix_axis(ax: Any, title: str) -> tuple[Any, list[list[Any]]]:
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        image = ax.imshow(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            cmap=COLORMAP_NAME,
            norm=Normalize(vmin=0, vmax=1),
            interpolation="nearest",
        )
        text_rows: list[list[Any]] = []
        for row_index in range(3):
            row_texts = []
            for col_index in range(3):
                row_texts.append(
                    ax.text(
                        col_index,
                        row_index,
                        "0",
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                    )
                )
            text_rows.append(row_texts)
        return image, text_rows

    setup_bar_axis(left_ax, f"Left (CH{LEFT_CHANNEL:02d})")
    setup_bar_axis(right_ax, f"Right (CH{RIGHT_CHANNEL:02d})")

    left_bar = left_ax.barh(0, 0, height=0.55, color=color_mappable.to_rgba(0), edgecolor="#36517a")[0]
    right_bar = right_ax.barh(0, 0, height=0.55, color=color_mappable.to_rgba(0), edgecolor="#36517a")[0]
    left_bar_text = left_ax.text(0, 0, "0", ha="left", va="center", fontsize=11, fontweight="bold")
    right_bar_text = right_ax.text(0, 0, "0", ha="left", va="center", fontsize=11, fontweight="bold")

    left_image, left_matrix_texts = setup_matrix_axis(left_matrix_ax, "Left Matrix (3x3)")
    right_image, right_matrix_texts = setup_matrix_axis(right_matrix_ax, "Right Matrix (3x3)")

    colorbar = fig.colorbar(
        color_mappable,
        ax=[left_ax, right_ax, left_matrix_ax, right_matrix_ax],
        shrink=0.92,
        pad=0.02,
    )
    colorbar.set_label(f"Signal Magnitude ({UNIT_LABEL})")

    fig.suptitle("Pressure Monitor", fontsize=15, fontweight="bold")

    return DashboardState(
        fig=fig,
        left_ax=left_ax,
        right_ax=right_ax,
        left_matrix_ax=left_matrix_ax,
        right_matrix_ax=right_matrix_ax,
        left_bar=left_bar,
        right_bar=right_bar,
        left_bar_text=left_bar_text,
        right_bar_text=right_bar_text,
        left_image=left_image,
        right_image=right_image,
        left_matrix_texts=left_matrix_texts,
        right_matrix_texts=right_matrix_texts,
        color_mappable=color_mappable,
        colorbar=colorbar,
    )


def update_bar(axis: Any, patch: Any, text_artist: Any, value: int, peak: float, x_min: float, x_max: float, state: DashboardState) -> None:
    """Update one horizontal bar and its numeric label."""
    magnitude = abs(value)
    patch.set_width(value)
    patch.set_color(state.color_mappable.to_rgba(magnitude))

    offset = max(peak * 0.04, 1.0)
    if value >= 0:
        text_x = min(float(value) + offset, x_max - offset)
        text_align = "left"
    else:
        text_x = max(float(value) - offset, x_min + offset)
        text_align = "right"

    text_artist.set_position((text_x, 0))
    text_artist.set_text(f"{value:d}")
    text_artist.set_ha(text_align)
    text_artist.set_color("white" if magnitude / peak >= 0.55 else "black")
    axis.set_xlim(x_min, x_max)


def update_matrix(image: Any, text_rows: list[list[Any]], values_matrix: list[list[int]], peak: float, state: DashboardState) -> None:
    """Update one 3x3 matrix image and its per-cell labels."""
    magnitude_matrix = [[abs(value) for value in row] for row in values_matrix]
    image.set_data(magnitude_matrix)
    image.set_clim(0, peak)

    for row_index, row in enumerate(values_matrix):
        for col_index, value in enumerate(row):
            magnitude = abs(value)
            text_rows[row_index][col_index].set_text(f"{value:d}")
            text_rows[row_index][col_index].set_color("white" if magnitude / peak >= 0.55 else "black")


def update_dashboard(state: DashboardState, values: list[int]) -> None:
    """Push one packet into the terminal and matplotlib views."""
    left_value = get_channel_value(values, LEFT_CHANNEL)
    right_value = get_channel_value(values, RIGHT_CHANNEL)
    left_matrix_values = extract_matrix(values, LEFT_MATRIX_CHANNELS)
    right_matrix_values = extract_matrix(values, RIGHT_MATRIX_CHANNELS)

    selected_values = [left_value, right_value] + flatten_matrix(left_matrix_values) + flatten_matrix(right_matrix_values)
    peak = float(max(1, max(abs(value) for value in selected_values)))

    has_negative = any(value < 0 for value in selected_values)
    margin = max(peak * 0.12, 2.0)
    if has_negative:
        x_min = -(peak + margin)
        x_max = peak + margin
    else:
        x_min = 0.0
        x_max = peak + margin * 2.0

    state.color_mappable.set_clim(0, peak)
    state.colorbar.update_normal(state.color_mappable)

    update_bar(state.left_ax, state.left_bar, state.left_bar_text, left_value, peak, x_min, x_max, state)
    update_bar(state.right_ax, state.right_bar, state.right_bar_text, right_value, peak, x_min, x_max, state)
    update_matrix(state.left_image, state.left_matrix_texts, left_matrix_values, peak, state)
    update_matrix(state.right_image, state.right_matrix_texts, right_matrix_values, peak, state)

    state.fig.canvas.draw_idle()
    state.fig.canvas.flush_events()
    plt.pause(0.001)


def main() -> None:
    enable_vt_mode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(SOCKET_TIMEOUT_S)

    # handshake
    try:
        sock.sendto(b"HELLO", (REMOTE_IP, REMOTE_PORT))
    except Exception:
        pass

    print(f"Sent handshake to {REMOTE_IP}:{REMOTE_PORT}")
    print(f"Listening on UDP port {UDP_PORT}... (Ctrl+C to quit)")

    dashboard = create_dashboard()
    plt.show(block=False)

    zeros = [0] * PACKET_VALUE_COUNT
    initial_lines = format_terminal_snapshot(zeros)
    clear_and_write_lines(initial_lines)
    display_line_count = len(initial_lines)

    try:
        while True:
            if not plt.fignum_exists(dashboard.fig.number):
                print("Matplotlib window closed.")
                break

            try:
                # Wait for the first packet via blocking (with timeout)
                # to prevent looping/consuming 100% CPU when no data
                data, addr = sock.recvfrom(BUFFER_SIZE)
                
                # Drain the UDP socket to only process the latest packet
                sock.setblocking(False)
                while True:
                    try:
                        next_data, addr = sock.recvfrom(BUFFER_SIZE)
                        data = next_data
                    except (BlockingIOError, socket.timeout):
                        break
                
                # Reset to blocking with timeout
                sock.setblocking(True)
                sock.settimeout(SOCKET_TIMEOUT_S)
                
            except socket.timeout:
                plt.pause(0.001)
                continue

            if len(data) < PACKET_SIZE:
                # ignore short packets
                continue

            timestamp_us, *values = struct.unpack(PACKET_FORMAT, data)
            if len(values) < PACKET_VALUE_COUNT:
                values = values + [0] * (PACKET_VALUE_COUNT - len(values))

            _ = timestamp_us
            lines = format_terminal_snapshot(values)
            update_dashboard(dashboard, values)

            move_cursor_up(display_line_count)
            clear_and_write_lines(lines)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        plt.close(dashboard.fig)
        sock.close()


if __name__ == "__main__":
    main()
