"""Pure data-contract helpers for Franka simulation collection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


CONTACT_STAGE_BY_PHASE = {
    0: "idle",
    1: "precontact",
    2: "contact_onset",
    3: "lift_grasp",
    4: "transport_grasp",
    5: "placement_grasp",
    6: "release",
    7: "idle",
}

_LEFT_MATRIX_WEIGHTS = (0.45, 0.65, 0.50, 0.68, 1.00, 0.72, 0.48, 0.67, 0.52)
_RIGHT_MATRIX_WEIGHTS = (0.50, 0.65, 0.45, 0.72, 1.00, 0.68, 0.52, 0.67, 0.48)
_CHANNEL_WEIGHTS = (1.00, 0.96) + _LEFT_MATRIX_WEIGHTS + _RIGHT_MATRIX_WEIGHTS


def physics_gpu_index(*, active_gpu: int, override: int | None) -> int:
    """Resolve the CUDA/PhysX index independently from the Vulkan GPU index."""
    return int(active_gpu if override is None else override)


def simulation_device(device: str, *, active_gpu: int) -> str:
    """Return an explicit compute device so physics follows the selected GPU."""
    return f"cuda:{int(active_gpu)}" if device == "cuda" else device


def should_render_for_camera_stream(
    *,
    headless: bool,
    render_every_step: bool,
    save_visual_data: bool,
) -> bool:
    """Keep Isaac camera annotators current while collecting visual data."""
    return not headless or render_every_step or save_visual_data


@dataclass
class FixedRateSchedule:
    """Emit deterministic timestamps for an independent fixed-rate stream."""

    target_hz: float
    next_sample_index: int = 1

    def __post_init__(self) -> None:
        if self.target_hz <= 0:
            raise ValueError("target_hz must be positive")

    def due_timestamps(self, epoch_us: int, elapsed_s: float) -> list[int]:
        due: list[int] = []
        epsilon = 1e-12
        while self.next_sample_index / self.target_hz <= elapsed_s + epsilon:
            offset_us = round(self.next_sample_index * 1_000_000.0 / self.target_hz)
            due.append(int(epoch_us) + int(offset_us))
            self.next_sample_index += 1
        return due


def _as_quaternion_wxyz(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=float).reshape(-1)
    if quat.size < 4:
        raise ValueError(f"Expected quaternion [w, x, y, z], got {quat.size} values")
    quat = quat[:4]
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be positive")
    return quat / norm


def quaternion_wxyz_to_euler_xyz(value: np.ndarray) -> np.ndarray:
    """Convert a quaternion to roll, pitch, yaw in radians."""
    w, x, y, z = _as_quaternion_wxyz(value)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=float)


def pose6_in_base_frame(
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    env_origin: np.ndarray,
) -> np.ndarray:
    """Return xyz in the environment base frame plus Euler XYZ radians."""
    xyz = np.asarray(position, dtype=float).reshape(-1)[:3]
    origin = np.asarray(env_origin, dtype=float).reshape(-1)[:3]
    if xyz.size != 3 or origin.size != 3:
        raise ValueError("Position and environment origin must each contain xyz")
    return np.concatenate([xyz - origin, quaternion_wxyz_to_euler_xyz(orientation_wxyz)])


def unwrap_euler_xyz(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Choose the equivalent Euler angles nearest to the previous sample."""
    current_values = np.asarray(current, dtype=float).reshape(-1)[:3]
    previous_values = np.asarray(previous, dtype=float).reshape(-1)[:3]
    if current_values.size != 3 or previous_values.size != 3:
        raise ValueError("Current and previous Euler values must each contain xyz")
    delta = (current_values - previous_values + math.pi) % (2.0 * math.pi) - math.pi
    return previous_values + delta


def contact_stage_for_phase(phase_id: int) -> str:
    return CONTACT_STAGE_BY_PHASE.get(int(phase_id), "idle")


def _stage_drop(stage: str, max_drop_adc: float, progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    fractions = {
        "idle": 0.0,
        "lift_grasp": 0.72,
        "transport_grasp": 0.59,
        "placement_grasp": 0.84,
    }
    if stage == "contact_onset":
        return max_drop_adc * (0.10 + 0.48 * progress)
    if stage == "release":
        return max_drop_adc * 0.59 * (1.0 - progress)
    return max_drop_adc * fractions.get(stage, 0.0)


def pressure_values_for_stage(
    stage: str,
    channels: Sequence[int],
    *,
    baseline_adc: int = 5000,
    max_drop_adc: int = 3200,
    progress: float = 0.0,
    sample_index: int = 0,
) -> list[int]:
    """Generate a reproducible 20-channel tactile profile for a task stage."""
    if len(channels) != 20:
        raise ValueError(f"Expected 20 pressure channels, got {len(channels)}")
    amplitude = _stage_drop(stage, float(max_drop_adc), progress)
    values: list[int] = []
    for index, weight in enumerate(_CHANNEL_WEIGHTS):
        baseline_jitter = 7.0 * math.sin((sample_index + 1) * 0.37 + index * 0.83)
        contact_modulation = 1.0 + 0.025 * math.sin(sample_index * 0.19 + index * 0.61)
        raw = float(baseline_adc) + baseline_jitter - amplitude * weight * contact_modulation
        values.append(int(np.clip(round(raw), 0, 5007)))

    channel_index = {int(channel): index for index, channel in enumerate(channels)}
    if stage == "precontact":
        approach_progress = float(np.clip(progress, 0.0, 1.0))
        if 51 in channel_index:
            values[channel_index[51]] = round(4500.0 - 1000.0 * approach_progress)
        if 50 in channel_index:
            values[channel_index[50]] = round(4480.0 - 950.0 * approach_progress)
    if all(channel in channel_index for channel in (58, 61, 57, 59)):
        adjacent = [values[channel_index[channel]] for channel in (61, 57, 59)]
        values[channel_index[58]] = round(sum(adjacent) / len(adjacent))
    return values
