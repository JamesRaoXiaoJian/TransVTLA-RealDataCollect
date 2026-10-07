"""Physical Franka gripper-state encoding for raw simulation sessions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FRANKA_FINGER_DOF_INDICES = (7, 8)
FRANKA_MAX_OPENING_M = 0.08
LEGACY_GRIPPER_SCALE = 1000.0


@dataclass(frozen=True)
class GripperState:
    left_finger_m: float
    right_finger_m: float
    opening_m: float
    opening_normalized: float
    legacy_position: float


def gripper_state_from_dofs(dof_positions: np.ndarray) -> GripperState:
    """Encode the physical opening from Franka's two finger DOFs.

    ``legacy_position`` remains on the existing [0, 1000] raw-data scale so
    the RLDS builder can normalize it with its established divide-by-1000
    contract. The extra fields retain unambiguous SI-unit measurements.
    """
    values = np.asarray(dof_positions, dtype=float).reshape(-1)
    if values.size <= FRANKA_FINGER_DOF_INDICES[-1]:
        raise ValueError(f"Expected Franka finger DOFs at indices {FRANKA_FINGER_DOF_INDICES}, got {values.size} values")

    left_finger_m = float(values[FRANKA_FINGER_DOF_INDICES[0]])
    right_finger_m = float(values[FRANKA_FINGER_DOF_INDICES[1]])
    opening_m = left_finger_m + right_finger_m
    opening_normalized = float(np.clip(opening_m / FRANKA_MAX_OPENING_M, 0.0, 1.0))
    return GripperState(
        left_finger_m=left_finger_m,
        right_finger_m=right_finger_m,
        opening_m=opening_m,
        opening_normalized=opening_normalized,
        legacy_position=opening_normalized * LEGACY_GRIPPER_SCALE,
    )
