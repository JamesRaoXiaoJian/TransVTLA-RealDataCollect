"""Tests for physical Franka finger-DOF recording."""

from __future__ import annotations

import unittest

import numpy as np

from isaacsim.scripts.sim_gripper_state import gripper_state_from_dofs


class SimGripperStateTests(unittest.TestCase):
    def test_open_fingers_use_legacy_open_scale_and_physical_units(self) -> None:
        state = gripper_state_from_dofs(np.array([0.0] * 7 + [0.04, 0.04]))

        self.assertAlmostEqual(state.left_finger_m, 0.04)
        self.assertAlmostEqual(state.right_finger_m, 0.04)
        self.assertAlmostEqual(state.opening_m, 0.08)
        self.assertAlmostEqual(state.opening_normalized, 1.0)
        self.assertAlmostEqual(state.legacy_position, 1000.0)

    def test_closed_fingers_report_closed_state(self) -> None:
        state = gripper_state_from_dofs(np.array([0.0] * 9))

        self.assertAlmostEqual(state.opening_m, 0.0)
        self.assertAlmostEqual(state.legacy_position, 0.0)

    def test_invalid_dof_vector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finger DOFs"):
            gripper_state_from_dofs(np.zeros(8))


if __name__ == "__main__":
    unittest.main()
