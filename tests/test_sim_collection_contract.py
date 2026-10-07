"""Contract tests for simulation streams aligned to real collection data."""

from __future__ import annotations

import math
import unittest

import numpy as np

from isaacsim.scripts.sim_collection_contract import (
    FixedRateSchedule,
    contact_stage_for_phase,
    pose6_in_base_frame,
    physics_gpu_index,
    pressure_values_for_stage,
    simulation_device,
    should_render_for_camera_stream,
    unwrap_euler_xyz,
)


CHANNELS = [51, 50, 63, 60, 57, 64, 61, 58, 49, 62, 59, 47, 44, 41, 48, 45, 42, 33, 46, 43]


class CameraRenderContractTests(unittest.TestCase):
    def test_headless_visual_collection_renders_every_physics_step(self) -> None:
        self.assertTrue(
            should_render_for_camera_stream(
                headless=True,
                render_every_step=False,
                save_visual_data=True,
            )
        )


class SimulationDeviceContractTests(unittest.TestCase):
    def test_physics_gpu_can_use_a_cuda_visible_device_index(self) -> None:
        self.assertEqual(physics_gpu_index(active_gpu=1, override=0), 0)

    def test_physics_gpu_defaults_to_renderer_gpu(self) -> None:
        self.assertEqual(physics_gpu_index(active_gpu=1, override=None), 1)

    def test_cuda_device_uses_the_selected_gpu_index(self) -> None:
        self.assertEqual(simulation_device("cuda", active_gpu=1), "cuda:1")

    def test_cpu_device_does_not_include_a_gpu_index(self) -> None:
        self.assertEqual(simulation_device("cpu", active_gpu=1), "cpu")


class FixedRateScheduleTests(unittest.TestCase):
    def test_independent_streams_emit_exact_one_second_counts(self) -> None:
        robot = FixedRateSchedule(60.0)
        pressure = FixedRateSchedule(140.0)

        robot_times: list[int] = []
        pressure_times: list[int] = []
        for step in range(1, 61):
            elapsed_s = step / 60.0
            robot_times.extend(robot.due_timestamps(1_000_000, elapsed_s))
            pressure_times.extend(pressure.due_timestamps(1_000_000, elapsed_s))

        self.assertEqual(len(robot_times), 60)
        self.assertEqual(len(pressure_times), 140)
        self.assertTrue(all(b > a for a, b in zip(robot_times, robot_times[1:])))
        self.assertTrue(all(b > a for a, b in zip(pressure_times, pressure_times[1:])))


class PoseContractTests(unittest.TestCase):
    def test_pose_is_relative_to_environment_origin_and_uses_euler_xyz(self) -> None:
        half = math.sqrt(0.5)
        pose = pose6_in_base_frame(
            position=np.array([[7.5, 3.0, 0.4]]),
            orientation_wxyz=np.array([[half, 0.0, 0.0, half]]),
            env_origin=np.array([7.0, 3.5, 0.0]),
        )

        np.testing.assert_allclose(pose[:3], [0.5, -0.5, 0.4], atol=1e-7)
        np.testing.assert_allclose(pose[3:], [0.0, 0.0, math.pi / 2.0], atol=1e-6)

    def test_euler_unwrap_uses_nearest_continuous_angle(self) -> None:
        previous = np.array([3.13, 0.2, -3.13])
        current = np.array([-3.13, 0.21, 3.13])

        unwrapped = unwrap_euler_xyz(current, previous)

        self.assertLess(float(np.max(np.abs(unwrapped - previous))), 0.1)
        self.assertGreater(unwrapped[0], math.pi)
        self.assertLess(unwrapped[2], -math.pi)


class PressureContractTests(unittest.TestCase):
    def test_controller_phases_map_to_contact_stages(self) -> None:
        expected = {
            0: "idle",
            1: "precontact",
            2: "contact_onset",
            3: "lift_grasp",
            4: "transport_grasp",
            5: "placement_grasp",
            6: "release",
            7: "idle",
        }
        self.assertEqual({phase: contact_stage_for_phase(phase) for phase in expected}, expected)

    def test_precontact_only_switches_left_and_right_tip_channels(self) -> None:
        early = pressure_values_for_stage(
            "precontact",
            CHANNELS,
            baseline_adc=5000,
            max_drop_adc=3200,
            progress=0.0,
            sample_index=0,
        )
        late = pressure_values_for_stage(
            "precontact",
            CHANNELS,
            baseline_adc=5000,
            max_drop_adc=3200,
            progress=1.0,
            sample_index=0,
        )
        early_by_channel = dict(zip(CHANNELS, early))
        late_by_channel = dict(zip(CHANNELS, late))

        for channel in (51, 50):
            self.assertGreaterEqual(early_by_channel[channel], 3500)
            self.assertLessEqual(early_by_channel[channel], 4500)
            self.assertGreaterEqual(late_by_channel[channel], 3500)
            self.assertLessEqual(late_by_channel[channel], 4500)
            self.assertGreater(early_by_channel[channel], late_by_channel[channel])
        for channel in CHANNELS[2:]:
            self.assertGreaterEqual(early_by_channel[channel], 4993)
            self.assertGreaterEqual(late_by_channel[channel], 4993)

    def test_pressure_is_spatial_bounded_and_preserves_ch58_interpolation(self) -> None:
        values = pressure_values_for_stage(
            "placement_grasp",
            CHANNELS,
            baseline_adc=5000,
            max_drop_adc=3200,
            progress=0.5,
            sample_index=17,
        )

        self.assertEqual(len(values), 20)
        self.assertGreater(len(set(values)), 5)
        self.assertGreaterEqual(min(values), 0)
        self.assertLessEqual(max(values), 5007)
        by_channel = dict(zip(CHANNELS, values))
        self.assertEqual(by_channel[58], round((by_channel[61] + by_channel[57] + by_channel[59]) / 3))

    def test_pressure_stage_strength_matches_contact_progression(self) -> None:
        def mean_drop(stage: str, progress: float = 0.5) -> float:
            values = pressure_values_for_stage(
                stage,
                CHANNELS,
                baseline_adc=5000,
                max_drop_adc=3200,
                progress=progress,
                sample_index=0,
            )
            return float(np.mean(5000 - np.asarray(values)))

        self.assertLess(mean_drop("idle"), 20.0)
        self.assertLess(mean_drop("contact_onset", 0.1), mean_drop("contact_onset", 0.9))
        self.assertLess(mean_drop("transport_grasp"), mean_drop("lift_grasp"))
        self.assertLess(mean_drop("lift_grasp"), mean_drop("placement_grasp"))
        self.assertLess(mean_drop("release", 0.9), mean_drop("release", 0.1))


if __name__ == "__main__":
    unittest.main()
