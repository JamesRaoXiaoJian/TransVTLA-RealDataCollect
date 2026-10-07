"""Tests for the read-only raw simulation dataset audit helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.audit_sim_dataset import discover_session_dirs, summarize_timestamps, summarize_values, wrapped_euler_delta


class AuditSimDatasetTests(unittest.TestCase):
    def test_discovers_direct_and_sharded_session_layouts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "session_ep001_env0"
            sharded = root / "shards" / "shard00" / "session_ep002_env1"
            direct.mkdir()
            sharded.mkdir(parents=True)

            self.assertEqual(discover_session_dirs(root), [direct, sharded])

    def test_summarize_timestamps_reports_rate_and_rejects_nonpositive_deltas(self) -> None:
        summary = summarize_timestamps(np.array([0, 100_000, 200_000, 200_000, 300_000]))

        self.assertEqual(summary["valid_intervals"], 3)
        self.assertEqual(summary["nonpositive_intervals"], 1)
        self.assertAlmostEqual(summary["median_hz"], 10.0)

    def test_wrapped_euler_delta_uses_shortest_rotation(self) -> None:
        previous = np.array([0.0, 0.0, 3.13], dtype=np.float32)
        current = np.array([0.0, 0.0, -3.13], dtype=np.float32)

        delta = wrapped_euler_delta(current, previous)

        self.assertLess(abs(float(delta[2])), 0.1)
        self.assertGreater(float(delta[2]), 0.0)

    def test_summarize_values_identifies_constant_simulated_signal(self) -> None:
        summary = summarize_values([3500.0, 3500.0, 3500.0])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min"], 3500.0)
        self.assertEqual(summary["max"], 3500.0)
        self.assertTrue(summary["constant"])


if __name__ == "__main__":
    unittest.main()
