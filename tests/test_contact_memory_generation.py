import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.add_contact_memory_to_rlds import (
    ContactDecisionConfig,
    build_contact_track,
    missing_shard_indices,
    output_shard_name,
    resolve_shard_window,
    robust_pressure_stats,
)


def test_small_pressure_drift_does_not_create_contact():
    baseline_rows = np.full((50, 20), 5000.0, dtype=np.float32)
    current_rows = np.full((6, 20), 4995.0, dtype=np.float32)
    baseline, noise = robust_pressure_stats(baseline_rows)

    track = build_contact_track(
        current_rows,
        baseline,
        noise,
        use_tactile=np.ones(6, dtype=bool),
        contact_xyz=np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (6, 1)),
        contact_region=np.array(["left_tip"] * 6),
        config=ContactDecisionConfig(min_delta_floor=20.0, stable_frames=2),
    )

    assert not track.valid_contact_mask.any()
    assert np.all(track.contact_time == -1)


def test_stable_pressure_drop_creates_one_active_anchor():
    baseline_rows = np.full((50, 20), 5000.0, dtype=np.float32)
    current_rows = np.full((6, 20), 5000.0, dtype=np.float32)
    current_rows[2:, 0] = 4920.0
    baseline, noise = robust_pressure_stats(baseline_rows)

    xyz = np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (6, 1))
    track = build_contact_track(
        current_rows,
        baseline,
        noise,
        use_tactile=np.ones(6, dtype=bool),
        contact_xyz=xyz,
        contact_region=np.array(["left_tip"] * 6),
        config=ContactDecisionConfig(min_delta_floor=20.0, stable_frames=2),
    )

    assert not track.valid_contact_mask[2].any()
    assert track.valid_contact_mask[3, 0]
    np.testing.assert_allclose(track.contact_memory[3, 0], xyz[3])
    assert track.contact_time[3, 0] == 3
    assert track.valid_contact_mask[5, 0]
    assert track.valid_contact_mask[5].sum() == 1


def test_use_tactile_false_masks_otherwise_valid_contact():
    baseline_rows = np.full((50, 20), 5000.0, dtype=np.float32)
    current_rows = np.full((6, 20), 4900.0, dtype=np.float32)
    baseline, noise = robust_pressure_stats(baseline_rows)

    track = build_contact_track(
        current_rows,
        baseline,
        noise,
        use_tactile=np.zeros(6, dtype=bool),
        contact_xyz=np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (6, 1)),
        contact_region=np.array(["left_tip"] * 6),
        config=ContactDecisionConfig(min_delta_floor=20.0, stable_frames=2),
    )

    assert not track.valid_contact_mask.any()


def test_resolve_shard_window_preserves_global_indices():
    shards = [Path(f"shard-{idx:05d}") for idx in range(512)]

    selected = resolve_shard_window(shards, start_shard=448, end_shard=450, max_shards=None)

    assert selected == [(448, shards[448]), (449, shards[449])]


def test_missing_shard_indices_treats_empty_files_as_missing():
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        (work_dir / output_shard_name(0, 3)).write_bytes(b"not-empty")
        (work_dir / output_shard_name(1, 3)).write_bytes(b"")

        assert missing_shard_indices(work_dir, 3) == [1, 2]


if __name__ == "__main__":
    test_small_pressure_drift_does_not_create_contact()
    test_stable_pressure_drop_creates_one_active_anchor()
    test_use_tactile_false_masks_otherwise_valid_contact()
    test_resolve_shard_window_preserves_global_indices()
    test_missing_shard_indices_treats_empty_files_as_missing()
