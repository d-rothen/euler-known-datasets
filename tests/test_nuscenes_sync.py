from __future__ import annotations

import numpy as np
import pytest

from euler_known_datasets.datasets.nuscenes.sync import (
    nominal_offset_us,
    pair_with_lidar,
)

NOMINAL = 43_000.0
TOLERANCE = 20_000.0
LIDAR = np.arange(10) * 50_000  # sweep ends


def test_pairs_with_the_sweep_covering_the_exposure():
    cameras = LIDAR[3:6] - 43_000
    index, offset, synced = pair_with_lidar(cameras, LIDAR, NOMINAL, TOLERANCE)
    assert index.tolist() == [3, 4, 5]
    assert offset.tolist() == [43_000] * 3
    assert synced.all()


def test_nearest_timestamp_would_pick_the_previous_sweep():
    camera = LIDAR[4] - 43_000
    assert np.argmin(np.abs(LIDAR - camera)) == 3  # 7 ms away, but 50 ms older
    index, _, _ = pair_with_lidar([camera], LIDAR, NOMINAL, TOLERANCE)
    assert index.tolist() == [4]


def test_dropped_sweep_leaves_the_frame_unsynchronised():
    lidar = np.delete(LIDAR, 4)
    _, offset, synced = pair_with_lidar([LIDAR[4] - 43_000], lidar, NOMINAL, TOLERANCE)
    assert not synced[0]
    assert abs(offset[0] - NOMINAL) >= 50_000 - 1


def test_late_stamped_sweep_is_within_tolerance():
    lidar = LIDAR.copy()
    lidar[4] += 10_000
    index, _, synced = pair_with_lidar([LIDAR[4] - 43_000], lidar, NOMINAL, TOLERANCE)
    assert index.tolist() == [4] and synced[0]
    _, _, strict = pair_with_lidar([LIDAR[4] - 43_000], lidar, NOMINAL, 5_000.0)
    assert not strict[0]


def test_spurious_record_next_to_the_camera_is_not_chosen():
    camera = LIDAR[4] - 43_000
    lidar = np.sort(np.append(LIDAR, camera + 1_000))
    index, offset, synced = pair_with_lidar([camera], lidar, NOMINAL, TOLERANCE)
    assert offset.tolist() == [43_000] and synced[0]


def test_no_sweeps():
    index, _, synced = pair_with_lidar([1, 2], [], NOMINAL, TOLERANCE)
    assert index.tolist() == [-1, -1] and not synced.any()


def test_unsorted_sweeps_are_rejected():
    with pytest.raises(ValueError):
        pair_with_lidar([0], [2, 1], NOMINAL, TOLERANCE)


def test_nominal_offset_is_the_median():
    assert nominal_offset_us([42_000, 43_000, 48_000]) == 43_000
    with pytest.raises(ValueError):
        nominal_offset_us([])
