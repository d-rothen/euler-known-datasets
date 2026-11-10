"""Pairing camera frames with the lidar sweep that saw the same moment.

A LIDAR_TOP sweep's timestamp is the *end* of a 50 ms clockwise rotation, and
each camera is triggered when the beam crosses the centre of its view. So a
camera fires a fixed time before the end of the sweep that covers its view:
about 43 ms for CAM_FRONT_LEFT, 35 ms for CAM_FRONT and 0.4 ms for
CAM_BACK_LEFT. These offsets are linear in the camera's azimuth, 360 degrees
to 50 ms, measured on v1.0-test.

A camera frame is therefore paired with the sweep whose ``t_lidar - t_cam`` is
closest to the camera's nominal offset, which is estimated from nuScenes' own
keyframe association (the camera and LIDAR_TOP rows of the same ``sample``).
On every keyframe this rule picks the sample's own sweep. Pairing by nearest
timestamp would instead pick the previous sweep, whose points in the camera's
view are about 50 ms older than the image.

Sweep timestamps jitter: now and then a sweep is stamped up to ~14 ms late and
the next one correspondingly early (their two periods still add up to 100 ms).
The sweep is the right one; only its stamp moved. Beyond that jitter there is a
clean gap. On CAM_FRONT_LEFT in v1.0-test, deviations from the nominal offset
run continuously up to 13.7 ms and then jump to 46.3 ms. Frames past the gap
have no sweep record covering them, because the sweep was dropped. That is
121 of 34,902 frames (0.35%), and none of them is a keyframe. The default
tolerance of 20 ms sits in that gap. It also stays below half a rotation
(25 ms), so a neighbouring sweep can never be accepted.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .tables import Frame

ROTATION_PERIOD_US = 50_000


def keyframe_offsets_us(
    camera_frames: Iterable[Frame], lidar_keyframe_by_sample: Mapping[str, Frame]
) -> list[int]:
    """``t_lidar - t_cam`` for each camera keyframe and its sample's lidar keyframe."""
    offsets = []
    for frame in camera_frames:
        if not frame.is_key_frame:
            continue
        lidar = lidar_keyframe_by_sample.get(frame.sample_token)
        if lidar is not None:
            offsets.append(lidar.timestamp - frame.timestamp)
    return offsets


def nominal_offset_us(offsets: Iterable[int]) -> float:
    values = np.asarray(list(offsets), dtype=np.float64)
    if values.size == 0:
        raise ValueError("no camera keyframe has a lidar keyframe in the same sample")
    return float(np.median(values))


def pair_with_lidar(
    camera_us: np.ndarray, lidar_us: np.ndarray, nominal_us: float, tolerance_us: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair each camera timestamp with a lidar sweep.

    Args:
        camera_us: Camera timestamps (microseconds).
        lidar_us: Sorted lidar sweep timestamps of the same scene.
        nominal_us: The camera's nominal ``t_lidar - t_cam``.
        tolerance_us: Largest accepted deviation from *nominal_us*.

    Returns:
        ``(index, offset_us, synced)``: the chosen sweep's index into
        *lidar_us* (``-1`` without sweeps), its ``t_lidar - t_cam``, and whether
        it is within the tolerance.
    """
    camera_us = np.asarray(camera_us, dtype=np.int64)
    lidar_us = np.asarray(lidar_us, dtype=np.int64)
    n = camera_us.shape[0]
    if lidar_us.size == 0:
        return np.full(n, -1), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
    if np.any(np.diff(lidar_us) < 0):
        raise ValueError("lidar timestamps must be sorted")
    target = camera_us + nominal_us
    upper = np.clip(np.searchsorted(lidar_us, target), 0, lidar_us.size - 1)
    lower = np.clip(upper - 1, 0, lidar_us.size - 1)
    pick_upper = np.abs(lidar_us[upper] - target) < np.abs(lidar_us[lower] - target)
    index = np.where(pick_upper, upper, lower)
    offset = lidar_us[index] - camera_us
    synced = np.abs(offset - nominal_us) <= tolerance_us
    return index, offset, synced
