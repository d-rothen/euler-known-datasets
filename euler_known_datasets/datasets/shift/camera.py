"""SHIFT's front camera: intrinsics, poses and the calibration files the recipe writes.

**Intrinsics.** Every frame of ``det_2d.json`` (train and val, 1.75 million
frames) gives ``focal (640, 640)``, ``center (640, 400)``, skew 0: an ideal
pinhole with a 90 degree horizontal field of view on the 1280x800 grid.

**Poses.** Each frame's ``extrinsics`` give the camera's ``location`` (meters)
and ``rotation`` (degrees) in CARLA's world, which is left-handed (x, y in
the ground plane, z up). The recipe mirrors it into a right-handed world by
negating y, in which the camera-to-world rotation is
``Rz(-r[2]) @ Ry(r[1]) @ Rx(-r[0])`` applied to the body axes (x forward, y
left, z up), and the translation is ``(x, -y, z)``. Established by depth
reprojection (2026-11-13): with this convention, frames 1-5 s apart in six
train sequences reproject onto each other's planar depth with a median
relative error of 0.14 %, also through 34-57 degree turns, where keeping the
world as given fails (``r[2]`` as yaw in a right-handed world: 47 % error on a
9 degree turn). ``r[2]`` is the heading of the motion, ``r[1]`` is negative
when the road climbs, ``r[0]`` is a small roll.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ...geometry import matrix_to_list
from .selection import SequenceInfo

FORMAT_NAME = "euler_known_datasets.shift.camera_frames"
FORMAT_VERSION = 1
WIDTH, HEIGHT = 1280, 800
CAMERA = "front"

#: Optical axes (x right, y down, z forward) expressed in body axes (x forward, y left, z up).
BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def intrinsics_matrix(intrinsics: Mapping[str, Any]) -> np.ndarray:
    """``K`` from a Scalabel ``intrinsics`` block (``focal``, ``center``, ``skew``)."""
    fx, fy = (float(v) for v in intrinsics["focal"])
    cx, cy = (float(v) for v in intrinsics["center"])
    return np.array([[fx, float(intrinsics.get("skew") or 0.0), cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _rotation(axis: int, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    matrix = np.eye(3)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


def camera_to_world(location: Any, rotation: Any) -> np.ndarray:
    """The ``(4, 4)`` pose of the optical frame in the right-handed world (see the module)."""
    x, y, z = (float(v) for v in location)
    r0, r1, r2 = (float(v) for v in rotation)
    pose = np.eye(4)
    pose[:3, :3] = _rotation(2, -r2) @ _rotation(1, r1) @ _rotation(0, -r0) @ BODY_FROM_OPTICAL
    pose[:3, 3] = (x, -y, z)
    return pose


def calibration_document(sequence: SequenceInfo, records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """The calibration file of one sequence: the camera matrix and each frame's pose.

    Raises if the frames disagree on the intrinsics.
    """
    matrices = {tuple(intrinsics_matrix(r["intrinsics"]).ravel()) for r in records.values()}
    if len(matrices) > 1:
        raise ValueError(f"sequence {sequence.name}: frames disagree on the intrinsics: {sorted(matrices)}")
    matrix = np.array(next(iter(matrices))).reshape(3, 3) if matrices else intrinsics_matrix(
        {"focal": (640, 640), "center": (640, 400)})
    frames = {}
    for frame, record in sorted(records.items()):
        extrinsics = record["extrinsics"]
        frames[f"{frame:08d}"] = {
            "camera_to_world": matrix_to_list(camera_to_world(extrinsics["location"], extrinsics["rotation"])),
            "source": {"name": record["name"], "location": list(extrinsics["location"]),
                       "rotation": list(extrinsics["rotation"])},
        }
    return {
        "format": {"name": FORMAT_NAME, "version": FORMAT_VERSION},
        "sequence": sequence.name,
        "split": sequence.split,
        "camera": {"name": CAMERA, "width": WIDTH, "height": HEIGHT, "intrinsics": matrix_to_list(matrix)},
        "world": "CARLA's world with y negated (right-handed, z up, meters)",
        "frames": frames,
    }
