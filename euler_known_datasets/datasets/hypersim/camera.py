"""Hypersim's camera model as an exact pinhole camera, and the calibration file.

Hypersim publishes, per scene, ``M_cam_from_uv`` (``contrib/mikeroberts3000/
metadata_camera_parameters.csv``): the 3x3 matrix taking a pixel's NDC
coordinates ``[u, v, 1]`` (``u`` right, ``v`` up, both in ``[-1, 1]``) to its
ray in the stored camera frame (x right, y up, z backward). For the many
scenes rendered with V-Ray tilt-shift settings that ray is not a plain
pinhole ray of the stored frame: its z component varies across the image.

Every invertible pixel-to-ray matrix is a pinhole camera all the same, only
rotated. With ``A`` the pixel-to-NDC matrix (pixel centres at integers) and
``S = diag(1, -1, -1)`` (stored axes to optical axes, y down and z forward),
the QR decomposition ``S @ M_cam_from_uv @ A = Q @ R`` gives the rotation
``Q`` from the image's *optical frame* into the stored camera's optical axes
and ``K = inv(R)`` (normalised), upper triangular. Then::

    ray in the optical frame  = K^-1 [x, y, 1]
    pose (optical -> world)   = [polar(R_world_from_cam) @ S @ Q | t * meters_per_asset_unit]
    planar depth              = distance / ||K^-1 [x, y, 1]||

Checked against Hypersim's world-space ``position`` images: the points agree
to about 1 mm for an ordinary scene (ai_001_001) and a tilt-shift one
(ai_037_002), and to 6 mm in ai_026_006, whose camera axes are 41.5 degrees
from its image's. ``polar`` removes a uniform scale that frame 0 of some
trajectories carries in its orientation matrix (ai_026_006: 0.25), which the
official ray-casting code divides out by normalising rays. Frame 0 of 38
trajectories is stretched or mirrored instead, and was rendered that way (its
rays match Hypersim's positions only with the stored matrix); those frames
are left out (:func:`rigid_frames`).

The calibration file, format ``euler_known_datasets.hypersim.camera_frames``
version 1, ``calibration/<scene>/<camera>.json``::

    {
      "format": {"name": ..., "version": 1}, "generator": {...},
      "scene": "ai_001_001", "conventions": {...},
      "meters_per_asset_unit": 0.0254,
      "camera": {"name": "cam_00", "width": 1024, "height": 768,
                 "intrinsics": 3x3, "optical_tilt_deg": 0.0,
                 "stored_camera_from_optical": 3x3, "camera_from_uv": 3x3},
      "frames": {"0000": {"camera_to_world": 4x4}, ...}
    }

``stored_camera_from_optical`` maps optical-frame vectors into the stored
camera frame (Hypersim's convention); ``camera_from_uv`` is the published
matrix. The frame keys are the sample IDs of the prepared image scopes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..._version import generator_info
from ...geometry import matrix_to_list

FORMAT_NAME = "euler_known_datasets.hypersim.camera_frames"
FORMAT_VERSION = 1

#: Stored camera axes (x right, y up, z backward) to optical axes.
STORED_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])

CONVENTIONS = {
    "matrices": "row-major; X_to_Y maps homogeneous points from frame X into frame Y",
    "camera": "optical frame of the image: x right, y down, z forward",
    "stored_camera": "Hypersim's camera frame: x right, y up, z backward",
    "world": "the scene's asset coordinates scaled to meters",
    "pixels": "pixel centres at integer (x, y), x right, y down",
}


@dataclass(frozen=True)
class SceneCamera:
    """One scene's pinhole camera, derived from ``M_cam_from_uv``."""

    width: int
    height: int
    camera_from_uv: np.ndarray
    intrinsics: np.ndarray
    #: Rotation taking optical-frame vectors into the stored camera's optical axes.
    rotation: np.ndarray

    @property
    def tilt_deg(self) -> float:
        cosine = (np.trace(self.rotation) - 1.0) / 2.0
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    @property
    def stored_from_optical(self) -> np.ndarray:
        return STORED_TO_OPTICAL @ self.rotation


def scene_camera(width: int, height: int, camera_from_uv: np.ndarray) -> SceneCamera:
    camera_from_uv = np.asarray(camera_from_uv, dtype=np.float64).reshape(3, 3)
    pixel_to_uv = np.array([
        [2.0 / width, 0.0, 1.0 / width - 1.0],
        [0.0, -2.0 / height, 1.0 - 1.0 / height],
        [0.0, 0.0, 1.0],
    ])
    q, r = np.linalg.qr(STORED_TO_OPTICAL @ camera_from_uv @ pixel_to_uv)
    signs = np.diag(np.sign(np.diag(r)))
    q, r = q @ signs, signs @ r
    if np.linalg.det(q) < 0:
        raise ValueError("M_cam_from_uv maps pixels to a mirrored camera")
    intrinsics = np.linalg.inv(r)
    intrinsics /= intrinsics[2, 2]
    intrinsics[1, 0] = intrinsics[2, 0] = intrinsics[2, 1] = 0.0
    return SceneCamera(width, height, camera_from_uv, intrinsics, q)


def camera_parameters(rows: list[Mapping[str, str]]) -> dict[str, SceneCamera]:
    """Parse ``metadata_camera_parameters.csv`` rows into cameras by scene name."""
    cameras = {}
    for row in rows:
        matrix = np.array([[float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)])
        cameras[row["scene_name"]] = scene_camera(
            int(float(row["settings_output_img_width"])),
            int(float(row["settings_output_img_height"])),
            matrix,
        )
    return cameras


#: How far an orientation may be from a uniformly scaled rotation: the
#: largest over the smallest singular value.
MAX_ANISOTROPY = 1.0 + 1e-3


def rigid_frames(orientations: np.ndarray) -> np.ndarray:
    """Which orientations (``(N, 3, 3)``) are uniformly scaled rotations.

    Hypersim renders with the stored matrix as it is. A uniform scale only
    changes the length of the rays, which are normalised, but a stretch or a
    reflection changes their directions: such a frame was rendered by another
    camera than its scene's (or a mirrored one) and has no pose in the
    scene's pinhole model. In the release these are frame 0 of 38
    trajectories, e.g. ai_015_003 cam_00 (one axis scaled by 0.54) and
    ai_008_003 cam_01 (mirrored).
    """
    orientations = np.asarray(orientations, dtype=np.float64)
    singular = np.linalg.svd(orientations, compute_uv=False)
    isotropic = singular.max(axis=1) <= singular.min(axis=1) * MAX_ANISOTROPY
    return isotropic & (np.linalg.det(orientations) > 0)


def orthonormal(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The nearest rotations to *rotations* (``(N, 3, 3)``) and their singular values."""
    u, s, vt = np.linalg.svd(rotations)
    return u @ vt, s


def camera_to_world(
    camera: SceneCamera, orientations: np.ndarray, positions: np.ndarray, meters_per_asset_unit: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """``(N, 4, 4)`` optical-frame poses in meters, which of them are valid
    (see :func:`rigid_frames`), and notes on the orientations."""
    orientations = np.asarray(orientations, dtype=np.float64)
    valid = rigid_frames(orientations)
    rotations, singular = orthonormal(orientations)
    poses = np.tile(np.eye(4), (len(rotations), 1, 1))
    poses[:, :3, :3] = rotations @ camera.stored_from_optical
    poses[:, :3, 3] = np.asarray(positions, dtype=np.float64) * meters_per_asset_unit
    notes: dict[str, Any] = {}
    scaled = valid & (np.abs(singular - 1.0).max(axis=1) > 1e-3)
    if np.any(scaled):
        notes["uniformly_scaled_frames"] = [int(i) for i in np.flatnonzero(scaled)]
    if not np.all(valid):
        notes["frames_rendered_with_another_camera"] = [int(i) for i in np.flatnonzero(~valid)]
    return poses, valid, notes


def calibration_document(
    *,
    scene: str,
    camera_name: str,
    camera: SceneCamera,
    meters_per_asset_unit: float,
    poses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "format": {"name": FORMAT_NAME, "version": FORMAT_VERSION},
        "generator": generator_info(),
        "scene": scene,
        "conventions": CONVENTIONS,
        "meters_per_asset_unit": meters_per_asset_unit,
        "camera": {
            "name": camera_name,
            "width": camera.width,
            "height": camera.height,
            "intrinsics": matrix_to_list(camera.intrinsics),
            "optical_tilt_deg": round(camera.tilt_deg, 4),
            "stored_camera_from_optical": matrix_to_list(camera.stored_from_optical),
            "camera_from_uv": matrix_to_list(camera.camera_from_uv),
        },
        "frames": {frame: {"camera_to_world": matrix_to_list(pose)} for frame, pose in poses.items()},
    }
