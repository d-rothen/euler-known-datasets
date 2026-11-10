"""Rigid-transform helpers shared by recipes.

Conventions: quaternions are ``(w, x, y, z)``, the order nuScenes and most
robotics datasets store; a 4x4 ``T_b_a`` maps homogeneous points from frame
``a`` into frame ``b`` (``p_b = T_b_a @ p_a``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Return the 3x3 rotation of a ``(w, x, y, z)`` quaternion (normalised first)."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have 4 components (w, x, y, z), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"quaternion {list(q)} has no rotation (norm {norm})")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rigid_transform(
    translation: Sequence[float], rotation: Sequence[float]
) -> np.ndarray:
    """Return the 4x4 transform with a ``(w, x, y, z)`` rotation and a translation."""
    t = np.asarray(translation, dtype=np.float64)
    if t.shape != (3,):
        raise ValueError(f"translation must have 3 components, got {t.shape}")
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(rotation)
    transform[:3, 3] = t
    return transform


def invert_rigid(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform without a general matrix inverse."""
    rotation = transform[:3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    """A JSON-ready nested list of Python floats."""
    return [[float(value) for value in row] for row in np.asarray(matrix)]
