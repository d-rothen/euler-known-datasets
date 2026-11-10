from __future__ import annotations

import math

import numpy as np
import pytest

from euler_known_datasets.geometry import (
    invert_rigid,
    matrix_to_list,
    quaternion_to_matrix,
    rigid_transform,
)


def test_identity_quaternion():
    assert np.allclose(quaternion_to_matrix([1, 0, 0, 0]), np.eye(3))


def test_yaw_quaternion_rotates_x_towards_y():
    half = math.pi / 4  # 90 degrees about z
    rotation = quaternion_to_matrix([math.cos(half), 0, 0, math.sin(half)])
    assert np.allclose(rotation @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(rotation @ [0, 1, 0], [-1, 0, 0])


def test_quaternion_is_normalised():
    assert np.allclose(quaternion_to_matrix([2, 0, 0, 0]), np.eye(3))


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError):
        quaternion_to_matrix([0, 0, 0, 0])


def test_rigid_transform_and_inverse_round_trip():
    transform = rigid_transform([1, 2, 3], [0.9, 0.1, -0.3, 0.2])
    assert np.allclose(transform @ invert_rigid(transform), np.eye(4))
    assert np.allclose(invert_rigid(transform), np.linalg.inv(transform))


def test_matrix_to_list_gives_python_floats():
    values = matrix_to_list(np.eye(2, dtype=np.float32))
    assert values == [[1.0, 0.0], [0.0, 1.0]]
    assert all(type(v) is float for row in values for v in row)
