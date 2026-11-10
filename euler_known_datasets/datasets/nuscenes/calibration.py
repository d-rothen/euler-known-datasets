"""The per-scene camera calibration file, format version 1.

``calibration/<scene>/<camera>.json`` holds one camera's calibration for one
scene together with a record per camera frame. Each record has the camera's
pose and, when the frame has a synchronised sweep, the 4x4 transform from that
sweep into the camera::

    {
      "format": {"name": "euler_known_datasets.nuscenes.camera_frames", "version": 1},
      "generator": {...}, "dataset_version": "v1.0-test",
      "scene": {"name", "token", "log", "location", "vehicle"},
      "conventions": {...},
      "camera": {"channel", "calibrated_sensor_token", "width", "height",
                 "intrinsics": 3x3, "camera_to_ego": 4x4},
      "lidar": {"channel", "calibrated_sensor_token", "lidar_to_ego": 4x4,
                "nominal_offset_ms", "sync_tolerance_ms"},
      "frames": {
        "<camera timestamp>": {
          "sample_data_token", "sample_token", "is_key_frame", "path",
          "camera_to_global": 4x4,
          "lidar": null | {"sample_data_token", "timestamp", "offset_ms", "path",
                           "lidar_to_camera": 4x4}
        }
      }
    }

The frame keys are the sample IDs of the prepared ``rgb`` and ``lidar`` scopes,
so a reader selects a sample's record by its ID. All matrices are row-major
and map points from the first-named frame into the second. ``camera_to_ego`` and
``lidar_to_ego`` come straight from ``calibrated_sensor``.
``camera_to_global = ego_pose(t_cam) @ camera_to_ego``, and
``lidar_to_camera = inv(camera_to_global) @ ego_pose(t_lidar) @ lidar_to_ego``,
which is the projection the nuScenes devkit uses (``map_pointcloud_to_image``).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ..._version import generator_info
from ...geometry import invert_rigid, matrix_to_list, rigid_transform
from .tables import Frame, Scene

FORMAT_NAME = "euler_known_datasets.nuscenes.camera_frames"
FORMAT_VERSION = 1

CONVENTIONS = {
    "matrices": "row-major; X_to_Y maps homogeneous points from frame X into frame Y",
    "camera": "optical axes: x right, y down, z forward",
    "lidar": "LIDAR_TOP sensor frame: x right, y forward, z up; points are "
    "given at the sweep's timestamp (the end of its rotation)",
    "ego": "vehicle frame: x forward, y left, z up (origin at the rear axle centre)",
    "global": "the nuScenes map frame of the scene's location",
    "timestamps": "microseconds",
}


def calibration_path(scene_name: str, camera: str) -> str:
    return f"calibration/{scene_name}/{camera}.json"


def pose_matrix(frame: Frame) -> np.ndarray:
    """``ego_to_global`` at the frame's timestamp."""
    return rigid_transform(frame.ego_translation, frame.ego_rotation)


def sensor_matrix(calibrated_sensor: Mapping[str, Any]) -> np.ndarray:
    """``sensor_to_ego`` of a ``calibrated_sensor`` row."""
    return rigid_transform(calibrated_sensor["translation"], calibrated_sensor["rotation"])


def frame_record(
    camera: Frame,
    camera_to_ego: np.ndarray,
    lidar: Optional[Frame],
    lidar_to_ego: np.ndarray,
) -> dict[str, Any]:
    camera_to_global = pose_matrix(camera) @ camera_to_ego
    record: dict[str, Any] = {
        "sample_data_token": camera.token,
        "sample_token": camera.sample_token,
        "is_key_frame": camera.is_key_frame,
        "path": camera.filename,
        "camera_to_global": matrix_to_list(camera_to_global),
        "lidar": None,
    }
    if lidar is not None:
        lidar_to_global = pose_matrix(lidar) @ lidar_to_ego
        record["lidar"] = {
            "sample_data_token": lidar.token,
            "timestamp": lidar.timestamp,
            "offset_ms": (lidar.timestamp - camera.timestamp) / 1e3,
            "path": lidar.filename,
            "lidar_to_camera": matrix_to_list(invert_rigid(camera_to_global) @ lidar_to_global),
        }
    return record


def calibration_document(
    *,
    version: str,
    scene: Scene,
    camera_calib: Mapping[str, Any],
    lidar_calib: Optional[Mapping[str, Any]],
    channel: str,
    lidar_channel: str,
    width: int,
    height: int,
    nominal_offset_ms: float,
    sync_tolerance_ms: float,
    frames: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "format": {"name": FORMAT_NAME, "version": FORMAT_VERSION},
        "generator": generator_info(),
        "dataset_version": version,
        "scene": {
            "name": scene.name,
            "token": scene.token,
            "log": scene.logfile,
            "location": scene.location,
            "vehicle": scene.vehicle,
        },
        "conventions": CONVENTIONS,
        "camera": {
            "channel": channel,
            "calibrated_sensor_token": camera_calib["token"],
            "width": width,
            "height": height,
            "intrinsics": [[float(v) for v in row] for row in camera_calib["camera_intrinsic"]],
            "camera_to_ego": matrix_to_list(sensor_matrix(camera_calib)),
        },
        # null when the scene has no sweeps of the lidar channel at all
        "lidar": None
        if lidar_calib is None
        else {
            "channel": lidar_channel,
            "calibrated_sensor_token": lidar_calib["token"],
            "lidar_to_ego": matrix_to_list(sensor_matrix(lidar_calib)),
            "nominal_offset_ms": round(nominal_offset_ms, 3),
            "sync_tolerance_ms": sync_tolerance_ms,
        },
        "frames": dict(frames),
    }
