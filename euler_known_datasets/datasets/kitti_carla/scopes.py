"""The ds-crawler scopes of prepared KITTI-CARLA towns.

Every regular scope uses the hierarchy ``/town:TownNN/camera:N/`` and the ID
``frame-NNNN``, the frame number as written in the file names. These are the
IDs euler-loading's KITTI-CARLA tests use, and the IDs that derived datasets
made from earlier in-archive indexes already carry, so those still join a
prepared town.

The ``1``/``2`` that segmentation and depth file names put before the camera
digit belong to neither the ID nor the hierarchy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

LOADER_MODULE = "kitti_carla"
CAMERAS = ("0", "1")
SKY = [70, 130, 180]
DEPTH_FAR_PLANE_M = 1000.0
DEPTH_SCALE_TO_METERS = DEPTH_FAR_PLANE_M / (2**24 - 1)
LIDAR_COLUMNS = ["x", "y", "z", "cos_angle_lidar_surface", "timestamp", "instance", "semantic"]


def addon(function: str) -> dict[str, Any]:
    return {"euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": function}}


@dataclass(frozen=True)
class ImageScope:
    """A per-camera image folder, indexed with regexes."""

    scope: str
    folder: str
    prefix: str  # before the camera digit: "" (rgb), "1" (segmentation), "2" (depth)
    modality: dict[str, Any]
    function: str

    @property
    def pattern(self) -> re.Pattern:
        return re.compile(
            rf"^(?P<town>Town\d{{2}})/generated/{self.folder}/"
            rf"(?P<frame>\d+)_{self.prefix}(?P<camera>\d)\.png$"
        )

    @property
    def indexing(self) -> dict[str, Any]:
        folder, prefix = self.folder, self.prefix
        return {
            "id": {
                "regex": rf"^Town\d{{2}}/generated/{folder}/(?P<frame>\d+)_{prefix}\d\.png$",
                "join_char": "+",
            },
            "hierarchy": {
                "regex": rf"^(?P<town>Town\d{{2}})/generated/{folder}/\d+_{prefix}(?P<camera>\d)\.png$",
                "separator": ":",
            },
            "files": {"extensions": [".png"]},
            "constraints": {"flat_ids_unique": False},
        }


@dataclass(frozen=True)
class FileScope:
    """One file per town or per town and camera, under a semantic ID."""

    scope: str
    file_id: str
    pattern_text: str  # hierarchy regex; names ``town`` and, per camera, ``camera``
    id_regex: str  # one capture group; its value is replaced by ``file_id``
    modality: dict[str, Any]
    function: str
    per_camera: bool

    @property
    def pattern(self) -> re.Pattern:
        return re.compile(self.pattern_text)

    @property
    def indexing(self) -> dict[str, Any]:
        return {
            "id": {"regex": self.id_regex, "join_char": "+", "override": self.file_id},
            "hierarchy": {"regex": self.pattern_text, "separator": ":"},
            "files": {"extensions": [".txt"]},
            "constraints": {"flat_ids_unique": False},
        }


IMAGE_SCOPES = {
    "rgb": ImageScope(
        scope="rgb",
        folder="images_rgb",
        prefix="",
        modality={"key": "rgb", "meta": {"range": [0, 255]}},
        function="rgb",
    ),
    "depth": ImageScope(
        scope="depth",
        folder="images_depth",
        prefix="2",
        # The PNG stores CARLA's 24-bit code; scale_to_meters turns a code into
        # planar metres, the code 2**24 - 1 being the 1000 m far plane. The
        # loader decodes it with its own constants.
        modality={
            "key": "depth",
            "meta": {
                "radial_depth": False,
                "scale_to_meters": DEPTH_SCALE_TO_METERS,
                "range": [0, DEPTH_FAR_PLANE_M],
            },
        },
        function="depth",
    ),
    "semantic_segmentation": ImageScope(
        scope="semantic_segmentation",
        folder="images_ss",
        prefix="1",
        modality={"key": "semantic_segmentation", "meta": {"skyclass": SKY}},
        function="class_segmentation",
    ),
}

LIDAR_SCOPE = "lidar"
LIDAR_PATTERN = re.compile(r"^(?P<town>Town\d{2})/generated/frames/frame_(?P<frame>\d+)\.ply$")
LIDAR_MODALITY = {
    "key": "lidar_point_cloud",
    "meta": {
        "representation": "point_cloud",
        "columns": LIDAR_COLUMNS,
        "coordinate_unit": "meters",
        "frame": "lidar",
        "axes": "x_forward_y_left_z_up",
    },
}
LIDAR_FUNCTION = "lidar_point_cloud"

_CALIBRATION = r"^(?P<town>Town\d{2})/generated/lidar_to_cam(?P<camera>\d)\.txt$"
_CALIBRATION_ID = r"^Town\d{2}/generated/(lidar_to_cam)\d\.txt$"

FILE_SCOPES = {
    "intrinsics": FileScope(
        scope="intrinsics",
        file_id="intrinsics",
        pattern_text=_CALIBRATION,
        id_regex=_CALIBRATION_ID,
        modality={"key": "intrinsics", "meta": {"camera_model": "pinhole", "dimensions": {"rows": 3, "columns": 3}}},
        function="read_intrinsics",
        per_camera=True,
    ),
    "camera_extrinsics": FileScope(
        scope="camera_extrinsics",
        file_id="extrinsics",
        pattern_text=_CALIBRATION,
        id_regex=_CALIBRATION_ID,
        modality={
            "key": "camera_extrinsics",
            "meta": {
                "source_frame": "lidar",
                "target_frame": "camera_optical",
                "direction": "target_from_source",
                "dimensions": {"rows": 4, "columns": 4},
            },
        },
        function="read_extrinsics",
        per_camera=True,
    ),
    "lidar_pose": FileScope(
        scope="lidar_pose",
        file_id="lidar_pose",
        pattern_text=r"^(?P<town>Town\d{2})/generated/full_poses_lidar\.txt$",
        id_regex=r"^Town\d{2}/generated/(full_poses_lidar)\.txt$",
        modality={
            "key": "lidar_pose",
            "meta": {
                "source_frame": "lidar",
                "target_frame": "world",
                "direction": "target_from_source",
                "dimensions": {"rows": 4, "columns": 4},
            },
        },
        function="lidar_pose",
        per_camera=False,
    ),
}

#: Regular scopes a user can choose; calibration and poses always come along.
MODALITIES = (*IMAGE_SCOPES, LIDAR_SCOPE)
