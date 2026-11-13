"""A small synthetic Hypersim, packaged like its downloads.

Three scenes of tiny 8x6 renders follow the release's layout,
``ai_VVV_NNN/{_detail,images}/...``:

- ``ai_001_001`` has an ordinary camera and two trajectories; frame 0 of
  ``cam_01`` has a stretched orientation, like ai_015_003;
- ``ai_002_001`` has a tilt-shift camera (``M_cam_from_uv`` with a ``v``
  dependent z and a shifted principal point), like ai_037_002;
- ``ai_003_001`` has a lens tilted both ways (skew, as in ai_019_007) and a trajectory whose
  first orientation carries a scale of 0.25, like ai_026_006.

Distances are generated with Hypersim's own ray formula
(``R_world_from_cam @ M_cam_from_uv @ [u, v, 1]``, normalised) from the
camera's stored position, so ``world_points`` holds what Hypersim's
``position`` images would: the tests compare the prepared depth and poses
against them independently of the recipe's camera model. Packaging is the
extracted scene folders (at the root or below ``scenes/``) or one zip per
scene; the split and camera CSVs go to ``ml-hypersim/`` at their repository
paths.
"""

from __future__ import annotations

import csv
import io
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

h5py = __import__("pytest").importorskip("h5py")

HEIGHT, WIDTH = 6, 8
METERS_PER_ASSET_UNIT = 0.0254
#: scene -> (M_cam_from_uv, {camera: (frames, split)})
SCENES = {
    "ai_001_001": (
        [[0.577, 0.0, 0.0], [0.0, 0.433, 0.0], [0.0, 0.0, -1.0]],
        {"cam_00": (6, "train"), "cam_01": (4, "train")},
    ),
    "ai_002_001": (
        [[0.593, 0.0, 0.0], [0.0, 0.433, 0.0], [0.0, 0.0996, -1.026]],
        {"cam_00": (4, "val")},
    ),
    "ai_003_001": (
        [[0.5816, 0.0, 0.0], [0.0, 0.4521, 0.0], [0.1641, -0.0318, -1.0335]],  # ai_019_007
        {"cam_00": (4, "test")},
    ),
}
#: Frames listed in the split but not publicly released.
NOT_PUBLIC = {("ai_001_001", "cam_00", 5)}
#: A trajectory whose frame 0 orientation is stretched (rendered by another camera).
STRETCHED = ("ai_001_001", "cam_01")
#: A scene folder the split does not list at all.
UNLISTED = "ai_009_009"
SPLIT_CSV = "ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
CAMERA_CSV = "ml-hypersim/contrib/mikeroberts3000/metadata_camera_parameters.csv"


def _hdf5(values) -> bytes:
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        handle.create_dataset("dataset", data=values, compression="gzip")
    return buffer.getvalue()


def _rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else -q


def rays(matrix) -> np.ndarray:
    """Hypersim's per-pixel rays in the stored camera frame, ``(H, W, 3)``."""
    u, v = np.meshgrid(np.linspace(-1 + 1 / WIDTH, 1 - 1 / WIDTH, WIDTH),
                       np.linspace(-1 + 1 / HEIGHT, 1 - 1 / HEIGHT, HEIGHT)[::-1])
    uv = np.stack([u, v, np.ones_like(u)], axis=-1)
    return uv @ np.asarray(matrix).T


def frame_path(scene: str, camera: str, frame: int, kind: str) -> str:
    folder = "final" if kind == "color" else "geometry"
    return f"{scene}/images/scene_{camera}_{folder}_hdf5/frame.{frame:04d}.{kind}.hdf5"


@dataclass
class FakeHypersim:
    source: Path
    #: scene file path (``ai_.../...``) -> content
    files: dict[str, bytes] = field(default_factory=dict)
    #: (scene, camera, frame) -> world points in meters, ``(H, W, 3)`` (NaN without geometry)
    world_points: dict = field(default_factory=dict)
    #: (scene, camera, frame) -> HDR colour, ``(H, W, 3)``
    colors: dict = field(default_factory=dict)
    #: (scene, camera, frame) -> render entity IDs
    entities: dict = field(default_factory=dict)

    def frames(self, split=None, public=True):
        out = []
        for scene, (_, cameras) in SCENES.items():
            for camera, (count, frame_split) in cameras.items():
                for frame in range(count):
                    if split not in (None, frame_split):
                        continue
                    if public and (scene, camera, frame) in NOT_PUBLIC:
                        continue
                    out.append((scene, camera, frame))
        return out


def _scene_files(fake: FakeHypersim, scene: str, matrix, cameras) -> None:
    fake.files[f"{scene}/_detail/metadata_scene.csv"] = (
        f"parameter_name,parameter_value\nmeters_per_asset_unit,{METERS_PER_ASSET_UNIT}\n".encode())
    fake.files[f"{scene}/_detail/metadata_cameras.csv"] = (
        "camera_name\n" + "".join(f"{c}\n" for c in cameras)).encode()
    stored_rays = rays(matrix)
    for camera, (count, _) in cameras.items():
        seed = zlib.crc32(f"{scene}/{camera}".encode()) % 1000
        orientations = np.stack([_rotation(seed + i) for i in range(count)])
        if scene == "ai_003_001":
            orientations[0] *= 0.25  # as in ai_026_006
        if (scene, camera) == STRETCHED:
            orientations[0] = orientations[0] @ np.diag([1.0, 1.0, 0.54])  # as in ai_015_003
        positions = np.random.default_rng(seed).uniform(-200, 200, (count, 3))
        detail = f"{scene}/_detail/{camera}"
        fake.files[f"{detail}/camera_keyframe_orientations.hdf5"] = _hdf5(orientations)
        fake.files[f"{detail}/camera_keyframe_positions.hdf5"] = _hdf5(positions)
        fake.files[f"{detail}/camera_keyframe_frame_indices.hdf5"] = _hdf5(np.arange(count))
        for frame in range(count):
            rng = np.random.default_rng(seed * 100 + frame)
            distance = rng.uniform(1.0, 8.0, (HEIGHT, WIDTH)).astype(np.float32)
            distance[0, 0] = np.nan  # a window onto nothing
            entity = rng.integers(1, 50, (HEIGHT, WIDTH)).astype(np.int32)
            entity[0, 0] = -1
            color = rng.uniform(0, 4, (HEIGHT, WIDTH, 3)).astype(np.float16)
            direction = stored_rays @ orientations[frame].T
            direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
            points = positions[frame] * METERS_PER_ASSET_UNIT + direction * distance.astype(np.float16)[..., None]
            key = (scene, camera, frame)
            fake.world_points[key] = points
            fake.colors[key] = color.astype(np.float32)
            fake.entities[key] = entity
            fake.files[frame_path(scene, camera, frame, "color")] = _hdf5(color)
            fake.files[frame_path(scene, camera, frame, "render_entity_id")] = _hdf5(entity)
            fake.files[frame_path(scene, camera, frame, "depth_meters")] = _hdf5(distance.astype(np.float16))
            # Files the recipe does not use.
            fake.files[frame_path(scene, camera, frame, "normal_cam")] = _hdf5(np.zeros((HEIGHT, WIDTH, 3), np.float16))


def _metadata(fake: FakeHypersim) -> dict[str, bytes]:
    split = io.StringIO()
    writer = csv.writer(split)
    writer.writerow(["scene_name", "camera_name", "frame_id", "included_in_public_release",
                     "exclude_reason", "split_partition_name"])
    for scene, (_, cameras) in SCENES.items():
        for camera, (count, frame_split) in cameras.items():
            for frame in range(count):
                public = (scene, camera, frame) not in NOT_PUBLIC
                writer.writerow([scene, camera, frame, public, "" if public else "CONTENT FLAGGED FOR REMOVAL",
                                 frame_split if public else ""])
    columns = ["scene_name", "settings_output_img_height", "settings_output_img_width",
               "settings_units_info_meters_scale"]
    columns += [f"M_cam_from_uv_{i}{j}" for i in range(3) for j in range(3)]
    cameras = io.StringIO()
    writer = csv.writer(cameras)
    writer.writerow(columns)
    for scene, (matrix, _) in SCENES.items():
        writer.writerow([scene, float(HEIGHT), float(WIDTH), METERS_PER_ASSET_UNIT]
                        + [matrix[i][j] for i in range(3) for j in range(3)])
    return {SPLIT_CSV: split.getvalue().encode(), CAMERA_CSV: cameras.getvalue().encode()}


def build_fake_hypersim(source: Path, layout: str = "extracted") -> FakeHypersim:
    """Write the fake below *source*: ``extracted``, ``nested`` (below ``scenes/``) or ``zip``."""
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    fake = FakeHypersim(source)
    for scene, (matrix, cameras) in SCENES.items():
        _scene_files(fake, scene, matrix, cameras)
    fake.files[f"{UNLISTED}/_detail/metadata_scene.csv"] = b"parameter_name,parameter_value\n"
    for rel, data in _metadata(fake).items():
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_bytes(data)
    if layout == "zip":
        by_scene: dict[str, list[str]] = {}
        for rel in fake.files:
            by_scene.setdefault(rel.split("/", 1)[0], []).append(rel)
        for scene, names in by_scene.items():
            with zipfile.ZipFile(source / f"{scene}.zip", "w", zipfile.ZIP_STORED) as archive:
                for name in sorted(names):
                    archive.writestr(name, fake.files[name])
        return fake
    base = source / "scenes" if layout == "nested" else source
    for rel, data in fake.files.items():
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return fake
