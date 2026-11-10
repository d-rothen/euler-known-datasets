"""A small synthetic KITTI-CARLA, shaped like the official town archives.

Each town holds ``TownNN/generated/`` with ``FRAMES`` frames of both cameras:
RGBA images, CARLA-palette segmentation (sky over road), 24-bit depth that
encodes ``10 * (frame + 1) + camera`` metres, a binary PLY sweep per frame,
``lidar_to_cam<N>.txt``, a 1 kHz ``full_poses_lidar.txt`` driving 10 m/s along
x, and the files the recipe must leave out (``full_ts_camera.txt``,
``poses_lidar.ply``, ``screenshot.png``). Archives also carry a stray
``.ds_crawler/`` like the real ``Town01.zip`` does.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

FRAMES = 4
CAMERAS = ("0", "1")
SKY = (70, 130, 180)
ROAD = (128, 64, 128)
SPEED_M_S = 10.0
POINTS = np.array(
    [
        [10.3, 0.0, -0.1, 0.5, 0.0, 0, 7],
        [-3.25, 12.5, 1.75, 1.0, 0.0994, 336, 10],
        [79.5, -79.5, -2.0, 0.003, 0.099, 4294967295, 22],
    ]
)
EXTRINSICS_HEADER = (
    "#R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2)"
)


def depth_metres(frame: int, camera: str) -> float:
    return 10.0 * (frame + 1) + int(camera)


def _png(rgb: np.ndarray) -> bytes:
    alpha = np.full(rgb.shape[:2] + (1,), 255, np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(np.concatenate([rgb, alpha], axis=-1), "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def _depth_png(metres: float) -> bytes:
    code = int(round(metres / 1000.0 * (2**24 - 1)))
    rgb = np.array([code & 255, (code >> 8) & 255, (code >> 16) & 255], np.uint8)
    return _png(np.broadcast_to(rgb, (2, 3, 3)).copy())


def _ply(points: np.ndarray) -> bytes:
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("cos_angle_lidar_surface", "f4"),
              ("timestamp", "f4"), ("instance", "u4"), ("semantic", "u4")]
    names = {"f4": "float32", "u4": "uint32"}
    vertices = np.empty(len(points), dtype=[(n, "<" + t) for n, t in fields])
    for column, (name, _) in enumerate(fields):
        vertices[name] = points[:, column]
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}"]
        + [f"property {names[t]} {n}" for n, t in fields]
        + ["end_header", ""]
    )
    return header.encode("ascii") + vertices.tobytes()


def _poses(frames: int) -> bytes:
    lines = [
        "# R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) "
        "R(2,0) R(2,1) R(2,2) t(2) timestamp"
    ]
    for step in range(frames * 100 + 1):
        t = step / 1000.0
        lines.append(f"1 0 0 {SPEED_M_S * t} 0 1 0 0 0 0 1 0 {t}")
    return ("\n".join(lines) + "\n").encode()


def town_files(town: str, frames: int = FRAMES) -> dict[str, bytes]:
    base = f"{town}/generated"
    files: dict[str, bytes] = {}
    rng = np.random.default_rng(int(town[-2:]))
    for frame in range(frames):
        name = f"{frame:04d}"
        for camera in CAMERAS:
            files[f"{base}/images_rgb/{name}_{camera}.png"] = _png(
                rng.integers(0, 256, (2, 3, 3), dtype=np.uint8)
            )
            files[f"{base}/images_ss/{name}_1{camera}.png"] = _png(
                np.array([[SKY, SKY, ROAD]] * 2, np.uint8)
            )
            files[f"{base}/images_depth/{name}_2{camera}.png"] = _depth_png(
                depth_metres(frame, camera)
            )
        files[f"{base}/frames/frame_{name}.ply"] = _ply(POINTS)
    for camera in CAMERAS:
        files[f"{base}/lidar_to_cam{camera}.txt"] = (
            f"{EXTRINSICS_HEADER}\n1 0 0 0.0 0 1 0 {0.5 * int(camera)} 0 0 1 0.0\n".encode()
        )
    files[f"{base}/full_poses_lidar.txt"] = _poses(frames)
    files[f"{base}/full_ts_camera.txt"] = b"# frame_id timestamp\n0 0.0\n"
    files[f"{base}/poses_lidar.ply"] = _ply(POINTS[:1])
    files[f"{base}/screenshot.png"] = _png(np.zeros((2, 2, 3), np.uint8))
    return files


@dataclass
class FakeKittiCarla:
    source: Path
    towns: tuple[str, ...]
    files: dict[str, bytes] = field(default_factory=dict)


def build_fake_kitti_carla(
    source: Path,
    *,
    towns: tuple[str, ...] = ("Town01", "Town02"),
    frames: int = FRAMES,
    archives: bool = True,
    extracted: bool = False,
) -> FakeKittiCarla:
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    fake = FakeKittiCarla(source=source, towns=towns)
    for town in towns:
        files = town_files(town, frames)
        fake.files.update(files)
        if archives:
            with zipfile.ZipFile(source / f"{town}.zip", "w") as zf:
                for directory in sorted({str(Path(n).parent) + "/" for n in files}):
                    zf.writestr(directory, b"")
                for name, content in files.items():
                    zf.writestr(name, content)
                zf.writestr(".ds_crawler/rgb/index.json", b"{}")
        if extracted:
            for name, content in files.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    return fake


def needed_files(fake: FakeKittiCarla, *, towns=None, cameras=CAMERAS, step=1,
                 modalities=("rgb", "depth", "semantic_segmentation", "lidar")) -> set[str]:
    """What the recipe should index for the given options."""
    folders = {"rgb": ("images_rgb", ""), "depth": ("images_depth", "2"),
               "semantic_segmentation": ("images_ss", "1")}
    wanted = set()
    for town in towns or fake.towns:
        base = f"{town}/generated"
        for frame in range(FRAMES):
            if frame % step:
                continue
            name = f"{frame:04d}"
            for modality in modalities:
                if modality == "lidar":
                    wanted.add(f"{base}/frames/frame_{name}.ply")
                    continue
                folder, prefix = folders[modality]
                for camera in cameras:
                    wanted.add(f"{base}/{folder}/{name}_{prefix}{camera}.png")
        for camera in cameras:
            wanted.add(f"{base}/lidar_to_cam{camera}.txt")
        wanted.add(f"{base}/full_poses_lidar.txt")
    return wanted
