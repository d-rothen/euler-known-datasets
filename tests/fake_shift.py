"""A small synthetic SHIFT (discrete set, front camera), in two layouts.

``layout="zipped"`` is shift-dev's image-rate download:
``discrete/images/<split>/front/{img,semseg,depth}.zip`` (JPEG images) with
``det_2d.json`` and ``seq.csv`` beside them. ``layout="decoded"`` is what
``slurm/tools/shift_decode.py`` writes from the video-rate mirror: extracted
``img/``, ``semseg/``, ``depth/`` folders with PNG images and a
``decoded_from_videos.json``, while ``det_2d.json`` and ``seq.csv`` stay in
``discrete/videos/<split>/front/``.

Every sequence has 500 annotated frames in ``det_2d.json`` (with labels, which
the recipe drops) and image-rate files for frames 0, 10, ..., 490 (frame
``HOLE`` lacks its depth map, like the mirror's holes). Depth is rendered by
ray casting a ground plane and a wall from poses given the way SHIFT gives
them: CARLA's left-handed world (x forward, y right, z up), ``rotation[2]`` the
yaw, ``rotation[1]`` the negated pitch. The renderer builds the camera from
the heading and pitch directly, not from the recipe's rotation matrices.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

HEIGHT, WIDTH = 40, 64
FOCAL = 32.0
CENTER = (32.0, 20.0)
CAMERA_HEIGHT = 1.6
WALL_X = 60.0
FAR = 1000.0
SKY, GROUND, WALL = 13, 26, 23

#: name -> (split, town, weather, time of day, motion)
SEQUENCES = {
    "0aa0-1111": ("train", "01", "clear", "noon", "straight"),
    "0bb0-2222": ("train", "02", "partial cloudy", "sunrise/sunset", "turning"),
    "0cc0-3333": ("train", "03", "heavy fog", "noon", "straight"),
    "0dd0-4444": ("train", "04", "small rain", "dark night", "parked"),
    "0ee0-5555": ("val", "05", "overcast", "morning/afternoon", "straight"),
}
FRAMES = tuple(range(0, 500, 10))
HOLE = ("0aa0-1111", 120)


def pose(motion: str, frame: int) -> tuple[list[float], list[float]]:
    """``(location, rotation)`` of a frame as det_2d.json gives them (CARLA world, degrees).

    straight: 0.1 m per frame along x, standing still from frame 200 to 299.
    turning: along a circle of radius 40 m, yaw turning right (+) 0.25 degrees per frame.
    parked: never moves.
    """
    if motion == "parked":
        return [10.0, -5.0, CAMERA_HEIGHT], [0.0, 0.0, 0.0]
    if motion == "straight":
        distance = 0.1 * (min(frame, 200) + max(0, frame - 299))
        return [distance, 0.0, CAMERA_HEIGHT], [0.0, -1.0, 0.0]
    yaw = 0.25 * frame
    radius = 40.0
    x = radius * math.sin(math.radians(yaw))
    y = radius * (1 - math.cos(math.radians(yaw)))
    return [x, y, CAMERA_HEIGHT], [0.0, -1.0, yaw]


def render(location, rotation) -> tuple[np.ndarray, np.ndarray]:
    """Planar depth (m) and labels of a ground plane z = 0 and a wall x = WALL_X, ray cast
    in CARLA's left-handed world from a camera with yaw rotation[2] and pitch -rotation[1]."""
    yaw, pitch = math.radians(rotation[2]), -math.radians(rotation[1])
    forward = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])  # y points right in CARLA's world
    up = np.array([-math.cos(yaw) * math.sin(pitch), -math.sin(yaw) * math.sin(pitch), math.cos(pitch)])
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    rays = (forward[:, None, None] + ((u - CENTER[0]) / FOCAL) * right[:, None, None]
            - ((v - CENTER[1]) / FOCAL) * up[:, None, None])  # forward component 1: t is planar depth
    origin = np.asarray(location, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = np.where(rays[2] < 0, -origin[2] / rays[2], np.inf)
        t_wall = np.where(rays[0] > 0, (WALL_X - origin[0]) / rays[0], np.inf)
    depth = np.minimum(np.minimum(t_ground, t_wall), FAR)
    labels = np.where(depth >= FAR, SKY, np.where(t_wall < t_ground, WALL, GROUND)).astype(np.uint8)
    return depth, labels


def encode_depth(depth: np.ndarray) -> np.ndarray:
    code = np.rint(np.clip(depth, 0, FAR) / (FAR / (2**24 - 1))).astype(np.uint32)
    code[depth >= FAR] = 16777210  # the sky, as SHIFT stores it
    return np.stack([code & 255, (code >> 8) & 255, (code >> 16) & 255], axis=-1).astype(np.uint8)


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpg(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def attributes(weather: str, time_of_day: str, town: str) -> dict:
    from euler_known_datasets.datasets.shift.selection import (
        TIME_OF_DAY_CLASSES,
        WEATHER_CLASSES,
    )

    return {
        "weather_coarse": WEATHER_CLASSES[weather], "timeofday_coarse": TIME_OF_DAY_CLASSES[time_of_day],
        "weather_fine": weather, "timeofday_fine": time_of_day, "view": "front", "town": town,
        "sun_altitude_angle": "45", "cloudiness": "30.0", "precipitation": "0.0",
        "precipitation_deposits": "0.0", "wind_intensity": "10.0", "sun_azimuth_angle": "0.0",
        "fog_density": "90.0" if "fog" in weather else "1.0", "fog_distance": "0.75", "wetness": "0.0",
        "fog_falloff": "0.1",
    }


@dataclass
class FakeShift:
    source: Path
    layout: str
    #: (split, sequence) -> frames with an image-rate file
    files: dict = field(default_factory=dict)


def build_fake_shift(dest: Path, *, layout: str = "zipped") -> FakeShift:
    if layout not in ("zipped", "decoded"):
        raise ValueError(layout)
    dest = Path(dest)
    fake = FakeShift(dest, layout)
    for split in ("train", "val"):
        names = [n for n, s in SEQUENCES.items() if s[0] == split]
        images = dest / "discrete/images" / split / "front"
        meta_dir = images if layout == "zipped" else dest / "discrete/videos" / split / "front"
        images.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "seq.csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["video", "view", "town", "start_weather_fine", "start_timeofday_fine"])
            for name in names:
                _, town, weather, time_of_day, _ = SEQUENCES[name]
                writer.writerow([name, "front", town, weather, time_of_day])
        frames = []
        for name in names:
            _, town, weather, time_of_day, motion = SEQUENCES[name]
            for frame in range(500):
                location, rotation = pose(motion, frame)
                frames.append({
                    "name": f"{frame:08d}_img_front.jpg", "videoName": name,
                    "intrinsics": {"focal": [FOCAL, FOCAL], "center": list(CENTER), "skew": 0},
                    "extrinsics": {"location": location, "rotation": rotation},
                    "attributes": attributes(weather, time_of_day, town), "frameIndex": frame,
                    "labels": [{"id": "7", "category": "car", "box2d": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}}],
                })
        (meta_dir / "det_2d.json").write_text(json.dumps({"frames": frames, "config": {}}))

        contents: dict[str, dict[str, bytes]] = {"img": {}, "semseg": {}, "depth": {}}
        for name in names:
            motion = SEQUENCES[name][4]
            fake.files[(split, name)] = list(FRAMES)
            for frame in FRAMES:
                depth, labels = render(*pose(motion, frame))
                shade = (np.clip(255 - depth * 4, 0, 255)).astype(np.uint8)
                rgb = np.stack([shade, shade // 2, labels * 5], axis=-1)
                ext = "jpg" if layout == "zipped" else "png"
                contents["img"][f"{name}/{frame:08d}_img_front.{ext}"] = _jpg(rgb) if ext == "jpg" else _png(rgb)
                red = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                red[..., 0] = labels
                contents["semseg"][f"{name}/{frame:08d}_semseg_front.png"] = _png(red)
                if (name, frame) != HOLE:
                    contents["depth"][f"{name}/{frame:08d}_depth_front.png"] = _png(encode_depth(depth))
        for group, members in contents.items():
            if layout == "zipped":
                with zipfile.ZipFile(images / f"{group}.zip", "w", zipfile.ZIP_STORED) as zf:
                    for member, data in sorted(members.items()):
                        zf.writestr(member, data)
            else:
                for member, data in members.items():
                    path = images / group / member
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
        if layout == "decoded":
            (images / "decoded_from_videos.json").write_text(json.dumps({
                "tool": "slurm/tools/shift_decode.py",
                "selection": {"weather": [], "time_of_day": [], "frame_step": 10,
                              "min_motion_m": 0.0, "min_rotation_deg": 0.0},
            }))
    return fake
