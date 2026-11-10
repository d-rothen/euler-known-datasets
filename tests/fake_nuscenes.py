"""A small synthetic nuScenes, shaped like the official downloads.

Timing follows the real sensors: LIDAR_TOP sweeps end every 50 ms, a camera
fires a fixed time before the end of the sweep covering it (43 ms for
CAM_FRONT_LEFT, 35.4 ms for CAM_FRONT), keyframes are every tenth sweep. The
second scene carries the irregularities seen in v1.0-test:

* the sweep at index ``DROPPED`` is missing, so the camera frames it covered
  have no synchronised sweep;
* the sweep at index ``LATE`` is stamped ``LATE_BY_US`` late;
* a spurious sweep record sits 1 ms after the CAM_FRONT_LEFT keyframe at
  index ``SPURIOUS_AT``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

CAMERA_POSES = {
    "CAM_FRONT": ([1.70, 0.016, 1.51], [0.4998, -0.5031, 0.4998, -0.4972]),
    "CAM_FRONT_LEFT": ([1.5753, 0.5005, 1.5070], [0.6812, -0.6688, 0.2102, -0.2111]),
}
# Measured on v1.0-test: each camera fires this long before its sweep ends.
CAMERA_OFFSET_US = {
    "CAM_FRONT": 35_400,
    "CAM_FRONT_LEFT": 43_000,
    "CAM_FRONT_RIGHT": 27_400,
    "CAM_BACK": 10_300,
    "CAM_BACK_LEFT": 400,
    "CAM_BACK_RIGHT": 19_800,
}
CAMERAS = tuple(CAMERA_OFFSET_US)
LIDAR_POSE = ([0.986, 0.0, 1.840], [0.7067, -0.0153, 0.0174, -0.7071])
INTRINSICS = [[126.6, 0.0, 81.6], [0.0, 126.6, 49.1], [0.0, 0.0, 1.0]]
WIDTH, HEIGHT = 160, 90

SWEEPS = 24
KEYFRAME_SWEEPS = (2, 12, 22)
CAMERA_SWEEPS = tuple(k for k in range(1, SWEEPS) if k % 4 != 3)
DROPPED = 6
LATE = 9
LATE_BY_US = 10_000
SPURIOUS_AT = 12

TABLES = (
    "attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
    "sample", "sample_annotation", "sample_data", "scene", "sensor", "visibility",
)


def token(*parts: object) -> str:
    return hashlib.md5("/".join(map(str, parts)).encode()).hexdigest()


def _jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()


def _points(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-20, 20, (8, 5)).astype(np.float32)
    points[:, 4] = np.arange(8) % 32
    return points.tobytes()


def ego_pose(t_us: int, t0_us: int) -> tuple[list[float], list[float]]:
    """Driving at 10 m/s along x while yawing at 0.1 rad/s."""
    seconds = (t_us - t0_us) / 1e6
    yaw = 0.1 * seconds
    translation = [100.0 + 10.0 * seconds, 50.0 + 0.5 * seconds, 0.0]
    rotation = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return translation, rotation


@dataclass
class FakeNuScenes:
    source: Path
    version: str
    scenes: list[str]
    cameras: tuple[str, ...]
    tables: dict[str, list] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)
    archives: list[Path] = field(default_factory=list)

    def rows(self, channel: str, scene: Optional[str] = None) -> list[dict]:
        sensors = {r["token"]: r["channel"] for r in self.tables["sensor"]}
        calib = {r["token"]: sensors[r["sensor_token"]] for r in self.tables["calibrated_sensor"]}
        scene_of_sample = {r["token"]: r["scene_token"] for r in self.tables["sample"]}
        scene_token = None
        if scene is not None:
            scene_token = next(r["token"] for r in self.tables["scene"] if r["name"] == scene)
        rows = [
            r
            for r in self.tables["sample_data"]
            if calib[r["calibrated_sensor_token"]] == channel
            and (scene_token is None or scene_of_sample[r["sample_token"]] == scene_token)
        ]
        return sorted(rows, key=lambda r: r["timestamp"])

    def camera_time(self, scene_index: int, camera: str, sweep: int) -> int:
        return _t0(scene_index) + 50_000 * sweep - CAMERA_OFFSET_US[camera]


def _t0(scene_index: int) -> int:
    return 1_533_000_000_000_000 + scene_index * 10_000_000


def build_fake_nuscenes(
    source: Path,
    *,
    version: str = "v1.0-test",
    scenes: Optional[list[str]] = None,
    cameras: tuple[str, ...] = CAMERAS,
    archives: bool = True,
    extracted: bool = False,
    blob_parts: int = 1,
) -> FakeNuScenes:
    """Write a fake nuScenes *version* below *source* and return its description."""
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    if scenes is None:
        from euler_known_datasets.datasets.nuscenes import official_scene_splits

        official = official_scene_splits()
        scenes = {
            "v1.0-test": official["test"][:2],
            "v1.0-trainval": [official["train"][0], official["val"][0]],
            "v1.0-mini": [official["mini_train"][0], official["mini_val"][0]],
        }[version]
    fake = FakeNuScenes(source=source, version=version, scenes=list(scenes), cameras=cameras)
    tables: dict[str, list] = {name: [] for name in TABLES}
    channels = (*cameras, "LIDAR_TOP", "RADAR_FRONT")
    for channel in channels:
        modality = "camera" if channel.startswith("CAM") else channel.split("_")[0].lower()
        tables["sensor"].append(
            {"token": token("sensor", channel), "channel": channel, "modality": modality}
        )

    for index, name in enumerate(scenes):
        t0 = _t0(index)
        logfile = f"n008-2018-08-0{index + 1}-15-16-36-0400"
        log_token = token(name, "log")
        tables["log"].append(
            {"token": log_token, "logfile": logfile, "vehicle": "n008",
             "date_captured": "2018-08-01", "location": "boston-seaport"}
        )
        calib = {}
        for channel in channels:
            calib[channel] = token(name, "calib", channel)
            if channel.startswith("CAM"):
                translation, rotation = CAMERA_POSES.get(channel, CAMERA_POSES["CAM_FRONT"])
                intrinsic = INTRINSICS
            elif channel == "LIDAR_TOP":
                (translation, rotation), intrinsic = LIDAR_POSE, []
            else:
                translation, rotation, intrinsic = [3.4, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], []
            tables["calibrated_sensor"].append(
                {"token": calib[channel], "sensor_token": token("sensor", channel),
                 "translation": translation, "rotation": rotation,
                 "camera_intrinsic": intrinsic}
            )

        sample_tokens = [token(name, "sample", k) for k in KEYFRAME_SWEEPS]
        for i, k in enumerate(KEYFRAME_SWEEPS):
            tables["sample"].append(
                {"token": sample_tokens[i], "timestamp": t0 + 50_000 * k, "scene_token": token(name),
                 "prev": sample_tokens[i - 1] if i else "",
                 "next": sample_tokens[i + 1] if i + 1 < len(sample_tokens) else ""}
            )
        tables["scene"].append(
            {"token": token(name), "name": name, "log_token": log_token,
             "nbr_samples": len(sample_tokens), "first_sample_token": sample_tokens[0],
             "last_sample_token": sample_tokens[-1], "description": "fake"}
        )

        def sample_for(sweep: int) -> str:
            later = [i for i, k in enumerate(KEYFRAME_SWEEPS) if k >= sweep]
            return sample_tokens[later[0] if later else -1]

        def add(channel: str, t_us: int, key: bool, sweep: int, ext: str, content: bytes) -> None:
            folder = "samples" if key else "sweeps"
            filename = f"{folder}/{channel}/{logfile}__{channel}__{t_us}{ext}"
            sd_token = token(name, channel, t_us)
            is_camera = channel.startswith("CAM")
            tables["sample_data"].append(
                {"token": sd_token, "sample_token": sample_for(sweep), "ego_pose_token": sd_token,
                 "calibrated_sensor_token": calib[channel], "timestamp": t_us,
                 "fileformat": ext.split(".")[1], "is_key_frame": key,
                 "height": HEIGHT if is_camera else 0, "width": WIDTH if is_camera else 0,
                 "filename": filename, "prev": "", "next": ""}
            )
            translation, rotation = ego_pose(t_us, t0)
            tables["ego_pose"].append(
                {"token": sd_token, "timestamp": t_us, "translation": translation, "rotation": rotation}
            )
            fake.files[filename] = content

        for k in range(SWEEPS):
            if index == 1 and k == DROPPED:
                continue
            t_us = t0 + 50_000 * k + (LATE_BY_US if index == 1 and k == LATE else 0)
            add("LIDAR_TOP", t_us, k in KEYFRAME_SWEEPS, k, ".pcd.bin", _points(t_us % 1000))
        if index == 1 and "CAM_FRONT_LEFT" in cameras:
            spurious = fake.camera_time(index, "CAM_FRONT_LEFT", SPURIOUS_AT) + 1_000
            add("LIDAR_TOP", spurious, False, SPURIOUS_AT, ".pcd.bin", _points(1))
        for camera in cameras:
            for k in CAMERA_SWEEPS:
                t_us = fake.camera_time(index, camera, k)
                add(camera, t_us, k in KEYFRAME_SWEEPS, k, ".jpg", _jpeg(t_us % 997))
        for k in KEYFRAME_SWEEPS:
            add("RADAR_FRONT", t0 + 50_000 * k - 5_000, True, k, ".pcd", b"radar")

    fake.tables = tables
    meta = {f"{version}/{name}.json": json.dumps(rows).encode() for name, rows in tables.items()}
    meta["maps/fake_map.png"] = b"png"
    meta["LICENSE"] = b"CC BY-NC-SA 4.0 (fake)"
    meta[f".{version}_meta.txt"] = b"thanks"

    if extracted:
        for relpath, content in {**meta, **fake.files}.items():
            path = source / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if archives:
        short = version.split("-", 1)[1]
        if version == "v1.0-mini":
            fake.archives.append(_tar(source / "v1.0-mini.tgz", {**meta, **fake.files}))
        else:
            fake.archives.append(_tar(source / f"{version}_meta.tgz", meta))
            names = sorted(fake.files)
            for part in range(blob_parts):
                chunk = {n: fake.files[n] for n in names[part::blob_parts]}
                suffix = f"{part + 1:02d}" if version == "v1.0-trainval" else ""
                fake.archives.append(_tar(source / f"v1.0-{short}{suffix}_blobs.tgz", chunk))
    return fake


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        directories = sorted({str(Path(n).parent) for n in members if "/" in n})
        for directory in directories:
            info = tarfile.TarInfo(directory + "/")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tar.addfile(info)
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
    return path
