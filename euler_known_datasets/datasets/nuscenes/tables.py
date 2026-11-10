"""The parts of the nuScenes relational tables the recipe needs.

nuScenes stores everything a frame needs in JSON tables joined by tokens:
``sample_data`` (one row per image or sweep) points at ``calibrated_sensor``
(intrinsics and the sensor's mounting on the vehicle), ``ego_pose`` (the
vehicle's pose at that sample_data's timestamp) and ``sample`` (the 2 Hz
keyframe it belongs to), which points at ``scene`` and then ``log``.

``sample_data`` and ``ego_pose`` are large (about 350 MB together for
v1.0-test, several GB for v1.0-trainval), so only rows of the requested
channels are kept.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..._jsonio import load_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """One ``sample_data`` row (an image or a lidar sweep) with its ego pose."""

    token: str
    sample_token: str
    channel: str
    timestamp: int  # microseconds
    is_key_frame: bool
    filename: str  # relative to the dataset root, e.g. "sweeps/CAM_FRONT_LEFT/....jpg"
    calibrated_sensor_token: str
    width: int
    height: int
    ego_translation: tuple
    ego_rotation: tuple  # (w, x, y, z)


@dataclass(frozen=True)
class Scene:
    token: str
    name: str
    log_token: str
    logfile: str
    location: str
    vehicle: str


class NuScenesTables:
    """One version's tables, reduced to the requested channels."""

    def __init__(self, version: str, tables_dir: Path, channels: Iterable[str]):
        self.version = version
        self.tables_dir = Path(tables_dir)
        wanted = set(channels)

        def table(name: str) -> list[dict]:
            return load_json(self.tables_dir / f"{name}.json")

        sensors = {row["token"]: row for row in table("sensor")}
        self.calibrated_sensor = {row["token"]: row for row in table("calibrated_sensor")}
        available = {row["channel"] for row in sensors.values()}
        unknown = wanted - available
        if unknown:
            raise ValueError(
                f"{version} has no sensor channel(s) {', '.join(sorted(unknown))}; "
                f"available: {', '.join(sorted(available))}"
            )
        channel_of_calib = {
            token: sensors[row["sensor_token"]]["channel"]
            for token, row in self.calibrated_sensor.items()
        }
        logs = {row["token"]: row for row in table("log")}
        self.scenes = {
            row["token"]: Scene(
                token=row["token"],
                name=row["name"],
                log_token=row["log_token"],
                logfile=logs[row["log_token"]]["logfile"],
                location=logs[row["log_token"]].get("location", ""),
                vehicle=logs[row["log_token"]].get("vehicle", ""),
            )
            for row in table("scene")
        }
        self.scene_of_sample = {row["token"]: row["scene_token"] for row in table("sample")}

        rows = []
        for row in table("sample_data"):
            channel = channel_of_calib.get(row["calibrated_sensor_token"])
            if channel in wanted:
                rows.append((channel, row))
        logger.info("%s: %d sample_data rows in %s", version, len(rows), ", ".join(sorted(wanted)))

        needed_poses = {row["ego_pose_token"] for _, row in rows}
        poses = {}
        for pose in table("ego_pose"):
            if pose["token"] in needed_poses:
                poses[pose["token"]] = pose
        missing = needed_poses - poses.keys()
        if missing:
            raise ValueError(f"{version}: {len(missing)} sample_data rows have no ego_pose")

        frames: dict[tuple[str, str], list[Frame]] = defaultdict(list)
        for channel, row in rows:
            scene_token = self.scene_of_sample.get(row["sample_token"])
            if scene_token is None:
                raise ValueError(
                    f"{version}: sample_data {row['token']} points at unknown sample "
                    f"{row['sample_token']}"
                )
            pose = poses[row["ego_pose_token"]]
            frames[(scene_token, channel)].append(
                Frame(
                    token=row["token"],
                    sample_token=row["sample_token"],
                    channel=channel,
                    timestamp=int(row["timestamp"]),
                    is_key_frame=bool(row["is_key_frame"]),
                    filename=row["filename"],
                    calibrated_sensor_token=row["calibrated_sensor_token"],
                    width=int(row.get("width") or 0),
                    height=int(row.get("height") or 0),
                    ego_translation=tuple(pose["translation"]),
                    ego_rotation=tuple(pose["rotation"]),
                )
            )
        for key in frames:
            frames[key].sort(key=lambda frame: frame.timestamp)
        self._frames = dict(frames)

    def frames(self, scene_token: str, channel: str) -> list[Frame]:
        return self._frames.get((scene_token, channel), [])

    def scenes_by_name(self) -> list[Scene]:
        return sorted(self.scenes.values(), key=lambda scene: scene.name)
