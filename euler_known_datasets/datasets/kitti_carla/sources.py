"""Recognising KITTI-CARLA downloads.

The dataset is published as one bundle, ``KITTI-CARLA.zip``, holding
``KITTI-CARLA/dataset/TownNN.zip`` for the seven towns (about 37 GB each), the
paper and helper scripts. Each town archive holds ``TownNN/generated/`` with:

* ``images_rgb/<frame>_<camera>.png``
* ``images_ss/<frame>_1<camera>.png``
* ``images_depth/<frame>_2<camera>.png``
* ``frames/frame_<frame>.ply``
* ``lidar_to_cam0.txt``, ``lidar_to_cam1.txt``
* ``full_poses_lidar.txt``, ``full_ts_camera.txt``
* ``poses_lidar.ply``, ``screenshot.png``

A recipe source is the directory holding the town archives, the extracted
``TownNN/`` directories, or both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TOWNS = tuple(f"Town{number:02d}" for number in range(1, 8))
BUNDLE = "KITTI-CARLA.zip"

_TOWN_DIR = re.compile(r"^Town\d{2}$")
_TOWN_ZIP = re.compile(r"^(?P<town>Town\d{2})\.zip$")


@dataclass
class TownSource:
    town: str
    archive: Optional[Path] = None
    #: The extracted ``TownNN/`` directory, when it holds ``generated/``.
    extracted: Optional[Path] = None

    def describe(self) -> list[dict]:
        inputs = []
        if self.extracted is not None:
            inputs.append({"town": self.town, "kind": "extracted", "path": str(self.extracted)})
        if self.archive is not None:
            inputs.append({"town": self.town, "kind": "archive", "path": str(self.archive)})
        return inputs


def discover_towns(source: Path) -> dict[str, TownSource]:
    """Map each town found directly in *source* to its archive and/or directory."""
    found: dict[str, TownSource] = {}
    for path in sorted(source.iterdir()):
        if path.is_dir() and _TOWN_DIR.match(path.name):
            if (path / "generated").is_dir():
                found.setdefault(path.name, TownSource(path.name)).extracted = path
            continue
        match = _TOWN_ZIP.match(path.name)
        if match and path.is_file():
            found.setdefault(match["town"], TownSource(match["town"])).archive = path
    return found


def misplaced_hint(source: Path) -> str:
    """Explain where the towns are when *source* is not their directory."""
    if (source / BUNDLE).is_file():
        return (
            f"{source} holds the bundle {BUNDLE}; unzip it (it contains one zip per "
            "town) and pass its KITTI-CARLA/dataset directory"
        )
    for candidate in (source / "dataset", source / "KITTI-CARLA" / "dataset"):
        if candidate.is_dir() and discover_towns(candidate):
            return f"the towns are in {candidate}; pass that directory as the source"
    return (
        f"no KITTI-CARLA towns in {source}; expected TownNN.zip archives or "
        "extracted TownNN/generated/ directories"
    )
