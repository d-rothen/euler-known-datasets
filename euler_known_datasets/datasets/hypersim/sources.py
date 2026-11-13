"""Recognising Hypersim downloads.

Hypersim is released as one zip per scene, ``ai_VVV_NNN.zip``, holding
``ai_VVV_NNN/_detail/...`` and ``ai_VVV_NNN/images/...``. A source may hold
those zips, the extracted scene folders (all 1.9 TB, or the subset a partial
downloader fetched), or both; the scenes may sit directly in the source or
one folder below it (such as ``scenes/``). The split and the per-scene camera
matrices are not in the zips but in the ml-hypersim repository; the source
must hold copies of the two CSV files, at the source root, in ``metadata/``,
or at their repository paths below an ``ml-hypersim/`` checkout.
"""

from __future__ import annotations

import csv
import os
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCENE = re.compile(r"^ai_\d{3}_\d{3}$")
SCENE_ZIP = re.compile(r"^(ai_\d{3}_\d{3})\.zip$")
SPLIT_CSV = "metadata_images_split_scene_v1.csv"
CAMERA_CSV = "metadata_camera_parameters.csv"
REPOSITORY_PATHS = {
    SPLIT_CSV: "evermotion_dataset/analysis/" + SPLIT_CSV,
    CAMERA_CSV: "contrib/mikeroberts3000/" + CAMERA_CSV,
}
SPLITS = ("train", "val", "test")

#: Files of one frame, relative to the scene folder, by role.
FRAME_FILES = {
    "color": "images/scene_{camera}_final_hdf5/frame.{frame}.color.hdf5",
    "render_entity_id": "images/scene_{camera}_geometry_hdf5/frame.{frame}.render_entity_id.hdf5",
    "depth": "images/scene_{camera}_geometry_hdf5/frame.{frame}.depth_meters.hdf5",
}
#: Files of one trajectory and of one scene, relative to the scene folder.
CAMERA_FILES = ("_detail/{camera}/camera_keyframe_orientations.hdf5", "_detail/{camera}/camera_keyframe_positions.hdf5")
SCENE_FILES = ("_detail/metadata_scene.csv",)
MEMBER = re.compile(
    r"^(?P<scene>ai_\d{3}_\d{3})/images/scene_(?P<camera>cam_\d{2})_(?:final|geometry)_hdf5/"
    r"frame\.(?P<frame>\d{4})\.(?P<kind>color|render_entity_id|depth_meters)\.hdf5$"
)
_ROLE = {"color": "color", "render_entity_id": "render_entity_id", "depth_meters": "depth"}


@dataclass
class SceneSource:
    """Where one scene's files are: extracted (below ``prefix``), in a zip, or both."""

    name: str
    prefix: Optional[str] = None
    archive: Optional[Path] = None
    #: Frame files present: ``(camera, frame) -> {role: relative path inside the scene}``.
    frames: dict = field(default_factory=dict)
    #: Trajectory and scene files present, relative to the scene folder.
    files: set = field(default_factory=set)

    @property
    def extracted(self) -> bool:
        return self.prefix is not None


@dataclass
class Metadata:
    """The ml-hypersim CSV files the recipe needs."""

    split_csv: Path
    camera_csv: Path

    def public_frames(self) -> dict[str, dict[tuple[str, str], str]]:
        """``scene -> {(camera, frame): split}`` for the publicly released images."""
        frames: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
        with open(self.split_csv, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["included_in_public_release"] == "True" and row["split_partition_name"] in SPLITS:
                    frames[row["scene_name"]][(row["camera_name"], f"{int(row['frame_id']):04d}")] = (
                        row["split_partition_name"]
                    )
        return dict(frames)

    def camera_rows(self) -> list[dict[str, str]]:
        with open(self.camera_csv, newline="") as handle:
            return list(csv.DictReader(handle))


def find_metadata(source: Path) -> tuple[Optional[Path], Optional[Path]]:
    found = []
    for name in (SPLIT_CSV, CAMERA_CSV):
        candidates = [source / name, source / "metadata" / name, source / "ml-hypersim" / REPOSITORY_PATHS[name],
                      source / REPOSITORY_PATHS[name]]
        found.append(next((path for path in candidates if path.is_file()), None))
    return found[0], found[1]


def _scene_dirs(base: Path) -> list[str]:
    try:
        with os.scandir(base) as entries:
            return sorted(e.name for e in entries if e.is_dir() and SCENE.match(e.name))
    except (FileNotFoundError, NotADirectoryError):
        return []


def _extracted_prefix(source: Path) -> Optional[str]:
    """Where the extracted scene folders are: the source or one folder below it."""
    if _scene_dirs(source):
        return ""
    with os.scandir(source) as entries:
        folders = sorted(e.name for e in entries if e.is_dir() and not e.name.startswith("."))
    return next((folder for folder in folders if _scene_dirs(source / folder)), None)


def _list(directory: Path) -> list[str]:
    try:
        with os.scandir(directory) as entries:
            return [e.name for e in entries if e.is_file()]
    except (FileNotFoundError, NotADirectoryError):
        return []


def _scan_extracted(scene: SceneSource, base: Path) -> None:
    """Fill *scene*'s file sets from directory listings (no per-file stat)."""
    root = base / scene.name
    for name in _list(root / "_detail"):
        scene.files.add(f"_detail/{name}")
    try:
        with os.scandir(root / "_detail") as entries:
            cameras = sorted(e.name for e in entries if e.is_dir() and e.name.startswith("cam_"))
    except FileNotFoundError:
        cameras = []
    for camera in cameras:
        for name in _list(root / "_detail" / camera):
            scene.files.add(f"_detail/{camera}/{name}")
    try:
        with os.scandir(root / "images") as entries:
            folders = sorted(e.name for e in entries if e.is_dir())
    except FileNotFoundError:
        folders = []
    for folder in folders:
        for name in _list(root / "images" / folder):
            _add_member(scene, f"{scene.name}/images/{folder}/{name}")


def _add_member(scene: SceneSource, name: str) -> None:
    match = MEMBER.match(name)
    if match is None or match.group("scene") != scene.name:
        return
    key = (match.group("camera"), match.group("frame"))
    scene.frames.setdefault(key, {})[_ROLE[match.group("kind")]] = name.split("/", 1)[1]


def _scan_zip(scene: SceneSource) -> None:
    with zipfile.ZipFile(scene.archive) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith(f"{scene.name}/_detail/"):
                scene.files.add(name.split("/", 1)[1])
            else:
                _add_member(scene, name)


def discover(source: Path, *, scan: Optional[set] = None) -> tuple[dict[str, SceneSource], Optional[str]]:
    """Find the scenes in *source*, extracted or zipped, and list their files.

    *scan* limits the file listing to those scenes (all when ``None``); the
    others are only named. Returns the scenes by name and the folder holding
    the extracted ones (``""`` for the source itself, ``None`` if there are none).
    """
    source = Path(source)
    scenes: dict[str, SceneSource] = {}
    prefix = _extracted_prefix(source)
    if prefix is not None:
        base = source / prefix if prefix else source
        for name in _scene_dirs(base):
            scenes[name] = SceneSource(name, prefix=prefix)
    for folder in ("", "downloads"):
        for name in sorted(_list(source / folder if folder else source)):
            match = SCENE_ZIP.match(name)
            if match:
                scene = scenes.setdefault(match.group(1), SceneSource(match.group(1)))
                if scene.archive is None:
                    scene.archive = source / folder / name if folder else source / name
    for name, scene in scenes.items():
        if scan is not None and name not in scan:
            continue
        # An extracted scene is listed from disk; its zip, if any, holds the
        # same members. A zipped-only scene is listed from its zip.
        if scene.extracted:
            _scan_extracted(scene, source / scene.prefix if scene.prefix else source)
        elif scene.archive is not None:
            _scan_zip(scene)
    return scenes, prefix


def read_scene_file(source: Path, scene: SceneSource, rel: str) -> bytes:
    """The bytes of *rel* (relative to the scene folder), extracted or from the zip."""
    if scene.extracted:
        base = Path(source) / scene.prefix if scene.prefix else Path(source)
        return (base / scene.name / rel).read_bytes()
    with zipfile.ZipFile(scene.archive) as archive:
        return archive.read(f"{scene.name}/{rel}")
