"""Recognising the CEREMA Pixel-Accurate Depth Benchmark downloads.

The release is one zip, ``pixel_accurate_depth_benchmark.zip`` (26 GB), whose
``pixel_accurate_depth_benchmark/`` folder holds one stored zip per folder of
the dataset: ``rgb_left_8bit.zip`` holds ``rgb_left_8bit/<frame>.png`` and so
on, except ``calibration.zip``, whose JSON files sit at its root. The source
may hold that zip, the inner zips (directly or in a
``pixel_accurate_depth_benchmark/`` folder), their extracted folders (also
below ``data/``, the benchmark code's layout), or a mix. Listings only: no file
is opened except the archives' central directories.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional

from ...archives import open_nested_zip

logger = logging.getLogger(__name__)

#: Folders whose files are per frame: scene<1-4>_<day|night>_<weather>_<0-9>.
FRAME_FOLDERS = (
    "rgb_left_8bit",
    "rgb_right_8bit",
    "gated0_10bit",
    "gated17_10bit",
    "gated31_10bit",
    "lidar_hdl64_rgb_left",
    "lidar_hdl64_gated",
    "sgm",
    "psmnet",
    "monodepth",
    "sparse2dense",
)
#: Folders holding one ground-truth file per scene: scene<1-4>.npz.
SCENE_FOLDERS = ("intermetric_rgb_left", "intermetric_gated")
CALIBRATION = "calibration"
FOLDERS = FRAME_FOLDERS + SCENE_FOLDERS + (CALIBRATION,)
CALIBRATION_FILES = ("calib_cam_stereo_left.json", "calib_gated_bwv.json", "calib_tf_tree.json")
#: Where extracted folders may sit, relative to the source.
ROOTS = ("", "pixel_accurate_depth_benchmark", "data")

FRAME = re.compile(
    r"^scene(?P<scene>\d+)_(?P<time_of_day>day|night)_(?P<weather>clear|fog\d+|rain\d+)_(?P<index>\d+)"
    r"\.(?P<ext>png|npz)$"
)
SCENE = re.compile(r"^scene(?P<scene>\d+)\.npz$")


def belongs(folder: str, name: str) -> bool:
    """Whether *name* is a file of *folder* as released."""
    if folder == CALIBRATION:
        return name in CALIBRATION_FILES
    if folder in SCENE_FOLDERS:
        return SCENE.match(name) is not None
    match = FRAME.match(name)
    return match is not None and match.group("ext") == ("png" if folder.endswith("bit") else "npz")


@dataclass
class FolderSource:
    """Where one folder of the release can be read from."""

    folder: str
    #: File names in the folder (not paths), from an archive or a listing.
    names: set[str] = field(default_factory=set)
    #: The zip to extract from, the nested zips inside it, and the prefix of the
    #: folder's members in the innermost zip ("" for calibration.zip).
    archive: Optional[Path] = None
    inner: tuple[str, ...] = ()
    member_prefix: str = ""
    #: The extracted folder, relative to the source, when complete.
    extracted: Optional[str] = None

    @property
    def archive_label(self) -> str:
        return " > ".join([self.archive.name, *self.inner]) if self.archive else ""

    def member(self, name: str) -> str:
        return self.member_prefix + name


def list_names(folder: Path) -> list[str]:
    try:
        with os.scandir(folder) as entries:
            return [entry.name for entry in entries]
    except (FileNotFoundError, NotADirectoryError):
        return []


def _entries(folder: Path) -> dict[str, bool]:
    """``{name: is_dir}`` of a directory, from one listing."""
    try:
        with os.scandir(folder) as entries:
            return {entry.name: entry.is_dir() for entry in entries}
    except (FileNotFoundError, NotADirectoryError):
        return {}


def _folder_members(zf: zipfile.ZipFile, folder: str) -> tuple[str, set[str]]:
    """The member prefix of *folder*'s files in its own zip, and their names."""
    by_prefix: dict[str, set[str]] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if belongs(folder, path.name):
            prefix = "" if str(path.parent) == "." else f"{path.parent.as_posix()}/"
            by_prefix.setdefault(prefix, set()).add(path.name)
    if not by_prefix:
        return "", set()
    prefix = max(by_prefix, key=lambda p: len(by_prefix[p]))
    return prefix, by_prefix[prefix]


def _inner_zips(outer: zipfile.ZipFile) -> dict[str, str]:
    """``{folder: member name}`` of the per-folder zips inside an outer zip."""
    found = {}
    for info in outer.infolist():
        path = PurePosixPath(info.filename)
        if path.suffix == ".zip" and path.stem in FOLDERS:
            found.setdefault(path.stem, info.filename)
    return found


def discover(source: Path) -> tuple[dict[str, FolderSource], list[str]]:
    """Map each folder of the release found in *source* to where it can be read.

    Returns the folders and human-readable notes. An extracted folder counts
    only when it holds every file its archive does (when there is one).
    """
    source = Path(source)
    found: dict[str, FolderSource] = {}
    notes: list[str] = []
    extracted: dict[str, tuple[str, set[str]]] = {}

    def offer(folder: str, archive: Path, inner: tuple[str, ...], zf: zipfile.ZipFile) -> None:
        prefix, names = _folder_members(zf, folder)
        if not names:
            notes.append(f"{' > '.join([archive.name, *inner])} holds no {folder} files; ignored")
            return
        entry = found.get(folder)
        # An archive on its own beats one nested in another: it is cheaper to read.
        if entry is None or (entry.archive is not None and len(inner) < len(entry.inner)):
            found[folder] = FolderSource(folder, names, archive, inner, prefix)

    for root in ROOTS:
        base = source / root if root else source
        for name, is_dir in sorted(_entries(base).items()):
            path = base / name
            if is_dir and name in FOLDERS + ("calib",):
                folder = CALIBRATION if name == "calib" else name
                names = {n for n in list_names(path) if belongs(folder, n)}
                if names and folder not in extracted:
                    extracted[folder] = (str(PurePosixPath(root) / name) if root else name, names)
            elif not is_dir and name.endswith(".zip"):
                try:
                    with zipfile.ZipFile(path) as zf:
                        nested = _inner_zips(zf)
                        if Path(name).stem in FOLDERS:
                            offer(Path(name).stem, path, (), zf)
                except (OSError, zipfile.BadZipFile) as exc:
                    notes.append(f"{name}: not a readable zip ({exc}); ignored")
                    continue
                for folder, member in sorted(nested.items()):
                    if folder in found and not found[folder].inner:
                        continue  # the folder's own zip is already at hand
                    with open_nested_zip(path, (member,)) as inner_zf:
                        offer(folder, path, (member,), inner_zf)

    for folder, (relpath, names) in extracted.items():
        entry = found.setdefault(folder, FolderSource(folder, set(names)))
        if names >= entry.names:
            entry.extracted = relpath
            entry.names = set(names) if entry.archive is None else entry.names
        else:
            missing = len(entry.names - names)
            notes.append(f"{relpath}/ lacks {missing} of the {len(entry.names)} files in "
                         f"{entry.archive_label}; reading the archive instead")
    return found, notes
