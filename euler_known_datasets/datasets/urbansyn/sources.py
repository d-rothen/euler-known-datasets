"""Recognising UrbanSyn downloads.

UrbanSyn is published as one zip per modality, each holding a flat folder of
that name: ``rgb.zip`` holds ``rgb/rgb_<NNNN>.png``, ``depth.zip``
``depth/depth_<NNNN>.exr`` and ``ss.zip`` ``ss/ss_<NNNN>.png``, with frames
numbered 0001 to 7539. The camera description, ``camera_metadata.json``, is a
separate download. ``rgb.zip`` also carries a stray ``rgb/Thumbs.db``. The
other zips (``ss_color``, ``panoptic``, ``bbox2d``) are not used.

The source may hold the zips, their extracted folders, or both, directly or
one folder down (as in a clone of the Hugging Face copy). An extracted folder
is used only when it holds every frame of its zip. Listings only: no file is
opened except the zips' central directories.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

#: Release folders the recipe uses -> extension of their files.
FOLDERS = {"rgb": ".png", "depth": ".exr", "ss": ".png"}
CAMERA_FILE = "camera_metadata.json"
#: A file of a folder: <folder>_<frame>.<ext>.
FILE = re.compile(r"^(?P<folder>rgb|depth|ss)_(?P<frame>\d{4})\.(?P<ext>png|exr)$")


def frame_of(folder: str, name: str) -> Optional[str]:
    """The frame number of *name* if it is a file of *folder* as released."""
    match = FILE.match(name)
    if match and match.group("folder") == folder and f".{match.group('ext')}" == FOLDERS[folder]:
        return match.group("frame")
    return None


@dataclass
class FolderSource:
    """Where one release folder can be read from."""

    folder: str
    #: Frame numbers (``"0001"``) of the folder's files.
    frames: set[str] = field(default_factory=set)
    #: The zip holding them and the path of the folder's files inside it.
    archive: Optional[Path] = None
    member_prefix: str = ""
    #: The extracted folder, relative to the source, when it is complete.
    extracted: Optional[str] = None

    def name(self, frame: str) -> str:
        return f"{self.folder}_{frame}{FOLDERS[self.folder]}"

    def member(self, frame: str) -> str:
        return f"{self.member_prefix}{self.name(frame)}"


@dataclass
class UrbanSynSource:
    folders: dict[str, FolderSource] = field(default_factory=dict)
    #: ``camera_metadata.json`` relative to the source, if present.
    camera_file: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> list[dict]:
        inputs = []
        for folder, src in sorted(self.folders.items()):
            item: dict = {"folder": folder, "files": len(src.frames)}
            if src.archive is not None:
                item["archive"] = src.archive.name
            if src.extracted is not None:
                item["extracted"] = f"{src.extracted}/"
            inputs.append(item)
        if self.camera_file:
            inputs.append({"camera_metadata": self.camera_file})
        return inputs


def _entries(directory: Path) -> dict[str, bool]:
    """``{name: is_dir}`` of a directory, from one listing."""
    try:
        with os.scandir(directory) as entries:
            return {entry.name: entry.is_dir() for entry in entries}
    except (FileNotFoundError, NotADirectoryError):
        return {}


def _zip_members(path: Path, folder: str) -> tuple[str, set[str]]:
    """The member prefix of *folder*'s files in the zip *path*, and their frames."""
    by_prefix: dict[str, set[str]] = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = PurePosixPath(info.filename)
            frame = frame_of(folder, member.name)
            if frame is not None:
                prefix = "" if str(member.parent) == "." else f"{member.parent.as_posix()}/"
                by_prefix.setdefault(prefix, set()).add(frame)
    if not by_prefix:
        return "", set()
    prefix = max(by_prefix, key=lambda p: len(by_prefix[p]))
    return prefix, by_prefix[prefix]


def discover(source: Path) -> UrbanSynSource:
    """Find UrbanSyn's folders in *source*: zips, extracted folders, or both."""
    source = Path(source)
    found = UrbanSynSource()
    top = _entries(source)
    roots = [""] + sorted(name for name, is_dir in top.items()
                          if is_dir and not name.startswith(".") and name not in FOLDERS)
    extracted: dict[str, tuple[str, set[str]]] = {}
    for root in roots:
        base = source / root if root else source
        entries = top if not root else _entries(base)
        for name, is_dir in sorted(entries.items()):
            stem = name[:-4] if name.endswith(".zip") else None
            if not is_dir and stem in FOLDERS and stem not in found.folders:
                try:
                    prefix, frames = _zip_members(base / name, stem)
                except (OSError, zipfile.BadZipFile) as exc:
                    found.notes.append(f"{name}: not a readable zip ({exc}); ignored")
                    continue
                if not frames:
                    found.notes.append(f"{name} holds no {stem} files; ignored")
                    continue
                found.folders[stem] = FolderSource(stem, frames, base / name, prefix)
            elif is_dir and name in FOLDERS and name not in extracted:
                with os.scandir(base / name) as files:
                    frames = {f for entry in files for f in [frame_of(name, entry.name)] if f is not None}
                if frames:
                    extracted[name] = (str(PurePosixPath(root) / name) if root else name, frames)
            elif not is_dir and name == CAMERA_FILE and found.camera_file is None:
                found.camera_file = str(PurePosixPath(root) / name) if root else name

    for folder, (relpath, frames) in extracted.items():
        entry = found.folders.setdefault(folder, FolderSource(folder, set(frames)))
        if frames >= entry.frames:
            entry.extracted = relpath
        else:
            missing = len(entry.frames - frames)
            found.notes.append(f"{relpath}/ lacks {missing} of the {len(entry.frames)} files in "
                               f"{entry.archive.name}; reading the zip instead")
    return found
