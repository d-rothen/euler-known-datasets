"""Recognising SHIFT downloads (discrete view, front camera, image-rate layout).

shift-dev's ``download.py`` stores the image-rate (1 fps) set as
``discrete/images/<split>/front/{img,semseg,depth}.zip`` with
``det_2d.json`` and ``seq.csv`` beside them. Each zip holds
``<sequence>/<frame>_<group>_front.<ext>``: ``.jpg`` for ``img``, ``.png``
for the others. Extracted, each zip becomes a folder of its group's name
(shift-dev's ``FileBackend`` layout). The source may hold the zips, the
folders, or both, below a root that holds ``discrete/`` directly or one
folder down.

The official server is gone; the surviving mirror has the video-rate set only.
``slurm/tools/shift_decode.py`` (outside this package) decodes selected frames
of it into the extracted layout, with ``.png`` images and a
``decoded_from_videos.json`` describing what it did. When a split's
``det_2d.json`` or ``seq.csv`` is missing from its image-rate folder, the
video-rate one (``discrete/videos/<split>/front/``) is used: it holds every
frame of the image-rate set under the same names.

Only directory listings and zip central directories are read.
"""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from .selection import SPLITS, VIEW

logger = logging.getLogger(__name__)

GROUPS = ("img", "semseg", "depth")
EXTENSIONS = {"img": ("jpg", "png"), "semseg": ("png",), "depth": ("png",)}
METADATA = ("det_2d.json", "seq.csv")
DECODED = "decoded_from_videos.json"
SEQUENCE = re.compile(r"^[0-9a-f]{4}-[0-9a-f]{4}$")
FILE = re.compile(r"^(?P<frame>\d{8})_(?P<group>img|semseg|depth)_front\.(?P<ext>jpg|png)$")


@dataclass
class GroupSource:
    """Where one group (``img``, ``semseg`` or ``depth``) of one split can be read from."""

    group: str
    ext: str = ""
    #: ``(sequence, frame)`` of every file.
    files: set[tuple[str, int]] = field(default_factory=set)
    #: The zip and the path prefix of the sequence folders inside it.
    archive: Optional[Path] = None
    member_prefix: str = ""
    #: The extracted folder relative to the source, when it is complete.
    extracted: Optional[str] = None

    def name(self, sequence: str, frame: int) -> str:
        return f"{sequence}/{frame:08d}_{self.group}_{VIEW}.{self.ext}"

    def member(self, sequence: str, frame: int) -> str:
        return f"{self.member_prefix}{self.name(sequence, frame)}"


@dataclass
class SplitSource:
    split: str
    #: The image-rate folder relative to the source, e.g. ``discrete/images/train/front``.
    folder: str
    groups: dict[str, GroupSource] = field(default_factory=dict)
    #: ``det_2d.json`` and ``seq.csv`` relative to the source.
    det2d: Optional[str] = None
    seq_csv: Optional[str] = None
    #: The decoding tool's record, when the frames were decoded from the videos.
    decoded: Optional[dict[str, Any]] = None


@dataclass
class ShiftSource:
    splits: dict[str, SplitSource] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> list[dict]:
        inputs = []
        for split, src in sorted(self.splits.items()):
            for group, g in sorted(src.groups.items()):
                item: dict = {"split": split, "group": group, "files": len(g.files), "format": g.ext}
                if g.archive is not None:
                    item["archive"] = g.archive.name
                if g.extracted is not None:
                    item["extracted"] = f"{g.extracted}/"
                inputs.append(item)
            inputs.append({"split": split, "det_2d": src.det2d, "seq_csv": src.seq_csv,
                           "decoded_from_videos": src.decoded is not None})
        return inputs


def _entries(directory: Path) -> dict[str, bool]:
    try:
        with os.scandir(directory) as entries:
            return {entry.name: entry.is_dir() for entry in entries}
    except (FileNotFoundError, NotADirectoryError):
        return {}


def _zip_group(path: Path, group: str) -> tuple[str, str, set[tuple[str, int]]]:
    """Member prefix, extension and files of *group* in the zip *path*."""
    by_prefix: dict[tuple[str, str], set[tuple[str, int]]] = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            member = PurePosixPath(info.filename)
            match = FILE.match(member.name)
            if info.is_dir() or not match or match["group"] != group or match["ext"] not in EXTENSIONS[group]:
                continue
            if not SEQUENCE.match(member.parent.name):
                continue
            parent = member.parent.parent
            prefix = "" if str(parent) == "." else f"{parent.as_posix()}/"
            by_prefix.setdefault((prefix, match["ext"]), set()).add((member.parent.name, int(match["frame"])))
    if not by_prefix:
        return "", "", set()
    (prefix, ext) = max(by_prefix, key=lambda k: len(by_prefix[k]))
    return prefix, ext, by_prefix[(prefix, ext)]


def _extracted_group(directory: Path, group: str) -> tuple[str, set[tuple[str, int]]]:
    """Extension and files of an extracted group folder (one listing per sequence)."""
    by_ext: dict[str, set[tuple[str, int]]] = {}
    for sequence, is_dir in _entries(directory).items():
        if not is_dir or not SEQUENCE.match(sequence):
            continue
        with os.scandir(directory / sequence) as files:
            for entry in files:
                match = FILE.match(entry.name)
                if match and match["group"] == group and match["ext"] in EXTENSIONS[group]:
                    by_ext.setdefault(match["ext"], set()).add((sequence, int(match["frame"])))
    if not by_ext:
        return "", set()
    ext = max(by_ext, key=lambda e: len(by_ext[e]))
    return ext, by_ext[ext]


def _base(source: Path) -> Optional[str]:
    """Where ``discrete/`` sits: the source or one folder below it."""
    top = _entries(source)
    if top.get("discrete"):
        return ""
    for name, is_dir in sorted(top.items()):
        if is_dir and not name.startswith(".") and _entries(source / name).get("discrete"):
            return name
    return None


def discover(source: Path) -> ShiftSource:
    """Find the image-rate front-camera files of each split in *source*."""
    source = Path(source)
    found = ShiftSource()
    base = _base(source)
    if base is None:
        return found
    prefix = f"{base}/" if base else ""
    for split in SPLITS:
        folder = f"{prefix}discrete/images/{split}/{VIEW}"
        entries = _entries(source / folder)
        src = SplitSource(split, folder)
        for group in GROUPS:
            g = GroupSource(group)
            if entries.get(f"{group}.zip") is False:
                try:
                    g.member_prefix, g.ext, g.files = _zip_group(source / folder / f"{group}.zip", group)
                    g.archive = source / folder / f"{group}.zip"
                except (OSError, zipfile.BadZipFile) as exc:
                    found.notes.append(f"{folder}/{group}.zip: not a readable zip ({exc}); ignored")
            if entries.get(group):
                ext, files = _extracted_group(source / folder / group, group)
                if files and (g.archive is None or (files >= g.files and ext == g.ext)):
                    g.extracted, g.ext, g.files = f"{folder}/{group}", ext, files
                elif files:
                    found.notes.append(f"{folder}/{group}/ lacks files of {group}.zip; reading the zip instead")
            if g.files:
                src.groups[group] = g
        if not src.groups:
            continue
        videos = f"{prefix}discrete/videos/{split}/{VIEW}"
        video_entries = _entries(source / videos)
        for name in METADATA:
            if entries.get(name) is False:
                path = f"{folder}/{name}"
            elif video_entries.get(name) is False:
                path = f"{videos}/{name}"
            else:
                path = None
            setattr(src, "det2d" if name == "det_2d.json" else "seq_csv", path)
        if entries.get(DECODED) is False:
            src.decoded = json.loads((source / folder / DECODED).read_text(encoding="utf-8"))
        found.splits[split] = src
    return found
