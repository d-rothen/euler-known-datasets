"""Recognising ACDC downloads.

ACDC keeps its anonymised images, distributed by default, as
``rgb_anon/<condition>/<split>/<sequence>/<sequence>_frame_<frame>_rgb_anon.png``.
Each one's normal-condition reference is
``rgb_anon/<condition>/<split>_ref/<sequence>/<sequence>_frame_<frame>_rgb_ref_anon.png``,
with the same sequence and frame (checked on all 4006 images, 2026-11-10).
The official ``rgb_anon_trainvaltest.zip`` holds that tree. A tarball of a
whole ACDC copy holds it next to ``gt/`` and ``lists_file_names/``, possibly
below a folder of its own. The source may hold either archive, or the
extracted tree. Archives are recognised from their first members, so a large
tarball is not read to its end just to be recognised.
"""

from __future__ import annotations

import logging
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONDITIONS = ("fog", "night", "rain", "snow")
SPLITS = ("train", "val", "test")
TYPE = "rgb_anon"
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
#: Members to look at before deciding an archive holds no ACDC images.
PEEK_MEMBERS = 50_000

#: <prefix>rgb_anon/<condition>/<split>[_ref]/<sequence>/<sequence>_frame_<frame>_rgb[_ref]_anon.png
IMAGE = re.compile(
    rf"^(?P<prefix>(?:[^/]+/)*?){TYPE}/(?P<condition>{'|'.join(CONDITIONS)})/"
    rf"(?P<split>{'|'.join(SPLITS)})(?P<ref>(?:_ref)?)/(?P<sequence>[^/]+)/"
    rf"(?P=sequence)_frame_(?P<frame>\d+)_rgb(?P=ref)_anon\.png$"
)
#: Documentation shipped with the images, extracted next to them.
DOCS = ("README.md", "License.pdf")


@dataclass
class AcdcSource:
    """Where the images can be read from."""

    #: An archive holding them, and the path prefix of ``rgb_anon/`` inside it.
    archive: Optional[Path] = None
    archive_prefix: str = ""
    #: The directory holding an extracted ``rgb_anon/``, relative to the source.
    extracted: Optional[str] = None

    def describe(self) -> list[dict]:
        inputs = []
        if self.archive is not None:
            inputs.append({"archive": self.archive.name, "images_below": f"{self.archive_prefix}{TYPE}/"})
        if self.extracted is not None:
            inputs.append({"extracted": f"{self.extracted}/{TYPE}/" if self.extracted else f"{TYPE}/"})
        return inputs


def _is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_SUFFIXES)


def _archive_prefix(path: Path) -> Optional[str]:
    """The prefix of ``rgb_anon/`` in *path*'s members, or ``None`` if it has none."""
    try:
        if path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                names = (info.filename for info in zf.infolist())
                return _first_prefix(names)
        with tarfile.open(path, mode="r|*") as tar:
            return _first_prefix(member.name for member in tar)
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        logger.warning("%s: not a readable archive (%s); ignored", path.name, exc)
        return None


def _first_prefix(names) -> Optional[str]:
    for count, name in enumerate(names):
        match = IMAGE.match(name.lstrip("./"))
        if match:
            return match.group("prefix")
        if count >= PEEK_MEMBERS:
            break
    return None


def _extracted_root(source: Path) -> Optional[str]:
    """Where an extracted ``rgb_anon/`` sits: the source or one folder below it."""
    candidates = [""] + sorted(
        entry.name for entry in os.scandir(source) if entry.is_dir() and not entry.name.startswith(".")
    )
    for relative in candidates:
        base = source / relative if relative else source
        try:
            with os.scandir(base / TYPE) as entries:
                if any(entry.is_dir() and entry.name in CONDITIONS for entry in entries):
                    return relative
        except (FileNotFoundError, NotADirectoryError):
            continue
    return None


def discover(source: Path) -> AcdcSource:
    """Find ACDC's images in *source*: an archive, an extracted tree, or both.

    Of several archives, the first (by name) that holds images is used.
    """
    source = Path(source)
    found = AcdcSource()
    for entry in sorted(os.scandir(source), key=lambda e: e.name):
        if entry.is_file() and _is_archive(entry.name):
            prefix = _archive_prefix(Path(entry.path))
            if prefix is not None:
                found.archive, found.archive_prefix = Path(entry.path), prefix
                break
    found.extracted = _extracted_root(source)
    return found

