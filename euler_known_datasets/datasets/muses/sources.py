"""Recognising MUSES downloads.

MUSES ships one zip package per data type, e.g. ``frame_camera_trainvaltest.zip``
or ``gt_semantic_trainval.zip``. Every package extracts into the same tree, as
the dataset's README instructs (``unzip <package>.zip -d /path/to``)::

    muses/<type>/<split>/<weather>/<time_of_day>/<sequence>_frame_<frame>_<type>.<ext>
    muses/calib.json  muses/meta.json  muses/README.md  muses/License.pdf

``calib.json`` and ``meta.json`` are byte-identical copies in every package.
Packages are recognised by their members, not their names, so a renamed or
re-downloaded package still counts. A source may hold the packages, their
extracted ``muses/`` tree, or both.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ...archives import archive_members

logger = logging.getLogger(__name__)

ROOT = "muses"
CALIB = f"{ROOT}/calib.json"
META = f"{ROOT}/meta.json"
#: Files kept alongside the data when present: the dataset's own documentation.
DOCS = (f"{ROOT}/README.md", f"{ROOT}/License.pdf")

#: Package names as downloaded; only used to recognise a MUSES source cheaply.
PACKAGE_NAME = re.compile(r"^[a-z0-9_]+_(?:trainvaltest|trainval|test)\.zip$")


@dataclass
class MusesSources:
    """What a source directory offers, per data type folder (``frame_camera``…)."""

    source: Path
    #: folder -> packages holding members below ``muses/<folder>/``
    archives: dict[str, list[Path]] = field(default_factory=dict)
    #: archive -> its regular members below ``muses/``
    members: dict[Path, list[str]] = field(default_factory=dict)
    #: folders extracted at ``<source>/muses/<folder>/``
    extracted: set[str] = field(default_factory=set)
    #: a package holding ``muses/calib.json`` and ``muses/meta.json``
    side_archive: Optional[Path] = None
    #: ``<source>/muses/calib.json`` and ``meta.json`` exist
    side_extracted: bool = False

    @property
    def folders(self) -> set[str]:
        return set(self.archives) | self.extracted

    def assign(self, files: Iterable[str]) -> dict[Path, set[str]]:
        """Which package each of *files* is extracted from.

        A file several packages hold (a re-zipped copy, say) comes from the
        first of them in :attr:`archives` order, which puts officially named
        packages first.
        """
        order = list(dict.fromkeys(a for archives in self.archives.values() for a in archives))
        if self.side_archive is not None and self.side_archive not in order:
            order.append(self.side_archive)
        owner: dict[str, Path] = {}
        for archive in order:
            for name in self.members[archive]:
                owner.setdefault(name, archive)
        assignment: dict[Path, set[str]] = {}
        for name in files:
            if name in owner:
                assignment.setdefault(owner[name], set()).add(name)
        return assignment

    def archive_files(self, folder: str) -> list[str]:
        prefix = f"{ROOT}/{folder}/"
        files: set[str] = set()
        for archive in self.archives.get(folder, []):
            files.update(name for name in self.members[archive] if name.startswith(prefix))
        return sorted(files)

    def describe(self) -> list[dict]:
        inputs = []
        for archive in sorted(self.members):
            folders = sorted(f for f, archives in self.archives.items() if archive in archives)
            inputs.append({"kind": "archive", "path": str(archive), "folders": folders})
        for folder in sorted(self.extracted):
            inputs.append(
                {"kind": "extracted", "path": str(self.source / ROOT / folder), "folders": [folder]}
            )
        return inputs


def looks_like_muses(source: Path) -> bool:
    """A cheap check, by names only, for ``detect``."""
    if (source / ROOT / "calib.json").is_file():
        return True
    return any(PACKAGE_NAME.match(path.name) for path in source.glob("*.zip"))


def discover_sources(source: Path, folders: tuple[str, ...]) -> MusesSources:
    """Find the packages and extracted folders of the known *folders* in *source*.

    Zip packages list their members in a central directory, so classifying
    them reads only that, never the data.
    """
    found = MusesSources(source=source)
    # Officially named packages first: they win over copies holding the same files.
    archives = sorted(source.glob("*.zip"), key=lambda a: (not PACKAGE_NAME.match(a.name), a.name))
    for archive in archives:
        names = [name for name in archive_members(archive) if name.startswith(f"{ROOT}/")]
        held = [f for f in folders if any(name.startswith(f"{ROOT}/{f}/") for name in names)]
        has_side = CALIB in names and META in names
        if not held and not has_side:
            continue
        found.members[archive] = names
        for folder in held:
            found.archives.setdefault(folder, []).append(archive)
        if has_side and found.side_archive is None:
            found.side_archive = archive
    for folder, holders in found.archives.items():
        if len(holders) > 1:
            logger.warning(
                "%s is in several packages (%s); each file is taken from the first "
                "that holds it",
                folder,
                ", ".join(a.name for a in holders),
            )
    root = source / ROOT
    if root.is_dir():
        found.extracted = {f for f in folders if (root / f).is_dir()}
        found.side_extracted = (root / "calib.json").is_file() and (root / "meta.json").is_file()
    return found
