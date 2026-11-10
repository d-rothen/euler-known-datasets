"""Recognising nuScenes downloads.

The official downloads are, per dataset version:

============== ==========================================================
v1.0-trainval  ``v1.0-trainval_meta.tgz`` and ``v1.0-trainval01_blobs.tgz``
               ... ``v1.0-trainval10_blobs.tgz``
v1.0-test      ``v1.0-test_meta.tgz`` and ``v1.0-test_blobs.tgz``
v1.0-mini      ``v1.0-mini.tgz`` (tables and media in one archive)
============== ==========================================================

Every archive extracts into the same top-level layout: ``v1.0-<version>/``
(the JSON tables), ``samples/``, ``sweeps/`` and ``maps/``. A source directory
may hold the archives, their extracted contents, or both.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VERSIONS = ("v1.0-trainval", "v1.0-test", "v1.0-mini")
REQUIRED_TABLES = ("sensor", "calibrated_sensor", "sample", "scene", "log", "sample_data", "ego_pose")

_ARCHIVE_SUFFIX = r"\.(?:tgz|tar\.gz|tar)"
_META = re.compile(rf"^v1\.0-(?P<version>trainval|test)_meta{_ARCHIVE_SUFFIX}$")
_BLOBS = re.compile(rf"^v1\.0-(?P<version>trainval|test)(?P<part>\d{{2}})?_blobs{_ARCHIVE_SUFFIX}$")
_FULL = re.compile(rf"^v1\.0-(?P<version>mini){_ARCHIVE_SUFFIX}$")


@dataclass
class VersionSources:
    """What the source directory offers for one nuScenes version."""

    version: str
    #: An extracted ``v1.0-<version>/`` holding every required table.
    tables_dir: Optional[Path] = None
    #: The archive holding ``v1.0-<version>/*.json``.
    meta_archive: Optional[Path] = None
    #: Archives holding ``samples/`` and ``sweeps/`` files.
    blob_archives: list[Path] = field(default_factory=list)

    @property
    def has_tables(self) -> bool:
        return self.tables_dir is not None or self.meta_archive is not None

    def describe(self) -> list[dict]:
        inputs = []
        if self.tables_dir is not None:
            inputs.append({"version": self.version, "kind": "tables", "path": str(self.tables_dir)})
        if self.meta_archive is not None:
            inputs.append(
                {"version": self.version, "kind": "meta_archive", "path": str(self.meta_archive)}
            )
        for archive in self.blob_archives:
            inputs.append({"version": self.version, "kind": "blob_archive", "path": str(archive)})
        return inputs


def _missing_tables(directory: Path) -> list[str]:
    return [name for name in REQUIRED_TABLES if not (directory / f"{name}.json").is_file()]


def discover_sources(source: Path) -> dict[str, VersionSources]:
    """Map each nuScenes version found in *source* to its inputs."""
    found: dict[str, VersionSources] = {}

    def entry(version: str) -> VersionSources:
        return found.setdefault(version, VersionSources(version))

    for path in sorted(source.iterdir()):
        name = path.name
        if path.is_dir() and name in VERSIONS:
            missing = _missing_tables(path)
            if missing:
                logger.warning("%s is missing tables %s; ignoring it", path, ", ".join(missing))
            else:
                entry(name).tables_dir = path
            continue
        if not path.is_file():
            continue
        match = _META.match(name)
        if match:
            entry(f"v1.0-{match['version']}").meta_archive = path
            continue
        match = _BLOBS.match(name)
        if match:
            entry(f"v1.0-{match['version']}").blob_archives.append(path)
            continue
        match = _FULL.match(name)
        if match:
            version = entry(f"v1.0-{match['version']}")
            version.meta_archive = path
            version.blob_archives.append(path)
    return found
