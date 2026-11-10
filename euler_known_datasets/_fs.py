"""Filesystem checks that stay fast on network filesystems.

On Lustre and GPFS every ``stat()`` is a round trip to a metadata server. On
the ETH Euler cluster one costs about 20 ms, so checking the 70,000 files of a
single-camera nuScenes test split one by one takes over twenty minutes, while
listing the handful of directories that hold them takes a fraction of a
second. :class:`DirectoryCache` answers "does this file exist?" from directory
listings, read once each; only a file's size needs a ``stat()``, and only for
files that exist.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


class DirectoryCache:
    def __init__(self) -> None:
        self._listings: dict[str, dict[str, os.DirEntry]] = {}

    def listing(self, directory: Path) -> dict[str, os.DirEntry]:
        key = str(directory)
        listing = self._listings.get(key)
        if listing is None:
            try:
                with os.scandir(directory) as entries:
                    listing = {entry.name: entry for entry in entries}
            except (FileNotFoundError, NotADirectoryError):
                listing = {}
            self._listings[key] = listing
        return listing

    def entry(self, path: Path) -> Optional[os.DirEntry]:
        return self.listing(path.parent).get(path.name)

    def is_file(self, path: Path) -> bool:
        """Whether *path* is a regular file, links followed, from the listing."""
        entry = self.entry(path)
        return entry is not None and entry.is_file()

    def size(self, path: Path) -> Optional[int]:
        """The size of the regular file at *path*, or ``None`` if there is none."""
        entry = self.entry(path)
        if entry is None or not entry.is_file():
            return None
        return entry.stat().st_size

    def forget(self, directory: Path) -> None:
        self._listings.pop(str(directory), None)


def missing_files(
    root: Path, relpaths: Iterable[str], cache: Optional[DirectoryCache] = None
) -> list[str]:
    """The *relpaths* that are not regular files below *root* (links followed)."""
    cache = cache or DirectoryCache()
    return [rel for rel in relpaths if not cache.is_file(root / rel)]


def list_files(root: Path, subdir: str = "", *, include_hidden: bool = False) -> list[str]:
    """Regular files below ``root/subdir`` as sorted paths relative to *root*.

    Walks with ``os.scandir`` (one listing per directory, no per-file stat) and
    follows directory symlinks. Hidden entries such as ``.ds_crawler`` are
    skipped unless *include_hidden*.
    """
    root = Path(root)
    start = root / subdir if subdir else root
    found: list[str] = []
    for directory, dirnames, filenames in os.walk(start, followlinks=True):
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]
        relative = Path(directory).relative_to(root)
        for name in filenames:
            found.append((relative / name).as_posix())
    return sorted(found)
