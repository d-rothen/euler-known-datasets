"""Safe, selective and resumable extraction of downloaded tar archives.

Only regular files are ever written. Members with absolute paths or ``..``
components raise :class:`UnsafeArchiveError`; links, devices and FIFOs are
skipped. Every file is written to a temporary name and renamed into place, so an
interrupted run never leaves a truncated file under its final name. A file that
already exists with the member's size is not rewritten, so re-running after an
interruption only writes what is missing.

Archives are read as a stream (``r|*``): a dataset archive is decompressed once,
front to back, whatever the selection. Selecting members saves disk space and
write time, not decompression time.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Optional

from ._fs import DirectoryCache
from ._jsonio import load_json, write_json_atomic

logger = logging.getLogger(__name__)

Selector = Callable[[str], bool]


class UnsafeArchiveError(ValueError):
    """An archive member would be written outside the destination."""


@dataclass
class ExtractionStats:
    archive: str
    members: int = 0
    extracted: int = 0
    skipped_existing: int = 0
    skipped_unselected: int = 0
    skipped_special: int = 0
    bytes_written: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def member_relpath(name: str) -> Optional[PurePosixPath]:
    """Normalise an archive member name, or raise if it escapes the destination.

    Returns ``None`` for the archive root entry (``.`` or ``./``).
    """
    if "\\" in name or "\x00" in name:
        raise UnsafeArchiveError(f"archive member {name!r} has an unsupported character")
    path = PurePosixPath(name)
    if path.is_absolute():
        raise UnsafeArchiveError(f"archive member {name!r} has an absolute path")
    parts = [part for part in path.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise UnsafeArchiveError(f"archive member {name!r} escapes the destination")
    if not parts:
        return None
    return PurePosixPath(*parts)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def extract_archive(
    archive: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    select: Optional[Selector] = None,
    overwrite: bool = False,
    log_every: float = 30.0,
) -> ExtractionStats:
    """Extract the regular files of *archive* below *destination*.

    Args:
        archive: A tar archive, optionally gzip/bz2/xz compressed.
        destination: Directory to extract into; created if missing.
        select: Called with each member's normalised relative path; members it
            rejects are skipped. ``None`` extracts every regular file.
        overwrite: Rewrite files that already exist with the member's size.
        log_every: Seconds between progress log lines.
    """
    archive = Path(archive)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    real_root = root.resolve()
    stats = ExtractionStats(archive=archive.name)
    started = last_log = time.monotonic()
    checked_dirs: set[Path] = set()
    listings = DirectoryCache()

    with tarfile.open(archive, mode="r|*") as tar:
        for member in tar:
            stats.members += 1
            relpath = member_relpath(member.name)
            if relpath is None or member.isdir():
                continue
            if not member.isfile():
                stats.skipped_special += 1
                logger.debug("skipping non-regular member %s in %s", member.name, archive.name)
                continue
            if select is not None and not select(relpath.as_posix()):
                stats.skipped_unselected += 1
                continue

            target = root.joinpath(*relpath.parts)
            parent = target.parent
            if parent not in checked_dirs:
                parent.mkdir(parents=True, exist_ok=True)
                # Never write through a symlink that points out of the destination,
                # e.g. a directory linked by an earlier ``--media symlink`` run.
                if not _is_within(parent.resolve(), real_root):
                    raise UnsafeArchiveError(
                        f"{parent} resolves outside {root}; refusing to extract "
                        f"{relpath} through a symlink"
                    )
                checked_dirs.add(parent)
            # Existence comes from one listing per directory, not a stat per member:
            # on network filesystems a stat is a server round trip.
            existing = listings.entry(target)
            if existing is not None:
                if existing.is_symlink():
                    raise UnsafeArchiveError(f"refusing to overwrite symlink {target}")
                if not overwrite and listings.size(target) == member.size:
                    stats.skipped_existing += 1
                    continue

            source = tar.extractfile(member)
            if source is None:  # pragma: no cover - isfile() members always have data
                stats.skipped_special += 1
                continue
            partial = target.with_name(f".{target.name}.part")
            with source, open(partial, "wb") as handle:
                shutil.copyfileobj(source, handle, 1 << 20)
            os.replace(partial, target)
            stats.extracted += 1
            stats.bytes_written += member.size

            now = time.monotonic()
            if now - last_log >= log_every:
                last_log = now
                logger.info(
                    "%s: %d members read, %d extracted (%.1f GB), %d already present",
                    archive.name,
                    stats.members,
                    stats.extracted,
                    stats.bytes_written / 1e9,
                    stats.skipped_existing,
                )

    stats.seconds = round(time.monotonic() - started, 3)
    return stats


def selection_fingerprint(names: Iterable[str]) -> str:
    """A stable fingerprint of a member selection, for the extraction ledger."""
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class ExtractionLedger:
    """Records which archives were fully extracted with which selection.

    A re-run skips an archive whose size, modification time and selection
    fingerprint all match a completed entry, instead of streaming it again.
    """

    def __init__(self, state_dir: str | os.PathLike[str]):
        self.directory = Path(state_dir) / "extracted"

    def _entry_path(self, archive: Path, destination: Path) -> Path:
        key = hashlib.sha256(str(destination.resolve()).encode("utf-8")).hexdigest()[:12]
        return self.directory / f"{archive.name}.{key}.json"

    @staticmethod
    def _identity(archive: Path) -> dict:
        stat = archive.stat()
        return {"name": archive.name, "size": stat.st_size, "mtime": int(stat.st_mtime)}

    def is_complete(self, archive: Path, destination: Path, fingerprint: str) -> bool:
        path = self._entry_path(archive, destination)
        if not path.is_file():
            return False
        try:
            entry = load_json(path)
        except ValueError:
            return False
        return (
            entry.get("archive") == self._identity(archive)
            and entry.get("selection") == fingerprint
        )

    def mark_complete(
        self, archive: Path, destination: Path, fingerprint: str, stats: ExtractionStats
    ) -> None:
        write_json_atomic(
            self._entry_path(archive, destination),
            {
                "archive": self._identity(archive),
                "destination": str(destination),
                "selection": fingerprint,
                "stats": stats.to_dict(),
            },
        )
