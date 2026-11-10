"""Safe, selective and resumable extraction of downloaded archives.

Only regular files are ever written. Members with absolute paths or ``..``
components raise :class:`UnsafeArchiveError`; links, devices and FIFOs are
skipped. Every file is written to a temporary name and renamed into place, so an
interrupted run never leaves a truncated file under its final name. A file that
already exists with the member's size is not rewritten, so re-running after an
interruption only writes what is missing.

Zip archives are read through their central directory, so a selection only
reads the selected members. Tar archives are read as a stream (``r|*``): they
are decompressed once, front to back, whatever the selection, so selecting
members saves disk space and write time, not decompression time.

A zip inside a zip is read in place (``inner=``): a stored inner zip is a byte
range of its parent and is read through a window onto it, never copied. A
compressed inner zip is spooled to a temporary file first.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import shutil
import stat
import struct
import tarfile
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable, Iterator, Optional, Sequence

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


@dataclass
class _Member:
    name: str
    is_file: bool
    is_dir: bool
    size: int
    open: Callable[[], Optional[BinaryIO]]


def is_zip_archive(archive: Path) -> bool:
    return archive.suffix.lower() == ".zip" or zipfile.is_zipfile(archive)


def _tar_members(archive: Path) -> Iterator[_Member]:
    # Stream mode: each member must be consumed before the next is read.
    with tarfile.open(archive, mode="r|*") as tar:
        for member in tar:
            yield _Member(
                member.name,
                member.isfile(),
                member.isdir(),
                member.size,
                lambda member=member: tar.extractfile(member),
            )


class _Window(io.RawIOBase):
    """Read-only view of the bytes ``[start, start + size)`` of a seekable stream."""

    def __init__(self, stream: BinaryIO, start: int, size: int):
        super().__init__()
        self._stream, self._start, self._size, self._pos = stream, start, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self._size - self._pos)
        if count <= 0:
            return 0
        self._stream.seek(self._start + self._pos)
        read = self._stream.readinto(memoryview(buffer)[:count])
        self._pos += read
        return read


def _member_data_offset(stream: BinaryIO, info: zipfile.ZipInfo) -> int:
    """Offset of a member's data in its archive, after the local file header."""
    stream.seek(info.header_offset)
    header = stream.read(30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"bad local header for {info.filename}")
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_length + extra_length


@contextlib.contextmanager
def open_nested_zip(
    archive: str | os.PathLike[str],
    inner: Sequence[str] = (),
    *,
    spool_dir: Optional[str | os.PathLike[str]] = None,
) -> Iterator[zipfile.ZipFile]:
    """Open the zip reached through the member names *inner* of *archive*.

    Each name is a zip member of the previous archive. Stored members are read
    in place, through a window onto their parent; compressed ones are spooled
    to a temporary file in *spool_dir* (the system default when ``None``).
    Every member read still has its CRC checked.
    """
    with contextlib.ExitStack() as stack:
        raw: BinaryIO = stack.enter_context(open(archive, "rb", buffering=0))
        zf = stack.enter_context(zipfile.ZipFile(io.BufferedReader(raw, 1 << 20)))
        for name in inner:
            info = zf.getinfo(name)
            if info.compress_type == zipfile.ZIP_STORED and not info.flag_bits & 0x1:
                raw = _Window(raw, _member_data_offset(raw, info), info.file_size)
                zf = stack.enter_context(zipfile.ZipFile(io.BufferedReader(raw, 1 << 20)))
            else:
                spool = stack.enter_context(tempfile.TemporaryFile(dir=spool_dir))
                with zf.open(info) as member:
                    shutil.copyfileobj(member, spool, 1 << 20)
                spool.seek(0)
                raw = spool
                zf = stack.enter_context(zipfile.ZipFile(spool))
        yield zf


def _zip_members(archive: Path, inner: Sequence[str] = ()) -> Iterator[_Member]:
    with open_nested_zip(archive, inner) as zf:
        for info in zf.infolist():
            mode = info.external_attr >> 16
            is_dir = info.is_dir()
            # Unix zip tools record the file type; a link must not be extracted
            # as a regular file holding its target path. Writers that record no
            # type (only permission bits) are writing regular files.
            kind = stat.S_IFMT(mode)
            is_file = not is_dir and kind in (0, stat.S_IFREG)
            yield _Member(
                info.filename,
                is_file,
                is_dir,
                info.file_size,
                lambda info=info: zf.open(info),
            )


def _members(archive: Path, inner: Sequence[str] = ()) -> Iterator[_Member]:
    if inner or is_zip_archive(archive):
        return _zip_members(archive, inner)
    return _tar_members(archive)


def archive_label(archive: Path, inner: Sequence[str] = ()) -> str:
    """``outer.zip > inner.zip``: how logs and statistics name a nested archive."""
    return " > ".join([archive.name, *inner])


def archive_members(archive: str | os.PathLike[str], inner: Sequence[str] = ()) -> list[str]:
    """The normalised relative paths of an archive's regular files.

    Cheap for zip archives, which list their members in a central directory. A
    tar archive has to be read to its end. *inner* names a zip nested inside a
    zip archive, as for :func:`open_nested_zip`.
    """
    names = []
    for member in _members(Path(archive), tuple(inner)):
        relpath = member_relpath(member.name)
        if relpath is not None and member.is_file:
            names.append(relpath.as_posix())
    return names


def extract_archive(
    archive: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    select: Optional[Selector] = None,
    overwrite: bool = False,
    log_every: float = 30.0,
    inner: Sequence[str] = (),
) -> ExtractionStats:
    """Extract the regular files of *archive* below *destination*.

    Args:
        archive: A zip archive, or a tar archive, optionally gzip/bz2/xz
            compressed.
        destination: Directory to extract into; created if missing.
        select: Called with each member's normalised relative path; members it
            rejects are skipped. ``None`` extracts every regular file.
        overwrite: Rewrite files that already exist with the member's size.
        log_every: Seconds between progress log lines.
        inner: Member names leading to a zip nested inside the zip *archive*,
            which is then extracted instead (see :func:`open_nested_zip`).
    """
    archive = Path(archive)
    inner = tuple(inner)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    real_root = root.resolve()
    stats = ExtractionStats(archive=archive_label(archive, inner))
    started = last_log = time.monotonic()
    checked_dirs: set[Path] = set()
    listings = DirectoryCache()

    for member in _members(archive, inner):
        stats.members += 1
        relpath = member_relpath(member.name)
        if relpath is None or member.is_dir:
            continue
        if not member.is_file:
            stats.skipped_special += 1
            logger.debug("skipping non-regular member %s in %s", member.name, stats.archive)
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

        source = member.open()
        if source is None:  # pragma: no cover - regular members always have data
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
                stats.archive,
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

    def _entry_path(self, archive: Path, destination: Path, inner: Sequence[str] = ()) -> Path:
        identity = str(destination.resolve())
        if inner:
            identity += "\n" + "\n".join(inner)
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        return self.directory / f"{archive.name}.{key}.json"

    @staticmethod
    def _identity(archive: Path) -> dict:
        info = archive.stat()
        return {"name": archive.name, "size": info.st_size, "mtime": int(info.st_mtime)}

    def is_complete(
        self, archive: Path, destination: Path, fingerprint: str, inner: Sequence[str] = ()
    ) -> bool:
        path = self._entry_path(archive, destination, inner)
        if not path.is_file():
            return False
        try:
            entry = load_json(path)
        except ValueError:
            return False
        return (
            entry.get("archive") == self._identity(archive)
            and entry.get("selection") == fingerprint
            and entry.get("inner", []) == list(inner)
        )

    def mark_complete(
        self,
        archive: Path,
        destination: Path,
        fingerprint: str,
        stats: ExtractionStats,
        inner: Sequence[str] = (),
    ) -> None:
        write_json_atomic(
            self._entry_path(archive, destination, inner),
            {
                "archive": self._identity(archive),
                "inner": list(inner),
                "destination": str(destination),
                "selection": fingerprint,
                "stats": stats.to_dict(),
            },
        )
