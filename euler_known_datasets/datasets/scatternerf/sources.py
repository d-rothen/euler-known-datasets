"""Recognising ScatterNeRF downloads.

The release consists of ``sequence_<n>.zip`` archives holding
``<n>/{rgb,intrinsics,pose}/<camera>_<frame>_<timestamp ns>.{png,txt}``. As
published (checked 2026-11-10), most archives do not hold the sequence their
name says. They wrap another archive, often several levels deep:
``sequence_12.zip`` > ``sequence_11.zip`` > … > ``sequence_6.zip``, and
``sequence_4.zip`` > ``squence_0.zip``. Only sequences 0, 6 and 13 exist.
A sequence's number is therefore taken from the directory inside the innermost
archive, never from an archive's name, and duplicate copies are compared by
their members' sizes and CRCs.
"""

from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ...archives import member_relpath

logger = logging.getLogger(__name__)

ARCHIVE = re.compile(r"^s(?:e)?quence_(\d+)\.zip$")
MEMBER = re.compile(
    r"^(?P<sequence>\d+)/(?P<kind>rgb|intrinsics|pose)/"
    r"(?P<camera>left|right)_(?P<frame>\d+)_(?P<timestamp>\d+)\.(?P<ext>png|txt)$"
)
EXTENSIONS = {"rgb": "png", "intrinsics": "txt", "pose": "txt"}


@dataclass
class SequenceSource:
    """Where one sequence can be read from."""

    sequence: str
    #: Members as {relative path: (size, crc)}; the identity of the content.
    members: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: The archive to read it from, and the nested archive names inside it.
    archive: Optional[Path] = None
    chain: tuple[str, ...] = ()
    #: Every archive holding this sequence, outermost names.
    archives: list[str] = field(default_factory=list)
    #: The sequence is extracted below the source directory.
    extracted: bool = False


def _members(zf: zipfile.ZipFile) -> dict[str, tuple[int, int]]:
    found = {}
    for info in zf.infolist():
        relpath = member_relpath(info.filename)
        if relpath is not None and not info.is_dir() and MEMBER.match(relpath.as_posix()):
            found[relpath.as_posix()] = (info.file_size, info.CRC)
    return found


def _sequence_of(members: dict[str, tuple[int, int]]) -> str:
    sequences = {MEMBER.match(name).group("sequence") for name in members}
    if len(sequences) != 1:
        raise ValueError(f"expected one sequence per archive, found {sorted(sequences)}")
    return sequences.pop()


def open_chain(archive: Path, chain: tuple[str, ...]) -> zipfile.ZipFile:
    """Open *archive* and descend through the nested archives *chain*."""
    zf = zipfile.ZipFile(archive)
    for name in chain:
        zf = zipfile.ZipFile(io.BytesIO(zf.read(name)))
    return zf


def _inspect_archive(archive: Path) -> tuple[tuple[str, ...], dict[str, tuple[int, int]]]:
    """Follow single-archive wrapping; return the chain and the sequence members."""
    chain: list[str] = []
    zf = zipfile.ZipFile(archive)
    while True:
        files = [info for info in zf.infolist() if not info.is_dir()]
        if len(files) == 1 and files[0].filename.lower().endswith(".zip"):
            chain.append(files[0].filename)
            zf = zipfile.ZipFile(io.BytesIO(zf.read(files[0])))
            continue
        return tuple(chain), _members(zf)


def _extracted(source: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """Extracted sequences by name, from directory listings only (no stat per file)."""
    found: dict[str, dict[str, tuple[int, int]]] = {}
    for directory in sorted(p for p in source.iterdir() if p.is_dir() and p.name.isdigit()):
        members = {}
        for kind in EXTENSIONS:
            for name in list_names(directory / kind):
                relpath = f"{directory.name}/{kind}/{name}"
                if MEMBER.match(relpath):
                    members[relpath] = (0, 0)  # sizes are not compared for extracted files
        if members:
            found[directory.name] = members
    return found


def list_names(folder: Path) -> list[str]:
    try:
        with os.scandir(folder) as entries:
            return [entry.name for entry in entries]
    except (FileNotFoundError, NotADirectoryError):
        return []


def discover(source: Path) -> tuple[dict[str, SequenceSource], list[str]]:
    """Map each sequence found in *source* to where it can be read from.

    Returns the sequences and human-readable notes about duplicate archives.
    Two archives holding different content for the same sequence raise.
    """
    sequences: dict[str, SequenceSource] = {}
    notes: list[str] = []
    archives = sorted(
        (p for p in source.iterdir() if p.is_file() and ARCHIVE.match(p.name)),
        key=lambda p: int(ARCHIVE.match(p.name).group(1)),
    )
    for archive in archives:
        chain, members = _inspect_archive(archive)
        if not members:
            logger.warning("%s holds no ScatterNeRF files; ignoring it", archive.name)
            continue
        sequence = _sequence_of(members)
        named = ARCHIVE.match(archive.name).group(1)
        if named != sequence or chain:
            notes.append(
                f"{archive.name} holds sequence {sequence}"
                + (f" inside {' > '.join(chain)}" if chain else "")
            )
        entry = sequences.get(sequence)
        if entry is None:
            sequences[sequence] = SequenceSource(
                sequence, dict(members), archive, chain, [archive.name]
            )
            continue
        if entry.members != members:
            raise ValueError(
                f"{archive.name} and {entry.archives[0]} hold different content for "
                f"sequence {sequence}; remove the wrong download"
            )
        entry.archives.append(archive.name)
        # Prefer the least nested copy; it is the cheapest to read.
        if len(chain) < len(entry.chain):
            entry.archive, entry.chain = archive, chain
    for sequence, members in _extracted(source).items():
        entry = sequences.setdefault(sequence, SequenceSource(sequence))
        if not entry.members:
            entry.members = members
        if set(members) >= set(entry.members):
            entry.extracted = True
        else:
            logger.warning("%s/ is only partly extracted; reading its archive instead", sequence)
    return sequences, notes
