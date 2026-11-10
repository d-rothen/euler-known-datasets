"""Getting a dataset's media files into the output directory.

``prepare`` never decodes media; it only decides where the files a prepared
dataset references come from:

=========== ================================================================
``auto``     ``symlink`` when the source already holds every needed file
             extracted, ``extract`` otherwise.
``symlink``  Link the source's top-level directories into the output. Instant
             and free, but the output breaks if the source moves.
``hardlink`` Hard-link every needed file. Free and independent of the source
             directory's lifetime, but needs the same filesystem.
``copy``     Copy every needed file.
``extract``  Extract only the needed files from the downloaded archives.
=========== ================================================================

When the output *is* the source directory the dataset is prepared in place:
nothing is moved and only missing files are extracted.
"""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path
from typing import Iterable

from ._fs import DirectoryCache, missing_files

__all__ = [
    "IN_PLACE",
    "MEDIA_MODES",
    "MediaError",
    "link_top_level",
    "materialize_files",
    "missing_files",
]

MEDIA_MODES = ("auto", "symlink", "hardlink", "copy", "extract")
IN_PLACE = "in_place"


class MediaError(RuntimeError):
    """The requested media strategy cannot produce the output."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def link_top_level(source: Path, output: Path, names: Iterable[str]) -> list[str]:
    """Symlink ``source/<name>`` as ``output/<name>``; return the names linked.

    Entries missing from the source are skipped. An existing link to the same
    target is kept; anything else already at the destination is an error, so a
    previous ``extract`` or ``copy`` run is never shadowed or overwritten.
    """
    linked = []
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = source / name
        dst = output / name
        if not src.exists():
            continue
        if dst.is_symlink():
            if dst.resolve() == src.resolve():
                linked.append(name)
                continue
            raise MediaError(f"{dst} links to {os.readlink(dst)}, not {src}")
        if dst.exists():
            raise MediaError(
                f"{dst} already exists and is not a link to {src}. It was probably "
                "written by an earlier run with another --media mode; prepare into a "
                "fresh output or keep that mode."
            )
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
        linked.append(name)
    return linked


def materialize_files(
    source: Path, output: Path, relpaths: Iterable[str], mode: str
) -> dict[str, int]:
    """Hard-link or copy each of *relpaths* from *source* into *output*.

    Files already present with the source's size are left alone, so an
    interrupted run resumes where it stopped.
    """
    if mode not in ("hardlink", "copy"):
        raise ValueError(f"materialize_files handles hardlink and copy, not {mode!r}")
    counts = {"linked" if mode == "hardlink" else "copied": 0, "present": 0}
    key = "linked" if mode == "hardlink" else "copied"
    checked: set[Path] = set()
    sources = DirectoryCache()
    targets = DirectoryCache()
    for rel in relpaths:
        src = source / rel
        dst = output / rel
        if dst.parent not in checked:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not _inside(dst.parent, output):
                raise MediaError(
                    f"{dst.parent} resolves outside {output}, probably through a link "
                    "from an earlier --media symlink run; prepare into a fresh output"
                )
            checked.add(dst.parent)
        existing = targets.entry(dst)
        if existing is not None:
            if existing.is_file() and targets.size(dst) == sources.size(src):
                counts["present"] += 1
                continue
            dst.unlink()
        if mode == "hardlink":
            try:
                os.link(src, dst)
            except OSError as exc:
                if exc.errno == errno.EXDEV:
                    raise MediaError(
                        f"cannot hard-link {src} into {output}: different filesystems. "
                        "Use --media copy or --media symlink."
                    ) from exc
                raise
        else:
            partial = dst.with_name(f".{dst.name}.part")
            shutil.copy2(src, partial)
            os.replace(partial, dst)
        counts[key] += 1
    return counts
