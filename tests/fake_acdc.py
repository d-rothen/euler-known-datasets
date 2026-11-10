"""A small synthetic ACDC, packaged like its downloads.

Images live in ``rgb_anon/<condition>/<split>/<sequence>/`` and their
normal-condition references in ``<split>_ref/``, with the same sequence and
frame. Packaging is either the official ``rgb_anon_trainvaltest.zip``, a
``tar.gz`` of a whole ACDC copy (with ``gt/`` and ``lists_file_names/`` next to
``rgb_anon/``), or the extracted tree.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

HEIGHT, WIDTH = 6, 8
#: (condition, split, sequence, frames)
SEQUENCES = (
    ("fog", "train", "GOPR0475", ("000041", "000049")),
    ("fog", "val", "GOPR0476", ("000761",)),
    ("fog", "test", "GOPR0477", ("000247",)),
    ("night", "train", "GP010376", ("000153",)),
    ("rain", "val", "GP030402", ("000022",)),
    ("snow", "test", "GP010606", ("000853", "000811")),
)


def frames(condition=None, split=None):
    return [
        (c, s, seq, f)
        for c, s, seq, fs in SEQUENCES for f in fs
        if (condition is None or c == condition) and (split is None or s == split)
    ]


def path(condition, split, sequence, frame, reference=False):
    kind = "_ref" if reference else ""
    return f"rgb_anon/{condition}/{split}{kind}/{sequence}/{sequence}_frame_{frame}_rgb{kind}_anon.png"


def _png(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class FakeAcdc:
    source: Path
    #: path below the dataset root -> content
    files: dict[str, bytes] = field(default_factory=dict)

    def images(self, reference: bool) -> set[str]:
        return {p for p in self.files if p.startswith("rgb_anon/") and ("_ref/" in p) == reference}


def build_fake_acdc(source: Path, *, layout: str = "tar") -> FakeAcdc:
    """*layout*: ``tar`` (a whole copy as ACDC.tar.gz), ``zip`` (the official
    rgb_anon_trainvaltest.zip) or ``extracted``."""
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    fake = FakeAcdc(source)
    for index, (c, s, seq, f) in enumerate(frames()):
        fake.files[path(c, s, seq, f)] = _png(2 * index)
        fake.files[path(c, s, seq, f, reference=True)] = _png(2 * index + 1)
    fake.files["README.md"] = b"# ACDC (fake)\n"
    fake.files["License.pdf"] = b"%PDF fake licence"
    extra = {
        "gt/fog/train/GOPR0475/GOPR0475_frame_000041_gt_labelIds.png": _png(99),
        "lists_file_names/rgb_anon/rgb_anon_all_train_relative_filenames.txt": b"fog/train/...\n",
    }
    if layout == "tar":
        with tarfile.open(source / "ACDC.tar.gz", "w:gz") as tar:
            for name, content in {**extra, **fake.files}.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    elif layout == "zip":
        with zipfile.ZipFile(source / "rgb_anon_trainvaltest.zip", "w") as zf:
            for name, content in fake.files.items():
                zf.writestr(name, content)
    elif layout == "extracted":
        for name, content in fake.files.items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    else:
        raise ValueError(layout)
    return fake
