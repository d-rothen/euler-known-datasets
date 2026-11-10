"""A small synthetic DrivingStereo, shaped like the downloads.

Weather archives hold ``<weather>/<kind>-<size>-size/<sequence>_<timestamp>.<ext>``
as released. The inner layout of the train and test archives is not
established, so the fake uses two different ones (a sequence folder for train,
flat for test); the recipe must index both.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

FOGGY = "2018-10-25-07-37-26"  # a training sequence, as in the release
TRAIN = "2018-07-09-16-11-56"
TEST = "2018-07-11-14-48-52"
HALF = (4, 6)  # (H, W); full size is exactly twice that
FULL = (8, 12)
FRAMES = 3

# calibration/train/half-image-calib/2018-10-25-07-37-26.txt as released
HALF_CALIB = """\
calib_time: 20180820
corner_dist: 1.000000e-01
S_101: 9.600000e+2 6.000000e+2
K_101: 2.061940e+03 5.499030e-01 9.719003e+02 0.000000e+00 2.060674e+03 5.797886e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_101: -9.138600e-02 1.127830e-01 1.650000e-04 9.200000e-05
R_101: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00
T_101: 0.000000e+00 0.000000e+00 0.000000e+00
S_rect_101: 8.790000e+2 4.000000e+2
P_rect_101: 1.001496e+3 0.000000e+0 4.526664e+2 0.000000e+0 0.000000e+0 1.001496e+3 1.942570e+2 0.000000e+0 0.000000e+0 0.000000e+0 1.000000e+0 0.000000e+0
R_rect_101: 9.995644e-01 2.191951e-02 -1.976031e-02 -2.198217e-02 9.997540e-01 -2.959274e-03 1.969058e-02 3.392360e-03 9.998004e-01
S_103: 9.600000e+2 6.000000e+2
K_103: 2.061268e+03 -5.951140e-01 9.684596e+02 0.000000e+00 2.060783e+03 5.868436e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_103: -9.292800e-02 1.271090e-01 -6.620000e-04 -4.950000e-04
R_103: 9.994810e-01 2.160800e-02 -2.390500e-02 -2.176000e-02 9.997440e-01 -6.141000e-03 2.376600e-02 6.658000e-03 9.996950e-01
T_103: -5.449150e-01 -1.550000e-04 -2.260000e-03
S_rect_103: 8.790000e+2 4.000000e+2
P_rect_103: 1.001496e+3 0.000000e+0 4.526664e+2 -1.091469e+3 0.000000e+0 1.001496e+3 1.942570e+2 0.000000e+0 0.000000e+0 0.000000e+0 1.000000e+0 0.000000e+0
R_rect_103: 9.999914e-01 2.844456e-04 4.147400e-03 -2.976170e-04 9.999949e-01 3.175553e-03 -4.146476e-03 -3.176760e-03 9.999864e-01
"""
K_HALF = np.array([[1001.496, 0.0, 452.6664], [0.0, 1001.496, 194.257], [0.0, 0.0, 1.0]])


def frames(sequence: str) -> list[str]:
    day = sequence[:10]
    return [f"{day}-08-00-0{i}-{100 * (i + 1):03d}" for i in range(FRAMES)]


def _jpeg(value: int, size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((*size, 3), value, np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _png16(raw: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(raw, dtype=np.uint16)).save(buffer, format="PNG")
    return buffer.getvalue()


def _file(kind: str, size: str, index: int) -> bytes:
    """Depth (index + 10) m and disparity (index + 1) half-size px everywhere."""
    if kind.endswith("image"):
        return _jpeg(40 * index, HALF if size == "half" else FULL)
    if size == "half":
        value = 256 * (index + 10) if kind == "depth-map" else 256 * (index + 1)
        return _png16(np.full(HALF, value))
    full = np.zeros(FULL, np.uint16)
    # Full-size disparity is stored /128 in full-size pixels, twice the half-size value.
    full[0::2, 0::2] = 256 * (index + 10) if kind == "depth-map" else 128 * 2 * (index + 1)
    return _png16(full)


@dataclass
class FakeDrivingStereo:
    source: Path
    archives: dict[str, dict[str, bytes]] = field(default_factory=dict)  # zip relpath -> members
    files: dict[str, bytes] = field(default_factory=dict)  # plain files (calibration)

    def extracted_paths(self) -> dict[str, str]:
        """Media path in the prepared layout -> the archive it comes from."""
        paths = {}
        for archive, members in self.archives.items():
            directory = archive.rsplit("/", 1)[0]
            for member in members:
                if member.endswith((".jpg", ".png")):
                    paths[f"{directory}/{member}"] = archive
        return paths


def build_fake_driving_stereo(
    source: Path,
    *,
    archives: bool = True,
    extracted: bool = False,
    calibration: bool = True,
) -> FakeDrivingStereo:
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    fake = FakeDrivingStereo(source=source)

    for kind, size in [
        ("left-image", "half"),
        ("right-image", "half"),
        ("depth-map", "half"),
        ("disparity-map", "half"),
        ("depth-map", "full"),
        ("disparity-map", "full"),
    ]:
        folder = f"{kind}-{size}-size"
        ext = "jpg" if kind.endswith("image") else "png"
        members = {
            f"foggy/{folder}/{FOGGY}_{frame}.{ext}": _file(kind, size, i)
            for i, frame in enumerate(frames(FOGGY))
        }
        if folder == "left-image-half-size":
            members[f"foggy/{folder}/README.txt"] = b"not indexed"
        fake.archives[f"weather/foggy/{folder}.zip"] = members

    for kind in ("left-image", "depth-map"):
        ext = "jpg" if kind.endswith("image") else "png"
        fake.archives[f"train/{kind}/{TRAIN}.zip"] = {
            f"{TRAIN}/{TRAIN}_{frame}.{ext}": _file(kind, "half", i)
            for i, frame in enumerate(frames(TRAIN))
        }
    for size in ("half", "full"):
        fake.archives[f"test/{size}-size/left-image/{TEST}.zip"] = {
            f"{TEST}_{frame}.jpg": _file("left-image", size, i)
            for i, frame in enumerate(frames(TEST))
        }

    if calibration:
        for split, sequence in (("train", FOGGY), ("train", TRAIN), ("test", TEST)):
            fake.files[f"calibration/{split}/half-image-calib/{sequence}.txt"] = HALF_CALIB.encode()
            fake.files[f"calibration/{split}/full-image-calib/{sequence}.txt"] = b"full size"

    for relpath, content in fake.files.items():
        _write(source / relpath, content)
    for archive, members in fake.archives.items():
        if archives:
            path = source / archive
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "w") as handle:
                for name, content in members.items():
                    handle.writestr(name, content)
        if extracted:  # unzipped in the directory holding the archive
            directory = source / archive.rsplit("/", 1)[0]
            for name, content in members.items():
                _write(directory / name, content)
    return fake


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
