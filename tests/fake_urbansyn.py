"""A small synthetic UrbanSyn, packaged like its downloads.

The release is one zip per modality, each holding a flat folder of that name
(``rgb/rgb_<NNNN>.png``, ``depth/depth_<NNNN>.exr``, ``ss/ss_<NNNN>.png``),
plus ``camera_metadata.json``. ``rgb.zip`` carries a stray ``Thumbs.db``, as
released, and ``ss_color.zip`` stands for the zips the recipe does not use.
Frames sit on both sides of each scene boundary; one lacks its depth map, and
one is among the frames whose released depth map misses the static scene.
Images are tiny. Depth maps are real EXRs when OpenEXR is installed, with the
render camera in their headers as the release has it.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

HEIGHT, WIDTH = 4, 8
#: (frame, scene)
FRAMES = (
    ("0001", "poblenou"), ("0002", "poblenou"), ("2688", "poblenou"),
    ("2689", "torstrasse"), ("5593", "torstrasse"),
    ("5594", "veteranenstrasse"), ("7000", "veteranenstrasse"), ("7539", "veteranenstrasse"),
)
NO_DEPTH = "5593"
#: In the release, this frame's depth map lacks the static scene.
INCOMPLETE = "7000"
CAMERA = {
    "parameters": [{
        "Camera": {"focalLength_mm": 4.900001049041748, "fov_deg": 61.237205505371094,
                   "sensorWidth_mm": 5.8000001907348633, "sensorHeight_mm": 2.90000009537},
        "render": {"bitdepth": 8, "colorSpace": "sRGB", "fileType": "png", "min_samples": 128, "max_samples": 256},
    }]
}


def depth_meters(frame: str) -> np.ndarray:
    """The fake frame's depth in meters; the top row is sky that hit nothing."""
    values = np.full((HEIGHT, WIDTH), 1.5 + int(frame) / 1000, np.float32)
    values[0] = 0.0
    return values


def labels(frame: str) -> np.ndarray:
    values = np.full((HEIGHT, WIDTH), 0, np.uint8)
    values[0] = 10  # sky
    values[1, :2] = 19
    values[2, 3] = int(frame) % 19
    return values


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def camera(frame: str) -> dict:
    """The render camera a fake depth EXR's header records: 1.3 m up, looking along +z, x = frame number."""
    x = float(int(frame))
    return {"pos": {"x": x, "y": 1.3, "z": 0.0}, "target": {"x": x, "y": 1.3, "z": 1.0},
            "up": {"x": 0.0, "y": 1.0, "z": 0.0}, "focalLength": 4.900001049041748, "sensorWidth": 5.8000001907348633}


def _exr(meters: np.ndarray, frame: str) -> bytes:
    stored = np.where(meters > 0, meters / 1e5, 1.0).astype(np.float32)
    try:
        import Imath
        import OpenEXR
    except ImportError:  # the recipe never decodes depth
        return b"fake exr " + stored.tobytes()
    header = OpenEXR.Header(WIDTH, HEIGHT)
    header["channels"] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    header["compression"] = Imath.Compression(Imath.Compression.NO_COMPRESSION)
    header["octane"] = json.dumps({"renderTarget": {"camera": camera(frame)}}).encode()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "depth.exr")
        output = OpenEXR.OutputFile(path, header)
        output.writePixels({"Y": stored.tobytes()})
        output.close()
        return Path(path).read_bytes()


@dataclass
class FakeUrbanSyn:
    source: Path
    #: folder -> {file name: content}
    folders: dict[str, dict[str, bytes]] = field(default_factory=dict)

    def paths(self, *folders: str) -> set[str]:
        return {f"{folder}/{name}" for folder in folders for name in self.folders[folder]}


def _folders() -> dict[str, dict[str, bytes]]:
    rng = np.random.default_rng(0)
    folders: dict[str, dict[str, bytes]] = {"rgb": {}, "depth": {}, "ss": {}}
    for frame, _scene in FRAMES:
        folders["rgb"][f"rgb_{frame}.png"] = _png(rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8))
        if frame != NO_DEPTH:
            folders["depth"][f"depth_{frame}.exr"] = _exr(depth_meters(frame), frame)
        folders["ss"][f"ss_{frame}.png"] = _png(np.repeat(labels(frame)[..., None], 3, axis=2))
    return folders


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def build_fake_urbansyn(source: Path, *, layout: str = "zips") -> FakeUrbanSyn:
    """*layout*: ``zips`` (as downloaded), ``extracted`` (the folders in the
    source) or ``nested`` (the folders one level down, in ``UrbanSyn/``)."""
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    fake = FakeUrbanSyn(source=source, folders=_folders())
    camera = json.dumps(CAMERA, indent=4).encode()
    if layout == "zips":
        for folder, files in fake.folders.items():
            members = {f"{folder}/": b""}
            members.update({f"{folder}/{name}": content for name, content in files.items()})
            if folder == "rgb":
                members["rgb/Thumbs.db"] = b"thumbnail cache"
            (source / f"{folder}.zip").write_bytes(_zip(members))
        colour = {f"ss_color/ss_color_{frame}.png": _png(np.zeros((HEIGHT, WIDTH, 3), np.uint8)) for frame, _ in FRAMES}
        (source / "ss_color.zip").write_bytes(_zip(colour))
        (source / "camera_metadata.json").write_bytes(camera)
        (source / "downloads.json").write_text("{}")
    elif layout in ("extracted", "nested"):
        base = source / "UrbanSyn" if layout == "nested" else source
        for folder, files in fake.folders.items():
            (base / folder).mkdir(parents=True, exist_ok=True)
            for name, content in files.items():
                (base / folder / name).write_bytes(content)
        (base / "camera_metadata.json").write_bytes(camera)
    else:
        raise ValueError(layout)
    return fake
