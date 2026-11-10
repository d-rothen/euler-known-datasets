"""A small synthetic CEREMA Pixel-Accurate Depth Benchmark, packaged like the release.

The release is ``pixel_accurate_depth_benchmark.zip`` holding one stored zip per
folder in ``pixel_accurate_depth_benchmark/``. Each inner zip holds
``<folder>/<file>``, except ``calibration.zip``, whose JSON files sit at its
root. Images are tiny; the method outputs have their released shapes, since
their loaders check them. One frame lacks its gated lidar map.
"""

from __future__ import annotations

import functools
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

SCENES = ("1", "2")
TIMES_OF_DAY = ("day", "night")
WEATHERS = ("clear", "fog40", "rain15")
INDICES = ("0",)
FRAMES = tuple(
    f"scene{s}_{t}_{w}_{i}" for s in SCENES for t in TIMES_OF_DAY for w in WEATHERS for i in INDICES
)
NO_GATED_LIDAR = "scene2_night_rain15_0"
HEIGHT, WIDTH = 6, 8
P_RGB = [2355.722801, 0.0, 988.138054, 0.0, 0.0, 2355.722801, 508.051838, 0.0, 0.0, 0.0, 1.0, 0.0]
P_GATED = [2322.4, 0.0, 667.777, 0.0, 0.0, 2322.4, 261.144, 0.0, 0.0, 0.0, 1.0, 0.0]
PSMNET_DISPARITY = 25.0  # -> 0.202993 * 2355.722801 / 50 m


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _npz(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, array)
    return buffer.getvalue()


@functools.lru_cache(maxsize=None)
def _constant_npz(shape: tuple, value: float, dtype: str) -> bytes:
    """A method output at its released shape; constant, so built once."""
    return _npz(np.full(shape, value, dtype))


def _calib(P: list, width: int, height: int) -> bytes:
    return json.dumps({
        "roi": {"width": 0, "do_rectify": False, "y_offset": 0, "x_offset": 0, "height": 0},
        "D": [0.0] * 5, "P": P, "width": width, "height": height,
        "K": [P[0], 0.0, P[2], 0.0, P[5], P[6], 0.0, 0.0, 1.0],
        "R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "distortion_model": "plumb_bob",
    }).encode()


def ground_truth(scene: str) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), 5.0 * int(scene), np.float32)


def _folders() -> dict[str, dict[str, bytes]]:
    folders: dict[str, dict[str, bytes]] = {}

    def put(folder: str, name: str, content: bytes) -> None:
        folders.setdefault(folder, {})[name] = content

    rng = np.random.default_rng(0)
    for frame in FRAMES:
        scene = frame[5]
        for folder in ("rgb_left_8bit", "rgb_right_8bit"):
            put(folder, f"{frame}.png", _png(rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)))
        for folder in ("gated0_10bit", "gated17_10bit", "gated31_10bit"):
            put(folder, f"{frame}.png", _png(rng.integers(87, 1110, (HEIGHT, WIDTH), dtype=np.uint16)))
        lidar = np.zeros((HEIGHT, WIDTH), np.float32)
        lidar[2, 3] = 5.0 * int(scene)
        put("lidar_hdl64_rgb_left", f"{frame}.npz", _npz(lidar))
        if frame != NO_GATED_LIDAR:
            put("lidar_hdl64_gated", f"{frame}.npz", _npz(lidar))
        put("sgm", f"{frame}.npz", _constant_npz((412, 960), 20.0, "float32"))
        put("psmnet", f"{frame}.npz", _constant_npz((512, 960), PSMNET_DISPARITY, "float32"))
        put("monodepth", f"{frame}.npz", _constant_npz((256, 512), 0.02, "float64"))
        put("sparse2dense", f"{frame}.npz", _npz(ground_truth(scene)))
    for scene in SCENES:
        put("intermetric_rgb_left", f"scene{scene}.npz", _npz(ground_truth(scene)))
        put("intermetric_gated", f"scene{scene}.npz", _npz(ground_truth(scene)))
    put("calibration", "calib_cam_stereo_left.json", _calib(P_RGB, 1920, 1024))
    put("calibration", "calib_gated_bwv.json", _calib(P_GATED, 1280, 720))
    put("calibration", "calib_tf_tree.json", b"[]")
    return folders


@dataclass
class FakeCerema:
    source: Path
    #: folder -> {file name: content}
    folders: dict[str, dict[str, bytes]] = field(default_factory=dict)

    def paths(self, *folders: str) -> set[str]:
        return {f"{folder}/{name}" for folder in folders for name in self.folders[folder]}


def _zip(members: dict[str, bytes], compression: int = zipfile.ZIP_STORED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def inner_zip(folder: str, files: dict[str, bytes]) -> bytes:
    prefix = "" if folder == "calibration" else f"{folder}/"
    return _zip({f"{prefix}{name}": content for name, content in files.items()})


def build_fake_cerema(source: Path, *, layout: str = "outer") -> FakeCerema:
    """*layout*: ``outer`` (the release zip), ``inner`` (its inner zips unpacked
    into ``pixel_accurate_depth_benchmark/``), ``extracted`` (folders in the
    source) or ``data`` (folders in ``data/``, the benchmark code's layout)."""
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    fake = FakeCerema(source=source, folders=_folders())
    if layout == "outer":
        members = {"pixel_accurate_depth_benchmark/": b""}
        members.update({
            f"pixel_accurate_depth_benchmark/{folder}.zip": inner_zip(folder, files)
            for folder, files in fake.folders.items()
        })
        (source / "pixel_accurate_depth_benchmark.zip").write_bytes(_zip(members))
        (source / "downloads.json").write_text("{}")
    elif layout == "inner":
        for folder, files in fake.folders.items():
            path = source / "pixel_accurate_depth_benchmark" / f"{folder}.zip"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(inner_zip(folder, files))
    elif layout in ("extracted", "data"):
        base = source / "data" if layout == "data" else source
        for folder, files in fake.folders.items():
            for name, content in files.items():
                path = base / folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    else:
        raise ValueError(layout)
    return fake
