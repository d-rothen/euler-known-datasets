"""A small synthetic MUSES, shaped like the official zip packages.

Every package extracts into ``muses/<type>/<split>/<weather>/<time_of_day>/``
and carries copies of ``muses/calib.json``, ``muses/meta.json``,
``README.md`` and ``License.pdf``, as the real ones do. Like the real dataset,
reference frames exist only for adverse conditions (not clear + day), and
ground truth only for train and val. The frame-camera package also holds a
stray ``.ds_crawler/`` index, as packages indexed in place do, which a prepared
output must ignore.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# (split, weather, time of day, sequence, frame)
FRAMES = (
    ("train", "clear", "day", "REC0001", "000101"),
    ("train", "clear", "day", "REC0001", "000102"),
    ("train", "fog", "night", "REC0002", "000201"),
    ("train", "fog", "night", "REC0002", "000202"),
    ("val", "clear", "day", "REC0003", "000301"),
    ("val", "rain", "day", "REC0004", "000401"),
    ("test", "snow", "night", "REC0005", "000501"),
    ("test", "snow", "night", "REC0005", "000502"),
)
WIDTH, HEIGHT = 12, 8
SKY_ROWS = 3  # the top rows are sky in every label image
K = [[10.0, 0.0, 6.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]
LIDAR_TO_RGB = [[0.0, -1.0, 0.0, 0.1], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, -0.02], [0, 0, 0, 1]]
LIDAR_TO_GNSS = [[1.0, 0.0, 0.0, 0.05], [0.0, -1.0, 0.0, 0.29], [0.0, 0.0, -1.0, -0.04], [0, 0, 0, 1]]
GNSS_LINE = (
    "123456 1676367100761000 25 85432100 473766700 512300 465400 3 17 850 1200 "
    "-120000 250000 9000000 50000 50000 120000 35000 -2000 947000 5800 -1200 "
    "30 5833 120 145 300 150\n"
)

PACKAGES = {
    "frame_camera_trainvaltest.zip": ("frame_camera",),
    "lidar_trainvaltest.zip": ("lidar",),
    "gnss_trainvaltest.zip": ("gnss",),
    "reference_frame_trainvaltest.zip": ("reference_frame",),
    "gt_semantic_trainval.zip": ("gt_semantic",),
}


def has_reference(weather: str, time_of_day: str) -> bool:
    return not (weather == "clear" and time_of_day == "day")


def has_ground_truth(split: str) -> bool:
    return split in ("train", "val")


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _rgb(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return _png(rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8))


def _labels(sky: int, ground: int) -> np.ndarray:
    labels = np.full((HEIGHT, WIDTH), ground, dtype=np.uint8)
    labels[:SKY_ROWS] = sky
    return labels


def _lidar(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    points = np.column_stack(
        [rng.uniform(2, 30, 6), rng.uniform(-5, 5, 6), rng.uniform(-1, 2, 6),
         rng.uniform(0, 255, 6), np.arange(6) % 32, 1676367100.70 + np.linspace(0, 0.085, 6)]
    )
    return points.astype(np.float64).tobytes()


def _meta() -> dict:
    meta = {}
    for split, weather, tod, seq, frame in FRAMES:
        sub = f"{split}/{weather}/{tod}/{seq}_frame_{frame}"
        base = 1676367100_000000 + int(frame) * 1000
        meta[f"{seq}_frame_{frame}"] = {
            "frame_camera_exposure_start_timestamp_us": str(base + 60_000),
            "frame_camera_exposure_end_timestamp_us": str(base + 61_000),
            "lidar_start_timestamp_us": str(base),
            "lidar_end_timestamp_us": str(base + 85_000),
            "radar_start_timestamp_us": str(base - 60_000),
            "radar_end_timestamp_us": str(base + 90_000),
            "path_to_frame_camera": f"frame_camera/{sub}_frame_camera.png",
            "path_to_lidar": f"lidar/{sub}_lidar.bin",
            "path_to_gnss": f"gnss/{sub}_gnss.txt",
            "path_to_reference_frame": f"reference_frame/{sub}_reference_frame.png"
            if has_reference(weather, tod) else "None",
            "split": split,
            "weather": weather,
            "time_of_day": tod,
        }
    return meta


@dataclass
class FakeMuses:
    source: Path
    files: dict[str, bytes] = field(default_factory=dict)  # muses/... -> content
    archives: list[Path] = field(default_factory=list)

    def folder_files(self, folder: str) -> list[str]:
        return sorted(p for p in self.files if p.startswith(f"muses/{folder}/"))


def build_fake_muses(source: Path, *, archives: bool = True, extracted: bool = False) -> FakeMuses:
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    fake = FakeMuses(source=source)
    shared = {
        "muses/calib.json": json.dumps(
            {"intrinsics": {"rgb": {"K": K, "D": [0.0] * 5}},
             "extrinsics": {"lidar2rgb": LIDAR_TO_RGB, "lidar2gnss": LIDAR_TO_GNSS}}
        ).encode(),
        "muses/meta.json": json.dumps(_meta()).encode(),
        "muses/README.md": b"# MUSES (fake)\n",
        "muses/License.pdf": b"%PDF fake licence",
    }
    per_folder: dict[str, dict[str, bytes]] = {folder: {} for (folder,) in PACKAGES.values()}
    for split, weather, tod, seq, frame in FRAMES:
        sub = f"{split}/{weather}/{tod}/{seq}_frame_{frame}"
        seed = int(frame)
        per_folder["frame_camera"][f"muses/frame_camera/{sub}_frame_camera.png"] = _rgb(seed)
        per_folder["lidar"][f"muses/lidar/{sub}_lidar.bin"] = _lidar(seed)
        per_folder["gnss"][f"muses/gnss/{sub}_gnss.txt"] = GNSS_LINE.encode()
        if has_reference(weather, tod):
            per_folder["reference_frame"][f"muses/reference_frame/{sub}_reference_frame.png"] = (
                _rgb(seed + 1)
            )
        if has_ground_truth(split):
            gt = f"muses/gt_semantic/{sub}"
            per_folder["gt_semantic"][f"{gt}_gt_labelIds.png"] = _png(_labels(23, 7))
            per_folder["gt_semantic"][f"{gt}_gt_labelTrainIds.png"] = _png(_labels(10, 0))
            color = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            color[:] = (128, 64, 128)
            color[:SKY_ROWS] = (70, 130, 180)
            per_folder["gt_semantic"][f"{gt}_gt_labelColor.png"] = _png(color)
    per_folder["gnss"]["muses/gnss/dataformat.txt"] = b"iTOW # GPS time of week\n"
    extras = {
        "frame_camera": {
            "muses/tags_train.csv": b"frame,tag\n",
            ".ds_crawler/index.json": b"{}",  # indexed in place by an earlier tool
        }
    }

    for folder_files in per_folder.values():
        fake.files.update(folder_files)
    fake.files.update(shared)

    if extracted:
        for relpath, content in fake.files.items():
            path = source / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if archives:
        for name, (folder,) in PACKAGES.items():
            members = {**shared, **per_folder[folder], **extras.get(folder, {})}
            fake.archives.append(_zip(source / name, members))
    return fake


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    directories = sorted({str(Path(n).parent) + "/" for n in members if "/" in n})
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for directory in directories:
            zf.writestr(zipfile.ZipInfo(directory), b"")
        for name, content in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o100664 << 16  # a regular file, as Unix zip writes it
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, content)
    return path
