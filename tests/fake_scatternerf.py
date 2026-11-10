"""A small synthetic ScatterNeRF, packaged like the published archives.

Each sequence holds ``<n>/{rgb,intrinsics,pose}/<camera>_<frame>_<timestamp
ns>.{png,txt}``. As in the release, archives wrap one another:
``sequence_4.zip`` holds ``squence_0.zip`` (misspelt as published), a copy of
``sequence_0.zip``; ``sequence_9.zip`` > ``sequence_8.zip`` >
``sequence_6.zip`` copies ``sequence_6.zip``; and sequence 13 exists only
inside ``sequence_14.zip``. Not every frame has both cameras, and one frame of
sequence 6 has no pose, which a prepared output must skip.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 8, 6
K = [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]
BASELINE_M = 478.200589 / 2355.722801

#: sequence -> (stored baseline in its normalized units, first timestamp,
#: {frame: cameras}).
SEQUENCES = {
    "0": (0.021, 1538982135676701005, {"06880": ("left", "right"), "06881": ("left", "right"), "06882": ("left",)}),
    "6": (0.0338, 1676358516936844849, {"00015": ("left", "right"), "00016": ("left",), "00017": ("right",)}),
    "13": (0.0381, 1676360091000000000, {"00002": ("left", "right"), "00003": ("left", "right")}),
}
#: A frame whose pose is missing from the release.
NO_POSE = ("6", "left", "00016")

#: Published archive name -> the chain of archives inside it, innermost last.
ARCHIVES = {
    "sequence_0.zip": ("0", ()),
    "sequence_4.zip": ("0", ("squence_0.zip",)),
    "sequence_6.zip": ("6", ()),
    "sequence_9.zip": ("6", ("sequence_8.zip", "sequence_6.zip")),
    "sequence_14.zip": ("13", ("sequence_13.zip",)),
}


def scale_to_meters(sequence: str) -> float:
    return round(BASELINE_M / SEQUENCES[sequence][0], 6)


def _png(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _text(matrix: np.ndarray) -> bytes:
    return (" ".join(repr(float(v)) for v in np.asarray(matrix).ravel()) + "\n").encode()


def left_pose(index: int) -> np.ndarray:
    """A camera-to-world pose driving forward and turning slowly."""
    yaw = 0.05 * index
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    pose[:3, 3] = (0.1 * index, 0.01, 0.5 * index)
    return pose


def pose(sequence: str, camera: str, index: int) -> np.ndarray:
    left = left_pose(index)
    if camera == "left":
        return left
    offset = np.eye(4)
    offset[0, 3] = SEQUENCES[sequence][0]  # the right camera sits on the left one's +x axis
    return left @ offset


@dataclass
class FakeScatterNeRF:
    source: Path
    #: <n>/<kind>/<stem>.<ext> -> content
    files: dict[str, bytes] = field(default_factory=dict)

    def sequence_files(self, sequence: str) -> dict[str, bytes]:
        return {p: c for p, c in self.files.items() if p.startswith(f"{sequence}/")}

    def frames(self, sequence: str, camera: str) -> list[str]:
        """Frames with an image, intrinsics and pose: the ones a prepared output indexes."""
        return [
            frame for frame, cameras in SEQUENCES[sequence][2].items()
            if camera in cameras and (sequence, camera, frame) != NO_POSE
        ]

    def stem(self, sequence: str, camera: str, frame: str) -> str:
        frames = list(SEQUENCES[sequence][2])
        return f"{camera}_{frame}_{SEQUENCES[sequence][1] + frames.index(frame) * 100_000_000}"


def build_fake_scatternerf(
    source: Path, *, archives: bool = True, extracted: bool = False
) -> FakeScatterNeRF:
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    fake = FakeScatterNeRF(source=source)
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    for sequence, (_, _, frames) in SEQUENCES.items():
        for index, (frame, cameras) in enumerate(frames.items()):
            for camera in cameras:
                stem = fake.stem(sequence, camera, frame)
                fake.files[f"{sequence}/rgb/{stem}.png"] = _png(index * 2 + (camera == "right"))
                fake.files[f"{sequence}/intrinsics/{stem}.txt"] = _text(intrinsics)
                if (sequence, camera, frame) != NO_POSE:
                    fake.files[f"{sequence}/pose/{stem}.txt"] = _text(pose(sequence, camera, index))

    if extracted:
        for relpath, content in fake.files.items():
            path = source / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if archives:
        for name, (sequence, chain) in ARCHIVES.items():
            data = _zip(fake.sequence_files(sequence))
            for inner in reversed(chain):
                data = _zip({inner: data})
            (source / name).write_bytes(data)
    return fake


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    directories = sorted({str(Path(n).parent) + "/" for n in members if "/" in n})
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for directory in directories:
            zf.writestr(zipfile.ZipInfo(directory), b"")
        for name, content in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o100664 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, content)
    return buffer.getvalue()
