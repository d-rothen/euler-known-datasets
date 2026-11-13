"""Which SHIFT sequences and frames to prepare.

Shared by the recipe and by the tool that decodes the mirror's videos into
frames, so both pick the same frames from the same options.

A sequence's conditions are fixed (checked: no attribute of
``det_2d.json`` changes within any of the 3500 train and val sequences).
``seq.csv`` gives each sequence's fine weather and time of day; their coarse
classes are the pairs ``det_2d.json`` records. The coarse classes, with ``/``
replaced (``dawn_dusk``), form the hierarchy and the ``weather`` and
``time_of_day`` options.

Frames are the multiples of ``frame_step`` below 500. The video-rate set has
frames 0-499 annotated; the image-rate (1 fps) set keeps 0, 10, ..., 490, so
``frame_step`` must be a multiple of 10. The optional motion filter then keeps
a sequence's first candidate and each later one whose camera moved at least
``min_motion_m`` (3D distance) or turned at least ``min_rotation_deg`` (yaw)
since the last kept frame. A threshold of 0 disables its test; with both 0
every candidate is kept.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

VIEW = "front"
SPLITS = ("train", "val")
#: Annotated frames per sequence at 10 fps (0-499).
SEQUENCE_FRAMES = 500
#: The image-rate set keeps every 10th frame.
IMAGE_RATE_STEP = 10

#: Fine weather -> coarse class, as ``det_2d.json`` pairs them.
WEATHER_CLASSES = {
    "clear": "clear", "slight cloudy": "clear", "partial cloudy": "cloudy", "overcast": "overcast",
    "small rain": "rainy", "mid rain": "rainy", "heavy rain": "rainy",
    "small fog": "foggy", "heavy fog": "foggy",
}
#: Fine time of day -> coarse class.
TIME_OF_DAY_CLASSES = {
    "noon": "daytime", "morning/afternoon": "daytime",
    "dawn/dusk": "dawn/dusk", "sunrise/sunset": "dawn/dusk",
    "night": "night", "dark night": "night",
}
WEATHERS = ("clear", "cloudy", "overcast", "rainy", "foggy")
TIMES_OF_DAY = ("daytime", "dawn_dusk", "night")


def slug(value: str) -> str:
    """A hierarchy-safe form of a condition: ``dawn/dusk`` -> ``dawn_dusk``."""
    return re.sub(r"[^0-9a-z]+", "_", value.strip().lower()).strip("_")


@dataclass(frozen=True)
class SequenceInfo:
    """A sequence's row of ``seq.csv``."""

    name: str
    split: str
    town: str
    weather_fine: str
    time_of_day_fine: str

    @property
    def weather(self) -> str:
        return slug(WEATHER_CLASSES[self.weather_fine])

    @property
    def time_of_day(self) -> str:
        return slug(TIME_OF_DAY_CLASSES[self.time_of_day_fine])

    @property
    def hierarchy(self) -> tuple[str, ...]:
        return (f"split:{self.split}", f"weather:{self.weather}", f"time_of_day:{self.time_of_day}",
                f"sequence:{self.name}")


def read_sequences(path: Path, split: str) -> dict[str, SequenceInfo]:
    """The front-camera sequences of a ``seq.csv``."""
    sequences = {}
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("view", VIEW) != VIEW:
                continue
            weather, time_of_day = row["start_weather_fine"], row["start_timeofday_fine"]
            if weather not in WEATHER_CLASSES or time_of_day not in TIME_OF_DAY_CLASSES:
                raise ValueError(f"{path}: sequence {row['video']} has unknown conditions "
                                 f"{weather!r} / {time_of_day!r}")
            sequences[row["video"]] = SequenceInfo(row["video"], split, row["town"], weather, time_of_day)
    return sequences


def frame_number(name: str) -> int:
    """The frame number of a file name such as ``00000050_img_front.jpg``."""
    return int(name.split("_", 1)[0])


def iter_det2d(path: Path, *, chunk: int = 1 << 22) -> Iterator[dict[str, Any]]:
    """Stream the frame records of a Scalabel ``det_2d.json`` without their labels.

    The file is several gigabytes; it is parsed one frame object at a time.
    """
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as stream:
        buffer = stream.read(chunk)
        key = buffer.find('"frames"')
        if key < 0:
            raise ValueError(f"{path}: no 'frames' array in the first {chunk} bytes")
        position = buffer.index("[", key) + 1
        exhausted = False
        while True:
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if position < len(buffer) or exhausted:
                    break
                more = stream.read(chunk)
                buffer, position, exhausted = buffer[position:] + more, 0, not more
            if position >= len(buffer):
                raise ValueError(f"{path}: the frames array is not closed")
            if buffer[position] == "]":
                return
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if exhausted:
                    raise
                more = stream.read(chunk)
                buffer, position, exhausted = buffer[position:] + more, 0, not more
                continue
            record.pop("labels", None)
            yield record
            position = end
            if position > chunk:
                buffer, position = buffer[position:], 0


@dataclass
class Selection:
    """The frames of one split to prepare, and why the others are not."""

    split: str
    #: Sequences passing the condition filters (and the explicit list, if any).
    sequences: dict[str, SequenceInfo]
    #: Kept frame numbers per sequence, ascending.
    frames: dict[str, list[int]]
    #: The ``det_2d.json`` record of each kept frame.
    records: dict[str, dict[int, dict[str, Any]]]
    #: Sequence and frame counts after each step.
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.frames.values())


def _yaw_change(a: float, b: float) -> float:
    return abs((b - a + 180.0) % 360.0 - 180.0)


def motion_filter(poses: list[tuple[int, list[float], list[float]]], min_motion_m: float,
                  min_rotation_deg: float) -> list[int]:
    """Frames of ``(frame, location, rotation)`` (ascending) the motion filter keeps.

    The first is kept; a later one is kept when, since the last kept frame,
    the camera moved at least *min_motion_m* or its yaw (``rotation[2]``,
    degrees) changed by at least *min_rotation_deg*. Thresholds of 0 are off.
    """
    tests = [t for t in (min_motion_m, min_rotation_deg) if t > 0]
    if not tests:
        return [frame for frame, _, _ in poses]
    kept: list[int] = []
    last: Optional[tuple[list[float], list[float]]] = None
    for frame, location, rotation in poses:
        if last is not None:
            moved = math.dist(location, last[0])
            turned = _yaw_change(last[1][2], rotation[2])
            if not ((min_motion_m > 0 and moved >= min_motion_m)
                    or (min_rotation_deg > 0 and turned >= min_rotation_deg)):
                continue
        kept.append(frame)
        last = (location, rotation)
    return kept


def select(
    seq_csv: Path,
    det2d: Path,
    split: str,
    *,
    weather: Iterable[str] = (),
    time_of_day: Iterable[str] = (),
    frame_step: int = IMAGE_RATE_STEP,
    min_motion_m: float = 0.0,
    min_rotation_deg: float = 0.0,
    sequences: Optional[Iterable[str]] = None,
) -> Selection:
    """Select the frames of one split; see the module docstring."""
    if frame_step <= 0 or frame_step % IMAGE_RATE_STEP:
        raise ValueError(f"frame_step must be a positive multiple of {IMAGE_RATE_STEP}, got {frame_step}")
    weather, time_of_day = set(weather), set(time_of_day)
    if weather - set(WEATHERS) or time_of_day - set(TIMES_OF_DAY):
        raise ValueError(f"unknown conditions {sorted(weather - set(WEATHERS))} / "
                         f"{sorted(time_of_day - set(TIMES_OF_DAY))}; weather is one of "
                         f"{', '.join(WEATHERS)}, time of day one of {', '.join(TIMES_OF_DAY)}")
    every = read_sequences(seq_csv, split)
    wanted = set(sequences) if sequences is not None else None
    chosen = {
        name: info for name, info in every.items()
        if (not weather or info.weather in weather) and (not time_of_day or info.time_of_day in time_of_day)
        and (wanted is None or name in wanted)
    }
    candidates = set(range(0, SEQUENCE_FRAMES, frame_step))
    poses: dict[str, list[tuple[int, list[float], list[float]]]] = {name: [] for name in chosen}
    records: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in chosen}
    for record in iter_det2d(det2d):
        name = record.get("videoName")
        if name not in chosen:
            continue
        frame = frame_number(record["name"])
        if frame not in candidates:
            continue
        extrinsics = record["extrinsics"]
        poses[name].append((frame, list(extrinsics["location"]), list(extrinsics["rotation"])))
        records[name][frame] = record
    frames: dict[str, list[int]] = {}
    for name in chosen:
        ordered = sorted(poses[name])
        frames[name] = motion_filter(ordered, min_motion_m, min_rotation_deg)
        records[name] = {f: records[name][f] for f in frames[name]}
    counts = {
        "sequences": len(every),
        "sequences_selected": len(chosen),
        "candidate_frames": sum(len(poses[n]) for n in chosen),
        "frames_selected": sum(len(f) for f in frames.values()),
    }
    return Selection(split, chosen, frames, records, counts)


def selection_parameters(*, weather: Iterable[str], time_of_day: Iterable[str], frame_step: int,
                         min_motion_m: float, min_rotation_deg: float) -> dict[str, Any]:
    """The options that determine a selection, as recorded next to decoded frames."""
    return {"weather": sorted(weather), "time_of_day": sorted(time_of_day), "frame_step": int(frame_step),
            "min_motion_m": float(min_motion_m), "min_rotation_deg": float(min_rotation_deg)}
