"""Where DrivingStereo keeps what, and how it becomes ds-crawler scopes.

The downloads (as mirrored on Hugging Face, file tree checked 2026-11-10):

====================================================== =======================
``train/<kind>/<sequence>.zip``                        training set, half size
``test/half-size/<kind>/<sequence>.zip``               test set, half size
``test/full-size/<kind>/<sequence>.zip``               test set, full size
``weather/<weather>/<kind>-<half|full>-size.zip``      weather subsets
``calibration/<train|test>/<half|full>-image-calib/<sequence>.txt``
====================================================== =======================

``<kind>`` is ``left-image``, ``right-image``, ``depth-map`` or
``disparity-map``; ``<weather>`` is ``sunny``, ``cloudy``, ``foggy`` or
``rainy``. Every image and map is named ``<sequence>_<timestamp>.<jpg|png>``.
A weather archive holds ``<weather>/<kind>-<size>-size/<name>``.

The prepared layout keeps the download's relative paths and puts each archive's
members in the directory that holds the archive, which is what unzipping in
place gives. The scope regexes therefore accept any directories between the
``<kind>`` folder and the file name, so they also match train and test archives
whose inner layout differs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

SEQUENCE = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
TIMESTAMP = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}"
WEATHERS = ("sunny", "cloudy", "foggy", "rainy")
SPLITS = ("train", "test")
SUBSETS = SPLITS + WEATHERS
KINDS = ("left-image", "right-image", "depth-map", "disparity-map")
EXTENSIONS = {"left-image": "jpg", "right-image": "jpg", "depth-map": "png", "disparity-map": "png"}

_W = "|".join(WEATHERS)
_SUBSET = "|".join(SUBSETS)
_KIND = "|".join(KINDS)

# uint16 PNG maps: value / 256 (half-size depth and disparity, full-size depth)
# or value / 128 (full-size disparity), with 0 marking pixels without lidar.
_SPARSE = {"invalid": {"non_finite": False, "sentinel": 0}, "sparse": True}
DEPTH_META = {
    "radial_depth": False,
    "scale_to_meters": 1 / 256,
    "range": [0, 65535],
    **_SPARSE,
}
DISPARITY_META = {"range": [0, 65535], **_SPARSE}


@dataclass(frozen=True)
class Stream:
    """One kind of file at one resolution, written as one scope."""

    scope: str
    kind: str
    size: str  # "half" or "full"
    key: str
    function: str
    title: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def extension(self) -> str:
        return EXTENSIONS[self.kind]

    def _where(self) -> tuple[str, str]:
        """Path prefixes up to the folder holding the files: (id regex, hierarchy regex)."""
        kind = self.kind
        if self.size == "half":
            strict = (
                rf"(?:train/{kind}/|test/half-size/{kind}/"
                rf"|weather/(?:{_W})/(?:[^/]+/)*?{kind}-half-size/)"
            )
            loose = rf"(?:(?:half-size/)?{kind}/|(?:[^/]+/)*?{kind}-half-size/)"
        else:
            strict = (
                rf"(?:test/full-size/{kind}/|weather/(?:{_W})/(?:[^/]+/)*?{kind}-full-size/)"
            )
            loose = rf"(?:full-size/{kind}/|(?:[^/]+/)*?{kind}-full-size/)"
        return strict, loose

    @property
    def id_regex(self) -> str:
        strict, _ = self._where()
        return rf"^{strict}(?:[^/]+/)*{SEQUENCE}_(?P<frame>{TIMESTAMP})\.{self.extension}$"

    @property
    def hierarchy_regex(self) -> str:
        _, loose = self._where()
        return (
            rf"^(?:weather/)?(?P<subset>{_SUBSET})/{loose}(?:[^/]+/)*"
            rf"(?P<sequence>{SEQUENCE})_{TIMESTAMP}\.{self.extension}$"
        )

    def indexing(self) -> dict[str, Any]:
        return {
            "id": {"regex": self.id_regex, "join_char": "+"},
            "hierarchy": {"regex": self.hierarchy_regex, "separator": ":"},
            "files": {"extensions": [f".{self.extension}"]},
        }

    def matches(self, relpath: str) -> bool:
        return _compiled(self.id_regex).match(relpath) is not None


_REGEX_CACHE: dict[str, re.Pattern] = {}


def _compiled(pattern: str) -> re.Pattern:
    compiled = _REGEX_CACHE.get(pattern)
    if compiled is None:
        compiled = _REGEX_CACHE[pattern] = re.compile(pattern)
    return compiled


STREAMS = (
    Stream("rgb", "left-image", "half", "rgb", "rgb", "left images",
           {"range": [0, 255], "resolution": "half"}),
    Stream("rgb_right", "right-image", "half", "rgb", "rgb", "right images",
           {"range": [0, 255], "resolution": "half"}),
    Stream("depth", "depth-map", "half", "depth", "depth", "depth maps",
           {**DEPTH_META, "resolution": "half"}),
    Stream("disparity", "disparity-map", "half", "disparity", "disparity", "disparity maps",
           {**DISPARITY_META, "resolution": "half"}),
    Stream("rgb_full_size", "left-image", "full", "rgb", "rgb", "full-size left images",
           {"range": [0, 255], "resolution": "full"}),
    Stream("rgb_right_full_size", "right-image", "full", "rgb", "rgb", "full-size right images",
           {"range": [0, 255], "resolution": "full"}),
    Stream("depth_full_size", "depth-map", "full", "depth", "depth_full_size",
           "full-size depth maps", {**DEPTH_META, "resolution": "full"}),
    Stream("disparity_full_size", "disparity-map", "full", "disparity", "disparity_full_size",
           "full-size disparity maps", {**DISPARITY_META, "resolution": "full"}),
)
INTRINSICS_SCOPE = "intrinsics"
SCOPES = tuple(stream.scope for stream in STREAMS) + (INTRINSICS_SCOPE,)


@dataclass(frozen=True)
class ArchiveInfo:
    """A recognised download archive."""

    relpath: str  # relative to the source
    subset: str
    stream: Stream

    @property
    def directory(self) -> str:
        """Where its members go: the directory holding the archive."""
        return self.relpath.rsplit("/", 1)[0] if "/" in self.relpath else ""


_TRAIN = re.compile(rf"^train/(?P<kind>{_KIND})/{SEQUENCE}\.zip$")
_TEST = re.compile(rf"^test/(?P<size>half|full)-size/(?P<kind>{_KIND})/{SEQUENCE}\.zip$")
_WEATHER = re.compile(rf"^weather/(?P<weather>{_W})/(?P<kind>{_KIND})-(?P<size>half|full)-size\.zip$")
_CALIBRATION = re.compile(
    rf"^calibration/(?P<split>train|test)/half-image-calib/(?P<sequence>{SEQUENCE})\.txt$"
)


def _stream(kind: str, size: str) -> Stream:
    return next(s for s in STREAMS if s.kind == kind and s.size == size)


def classify_archive(relpath: str) -> Optional[ArchiveInfo]:
    """The subset and stream of a download archive, or ``None``."""
    match = _TRAIN.match(relpath)
    if match:
        return ArchiveInfo(relpath, "train", _stream(match["kind"], "half"))
    match = _TEST.match(relpath)
    if match:
        return ArchiveInfo(relpath, "test", _stream(match["kind"], match["size"]))
    match = _WEATHER.match(relpath)
    if match:
        return ArchiveInfo(relpath, match["weather"], _stream(match["kind"], match["size"]))
    return None


def classify_calibration(relpath: str) -> Optional[tuple[str, str]]:
    """``(split, sequence)`` of a half-size calibration file, or ``None``."""
    match = _CALIBRATION.match(relpath)
    return (match["split"], match["sequence"]) if match else None


def calibration_path(split: str, sequence: str) -> str:
    return f"calibration/{split}/half-image-calib/{sequence}.txt"


def subset_of(relpath: str) -> Optional[str]:
    """The subset an indexed media path belongs to."""
    parts = relpath.split("/")
    if parts[0] == "weather" and len(parts) > 1 and parts[1] in WEATHERS:
        return parts[1]
    if parts[0] in SPLITS:
        return parts[0]
    return None
