"""SHIFT's semantic classes, and the labels its segmentation videos keep.

SHIFT labels pixels with the 23 semantic tags of CARLA 0.9.10-0.9.13
(:data:`CLASS_NAMES`, sky ``13``). The official image-rate ``semseg.zip`` stores
them as PNGs with the tag in the red channel and green = blue = 0 (shift-dev's
loader reads channel 0).

The video-rate ``semseg.tar`` (the only segmentation the surviving mirror has)
holds one H.264 video per sequence, lossless (``qp=0``, yuv444p) but encoded
from the colour ``(tag, 0, 0)`` after a BT.601 limited-range RGB to YUV
conversion. Nine pairs of tags map to the same YUV triple and cannot be told
apart; five tags, the sky among them, keep a triple of their own. Every pixel
of the decoded videos checked (six train sequences, 2026-11-13) holds one of
the 14 triples in :data:`VIDEO_TRIPLES`. Decoded frames are stored with
:data:`MERGED_LABELS`: the five exact tags keep their CARLA ID, each pair gets
an ID from 23 to 31.
"""

from __future__ import annotations

import numpy as np

#: CARLA 0.9.10-0.9.13 semantic tags, indexed by tag ID.
CLASS_NAMES = (
    "unlabeled", "building", "fence", "other", "pedestrian", "pole",
    "road_line", "road", "sidewalk", "vegetation", "vehicle", "wall",
    "traffic_sign", "sky", "ground", "bridge", "rail_track", "guard_rail",
    "traffic_light", "static", "dynamic", "water", "terrain",
)
SKY = 13
#: ``skyclass`` of a label PNG: the tag in red, green = blue = 0.
SKYCLASS = [SKY, 0, 0]

#: The pairs the segmentation videos cannot separate, by the ID decoded frames give them.
MERGED_PAIRS = {
    23: (0, 1), 24: (2, 3), 25: (4, 5), 26: (6, 7), 27: (8, 9),
    28: (11, 12), 29: (15, 16), 30: (18, 19), 31: (20, 21),
}
#: Tags whose video triple is their own.
EXACT_TAGS = (10, 13, 14, 17, 22)
#: Label ID -> name in decoded frames.
MERGED_LABELS = {
    **{tag: CLASS_NAMES[tag] for tag in EXACT_TAGS},
    **{label: f"{CLASS_NAMES[a]}|{CLASS_NAMES[b]}" for label, (a, b) in MERGED_PAIRS.items()},
}
MERGED_LABELS = dict(sorted(MERGED_LABELS.items()))


def video_triple(tag: int) -> tuple[int, int, int]:
    """The ``(Y, U, V)`` the videos hold for *tag*: BT.601 limited range of ``(tag, 0, 0)``."""
    return (round(16 + 65.481 * tag / 255), round(128 - 37.797 * tag / 255), round(128 + 112 * tag / 255))


def _merged_label(tag: int) -> int:
    for label, pair in MERGED_PAIRS.items():
        if tag in pair:
            return label
    return tag


#: Video ``(Y, U, V)`` -> label ID of a decoded frame.
VIDEO_TRIPLES = {video_triple(tag): _merged_label(tag) for tag in range(len(CLASS_NAMES))}
assert len(VIDEO_TRIPLES) == len(MERGED_LABELS) == 14

_KEYS = np.array(sorted((y << 16) | (u << 8) | v for y, u, v in VIDEO_TRIPLES), dtype=np.int64)
_VALUES = np.array([VIDEO_TRIPLES[(k >> 16, (k >> 8) & 255, k & 255)] for k in _KEYS.tolist()], dtype=np.uint8)

#: Label of a pixel whose triple is not in :data:`VIDEO_TRIPLES`.
UNKNOWN = 255


def labels_from_yuv(yuv: np.ndarray) -> tuple[np.ndarray, int]:
    """Labels of a decoded ``(3, H, W)`` yuv444p frame, and the number of unknown pixels.

    Unknown triples (none in the release) get :data:`UNKNOWN`.
    """
    yuv = np.asarray(yuv)
    if yuv.ndim != 3 or yuv.shape[0] != 3:
        raise ValueError(f"expected planar (3, H, W) YUV, got shape {yuv.shape}")
    packed = (yuv[0].astype(np.int64) << 16) | (yuv[1].astype(np.int64) << 8) | yuv[2].astype(np.int64)
    position = np.minimum(np.searchsorted(_KEYS, packed), len(_KEYS) - 1)
    known = _KEYS[position] == packed
    labels = np.where(known, _VALUES[position], np.uint8(UNKNOWN)).astype(np.uint8)
    return labels, int((~known).sum())


def label_set_meta(decoded_from_video: bool) -> dict:
    """The ``semantic_segmentation`` head meta describing a label set."""
    if decoded_from_video:
        return {
            "label_ids": "shift_video_merged",
            "classes": {str(k): v for k, v in MERGED_LABELS.items()},
            "merged_pairs": {str(k): [CLASS_NAMES[a], CLASS_NAMES[b]] for k, (a, b) in MERGED_PAIRS.items()},
        }
    return {"label_ids": "carla_0.9.10", "classes": list(CLASS_NAMES)}
