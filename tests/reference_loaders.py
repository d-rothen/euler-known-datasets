"""Minimal readers of the nuScenes layout v1, standing in for euler-loading's future module."""
from __future__ import annotations

import json

import numpy as np
from euler_loading.loaders._annotations import SampleContext, selects_per_sample
from PIL import Image


def rgb(path, meta=None, *, attributes=None):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def lidar_point_cloud(path, meta=None, *, attributes=None):
    data = path.read() if hasattr(path, "read") else open(path, "rb").read()
    return np.frombuffer(data, dtype=np.float32).reshape(-1, 5).copy()


def _doc(path):
    return json.load(path) if hasattr(path, "read") else json.load(open(path))


def read_intrinsics(path, meta=None, *, attributes=None):
    return np.asarray(_doc(path)["camera"]["intrinsics"], dtype=np.float32)


def _select(doc, context: SampleContext):
    lidar = doc["frames"][context.id]["lidar"]
    if lidar is None:
        raise KeyError(f"{context.full_id} has no synchronised sweep")
    return np.asarray(lidar["lidar_to_camera"], dtype=np.float32)


@selects_per_sample(_select)
def read_extrinsics(path, meta=None, *, attributes=None):
    return _doc(path)
