"""Run the KITTI-CARLA recipe on the real download (``-m real``).

Set ``KITTI_CARLA_ROOT`` to the directory holding ``Town01.zip`` (and/or an
extracted ``Town01/``). Every 1000th frame of Town01 is prepared into a
temporary directory, then loaded through euler-loading, and the lidar is
projected into the depth image.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.datasets import kitti_carla  # noqa: F401  (registers the recipe)

pytestmark = pytest.mark.real

ROOT = os.environ.get("KITTI_CARLA_ROOT")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not ROOT or not any((Path(ROOT) / name).exists() for name in ("Town01.zip", "Town01")):
        pytest.skip("KITTI_CARLA_ROOT does not hold Town01")
    out = tmp_path_factory.mktemp("kitti_carla")
    receipt = prepare("kitti_carla", ROOT, out, towns=["Town01"], frame_step=1000).receipt
    return out, receipt


def test_town_layout(prepared):
    out, receipt = prepared
    counts = receipt["summary"]["towns"]["Town01"]
    assert counts["rgb"] == counts["depth"] == counts["semantic_segmentation"] == 10
    assert counts["lidar"] == 5
    report = verify(out)
    assert report.ok, report.format_text()


def test_lidar_projects_onto_depth(prepared):
    pytest.importorskip("euler_loading")
    from euler_loading import Modality, MultiModalDataset
    from euler_loading.loaders.cpu import kitti_carla as kc

    out, _ = prepared
    root = str(out)
    dataset = MultiModalDataset(
        modalities={
            "depth": Modality(root, metadata_scope="depth", loader=kc.depth, writer=kc.write_depth),
            "lidar": Modality(
                root, metadata_scope="lidar", loader=kc.lidar_point_cloud,
                writer=kc.write_lidar_point_cloud,
            ),
        },
        hierarchical_modalities={
            "intrinsics": Modality(
                root, metadata_scope="intrinsics", loader=kc.read_intrinsics,
                writer=kc.write_intrinsics, collapse_single=True,
            ),
            "extrinsics": Modality(
                root, metadata_scope="camera_extrinsics", loader=kc.read_extrinsics,
                writer=kc.write_extrinsics, collapse_single=True,
            ),
        },
    )
    assert len(dataset) == 10
    sample = dataset[0]
    K = sample["intrinsics"].astype(float)
    T = sample["extrinsics"].astype(float)
    points = sample["lidar"][:, :3]
    camera = (T @ np.c_[points, np.ones(len(points))].T)[:3]
    camera = camera[:, camera[2] > 0.5]
    uvw = K @ camera
    u, v, z = uvw[0] / uvw[2], uvw[1] / uvw[2], uvw[2]
    height, width = sample["depth"].shape
    inside = (u >= 0) & (u < width - 0.5) & (v >= 0) & (v < height - 0.5)
    depth = sample["depth"][np.rint(v[inside]).astype(int), np.rint(u[inside]).astype(int)]
    valid = depth < 999  # the far plane is sky
    error = np.abs(z[inside][valid] - depth[valid])
    assert valid.sum() > 10_000
    assert np.median(error) < 0.05
