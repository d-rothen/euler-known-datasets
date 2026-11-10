"""Run the MUSES recipe on the real downloads (``-m real``).

Set ``MUSES_ROOT`` to a directory holding the MUSES zip packages (or their
extracted ``muses/`` tree). Only the val split is prepared, to keep it quick.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.datasets import muses  # noqa: F401  (registers the recipe)

pytestmark = pytest.mark.real

ROOT = os.environ.get("MUSES_ROOT")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not ROOT or not Path(ROOT).is_dir():
        pytest.skip("MUSES_ROOT is not set")
    out = tmp_path_factory.mktemp("muses")
    result = prepare("muses", ROOT, out, splits=["val"])
    return out, result.receipt


def test_val_split_counts(prepared):
    _, receipt = prepared
    files = receipt["summary"]["files"]
    assert files["rgb"] == files["lidar"] == files["gnss"] == 250
    # Reference frames exist only for adverse conditions: not the 50 clear-day frames.
    if "reference_rgb" in files:
        assert files["reference_rgb"] == 200
    for scope in ("semantic_segmentation", "semantic_segmentation_train_ids"):
        if scope in files:
            assert files[scope] == 250


def test_output_verifies_and_loads(prepared):
    out, _ = prepared
    report = verify(out)
    assert report.ok, report.format_text()

    euler_loading = pytest.importorskip("euler_loading")
    from euler_loading.loaders.cpu import muses as cpu

    def modality(scope, loader, **kwargs):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return euler_loading.Modality(
            str(out), metadata_scope=scope, loader=loader, writer=lambda *a, **k: None, **kwargs
        )

    dataset = euler_loading.MultiModalDataset(
        modalities={"rgb": modality("rgb", cpu.rgb), "lidar": modality("lidar", cpu.lidar_point_cloud)},
        hierarchical_modalities={
            "intrinsics": modality("intrinsics", cpu.read_intrinsics, collapse_single=True),
            "extrinsics": modality("camera_extrinsics", cpu.read_extrinsics, collapse_single=True),
            "timing": modality("sensor_timing", cpu.sensor_timing, collapse_single=True),
        },
    )
    assert len(dataset) == 250
    sample = dataset[0]
    assert sample["rgb"].shape == (1080, 1920, 3)
    points = sample["lidar"]
    camera = (sample["extrinsics"] @ np.c_[points[:, :3], np.ones(len(points))].T)[:3]
    assert (camera[2] > 0).mean() > 0.99  # lidar2rgb maps into optical camera axes
    assert sample["timing"]["frame"] in sample["meta"]["rgb"]["path"]
