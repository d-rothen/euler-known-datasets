"""Run the DrivingStereo recipe on the real download (``-m real``).

Set ``DRIVINGSTEREO_ROOT`` to a directory holding at least
``weather/foggy/*.zip`` (the layout of the official downloads). The foggy
subset is extracted into a temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.datasets import driving_stereo  # noqa: F401  (registers)

pytestmark = pytest.mark.real

ROOT = os.environ.get("DRIVINGSTEREO_ROOT")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not ROOT or not (Path(ROOT) / "weather" / "foggy").is_dir():
        pytest.skip("DRIVINGSTEREO_ROOT does not hold weather/foggy")
    out = tmp_path_factory.mktemp("driving_stereo")
    result = prepare("driving_stereo", ROOT, out, subsets=["foggy"])
    return out, result.receipt


def test_foggy_subset(prepared):
    out, receipt = prepared
    files = receipt["summary"]["subsets"]["foggy"]["files"]
    assert files["rgb"] == files["depth"] == files["disparity"] == 500
    assert receipt["splits"]["rgb"] == ["foggy"]
    report = verify(out)
    assert report.ok, report.format_text()


def test_loads_on_the_half_size_left_grid(prepared):
    pytest.importorskip("euler_loading")
    from euler_loading import Modality, MultiModalDataset
    from euler_loading.loaders.cpu import driving_stereo as ds

    out, _ = prepared

    def modality(scope, function):
        return Modality(str(out), metadata_scope=scope, split="foggy",
                        loader=getattr(ds, function), writer=getattr(ds, "write_" + function.replace("read_", "", 1)))

    dataset = MultiModalDataset(modalities={
        "rgb": modality("rgb", "rgb"),
        "depth": modality("depth", "depth"),
        "disparity": modality("disparity", "disparity"),
        "depth_full": modality("depth_full_size", "depth_full_size"),
    })
    assert len(dataset) == 500
    sample = dataset[0]
    assert sample["rgb"].shape == (400, 879, 3)
    assert sample["depth"].shape == sample["disparity"].shape == sample["depth_full"].shape == (400, 879)
    # Planar depth: depth * disparity = fx * baseline (1001.496 * 0.544915 m).
    valid = (sample["depth"] > 0) & (sample["disparity"] > 0)
    product = np.median(sample["depth"][valid] * sample["disparity"][valid])
    assert abs(product - 1001.496 * 0.544915) / product < 1e-3
