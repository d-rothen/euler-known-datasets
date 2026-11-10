"""Run the CEREMA recipe on the real download (``-m real``).

Set ``CEREMA_ROOT`` to a directory holding ``pixel_accurate_depth_benchmark.zip``
(or its inner zips, or their extracted folders). Scene 1 in clear weather and
fog at 40 m is prepared (40 frames), to keep it quick.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.datasets import cerema  # noqa: F401  (registers the recipe)
from euler_known_datasets.datasets.cerema.sources import FOLDERS, discover

pytestmark = pytest.mark.real

ROOT = os.environ.get("CEREMA_ROOT")
SCOPES = ["rgb", "gated0", "lidar", "depth", "intrinsics", "sgm", "psmnet"]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not ROOT or not Path(ROOT).is_dir():
        pytest.skip("CEREMA_ROOT is not set")
    out = tmp_path_factory.mktemp("cerema")
    result = prepare("cerema", ROOT, out, modalities=SCOPES, scenes=["1"], weather=["clear", "fog40"])
    return out, result.receipt


def test_release_layout():
    if not ROOT or not Path(ROOT).is_dir():
        pytest.skip("CEREMA_ROOT is not set")
    found, notes = discover(Path(ROOT))
    assert set(found) == set(FOLDERS), notes
    for folder in ("rgb_left_8bit", "lidar_hdl64_rgb_left", "sgm", "gated31_10bit"):
        assert len(found[folder].names) == 1600, folder
    assert sorted(found["intermetric_rgb_left"].names) == [f"scene{s}.npz" for s in range(1, 5)]


def test_counts(prepared):
    _, receipt = prepared
    files = receipt["summary"]["files"]
    assert files == {**{s: 40 for s in SCOPES}, "intrinsics": 1}
    assert receipt["summary"]["conditions"]["weather"] == ["clear", "fog40"]


def test_output_verifies_and_loads(prepared):
    out, _ = prepared
    report = verify(out)
    assert report.ok, report.format_text()

    cpu = pytest.importorskip("euler_loading.loaders.cpu.cerema")
    from euler_loading import Modality, MultiModalDataset

    def modality(scope, loader, **kwargs):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=lambda *a, **k: None, **kwargs)

    dataset = MultiModalDataset(
        modalities={
            "rgb": modality("rgb", cpu.rgb),
            "lidar": modality("lidar", cpu.depth),
            "depth": modality("depth", cpu.depth),
            "sgm": modality("sgm", cpu.sgm_depth),
        },
        hierarchical_modalities={"intrinsics": modality("intrinsics", cpu.read_intrinsics, collapse_single=True)},
    )
    assert len(dataset) == 40
    sample = dataset[0]
    assert sample["rgb"].shape == (1024, 1920, 3)
    assert sample["depth"].shape == sample["lidar"].shape == sample["sgm"].shape == (1024, 1920)
    assert sample["intrinsics"][0, 0] == pytest.approx(2355.7228, rel=1e-6)
    # Lidar and ground truth are both z-depth on the left camera's grid.
    both = (sample["lidar"] > 0) & (sample["depth"] > 0)
    assert np.median(sample["lidar"][both] / sample["depth"][both]) == pytest.approx(1.0, abs=0.01)
    # SGM, converted with the benchmark's baseline and focal length, agrees to a few percent.
    crop = np.zeros(sample["depth"].shape, bool)
    crop[270:1004, 170:1900] = True
    valid = crop & (sample["sgm"] > 0) & (sample["depth"] > 0) & (sample["depth"] < 28)
    assert np.median(sample["sgm"][valid] / sample["depth"][valid]) == pytest.approx(1.0, abs=0.05)
