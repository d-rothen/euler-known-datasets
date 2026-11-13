"""Run the Hypersim recipe on the real download (``-m real``).

Set ``HYPERSIM_ROOT`` to a directory holding the extracted scene folders (all,
or the files a partial download fetched) and ml-hypersim's split and camera
CSVs. Three scenes are prepared: an ordinary camera (ai_001_001), a tilt-shift
one (ai_037_002) and the most tilted one (ai_026_006, 41.5 degrees, with
a uniformly scaled frame-0 orientation).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.datasets import hypersim  # noqa: F401  (registers the recipe)

pytestmark = pytest.mark.real
pytest.importorskip("h5py")

ROOT = os.environ.get("HYPERSIM_ROOT")
SCENES = ["ai_001_001", "ai_037_002", "ai_026_006"]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not ROOT or not Path(ROOT).is_dir():
        pytest.skip("HYPERSIM_ROOT is not set")
    out = tmp_path_factory.mktemp("hypersim")
    # Symlinks: node-local scratch is another filesystem than the download.
    result = prepare("hypersim", ROOT, out, scenes=SCENES, media="symlink")
    return out, result.receipt


def test_summary(prepared):
    _, receipt = prepared
    summary = receipt["summary"]
    assert summary["scenes"] == 3
    assert summary["cameras"]["max_tilt_deg"] == pytest.approx(41.5, abs=0.1)
    assert summary["cameras"]["orientation_notes"]["ai_026_006/cam_00"] == {"uniformly_scaled_frames": [0]}
    assert summary["tonemap"]["frames_without_entity_ids"] == 0
    assert summary["depth"]["frames_beyond_range"] == 0


def test_output_verifies_and_loads(prepared):
    out, _ = prepared
    report = verify(out)
    assert report.ok, report.format_text()

    cpu = pytest.importorskip("euler_loading.loaders.cpu.hypersim")
    from euler_loading import Modality, MultiModalDataset

    def modality(scope, loader):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=lambda *a, **k: None)

    dataset = MultiModalDataset(
        modalities={"rgb": modality("rgb", cpu.rgb), "depth": modality("depth", cpu.depth)},
        hierarchical_modalities={"intrinsics": modality("intrinsics", cpu.read_intrinsics),
                                 "pose": modality("camera_pose", cpu.camera_pose)},
    )
    sample = dataset[0]
    assert sample["rgb"].shape == (768, 1024, 3)
    assert sample["depth"].shape == (768, 1024) and (sample["depth"] > 0).mean() > 0.9
    K = next(iter(sample["intrinsics"].values()))
    assert K.shape == (3, 3)
    pose = next(iter(sample["pose"].values()))
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-5)
