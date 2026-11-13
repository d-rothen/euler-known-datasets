"""Run the UrbanSyn recipe on the real download (``-m real``).

Set ``URBANSYN_ROOT`` to a directory holding ``rgb.zip``, ``depth.zip``,
``ss.zip`` and ``camera_metadata.json``, or their extracted folders (an output
prepared from them works too, and is linked instantly). The veteranenstrasse
scene (1946 frames) is prepared; from the zips that extracts about 22 GB.
Frames 5920 and 5977 of it see the same street, which checks the depth
convention, the intrinsics and the poses together.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.datasets import urbansyn  # noqa: F401  (registers the recipe)
from euler_known_datasets.datasets.urbansyn import SCENES
from euler_known_datasets.datasets.urbansyn.sources import discover

pytestmark = pytest.mark.real

ROOT = os.environ.get("URBANSYN_ROOT")
SCENE = "veteranenstrasse"


def _root():
    if not ROOT or not Path(ROOT).is_dir():
        pytest.skip("URBANSYN_ROOT is not set")
    return Path(ROOT)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = _root()
    out = tmp_path_factory.mktemp("urbansyn")
    result = prepare("urbansyn", root, out, scenes=[SCENE])
    return out, result.receipt


def test_release_layout():
    found = discover(_root())
    assert set(found.folders) == {"rgb", "depth", "ss"}, found.notes
    for folder in ("rgb", "depth", "ss"):
        assert len(found.folders[folder].frames) == 7539, folder
    assert found.camera_file is not None
    last = max(int(f) for f in found.folders["rgb"].frames)
    assert last == SCENES[-1][3] == 7539


def test_counts(prepared):
    _, receipt = prepared
    # Depth leaves out the 57 maps that lack the static scene (frames 6973-7029).
    assert receipt["summary"]["files"] == {"rgb": 1946, "depth": 1889, "semantic_segmentation": 1946,
                                           "intrinsics": 1, "camera_pose": 1946}
    assert receipt["summary"]["frames_per_scene"] == {SCENE: 1946}
    assert receipt["summary"]["incomplete_depth"] == {"frames": 57, "indexed": False, "left_out": "6973-7029"}


def test_output_verifies_and_loads(prepared):
    out, _ = prepared
    report = verify(out)
    assert report.ok, report.format_text()

    cpu = pytest.importorskip("euler_loading.loaders.cpu.urbansyn")
    pytest.importorskip("OpenEXR")
    from euler_loading import Modality, MultiModalDataset

    def modality(scope, loader, **kwargs):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=lambda *a, **k: None, **kwargs)

    dataset = MultiModalDataset(
        modalities={
            "rgb": modality("rgb", cpu.rgb),
            "depth": modality("depth", cpu.depth),
            "labels": modality("semantic_segmentation", cpu.semantic_segmentation),
            "pose": modality("camera_pose", cpu.camera_pose),
        },
        hierarchical_modalities={"intrinsics": modality("intrinsics", cpu.read_intrinsics, collapse_single=True)},
    )
    assert len(dataset) == 1889
    sample = dataset[0]
    assert sample["full_id"] == f"/scene:{SCENE}/5594"
    assert sample["rgb"].shape == (1024, 2048, 3)
    assert sample["depth"].shape == sample["labels"].shape == (1024, 2048)
    assert set(np.unique(sample["labels"]).tolist()) <= set(range(20))
    assert sample["intrinsics"][0, 0] == pytest.approx(1730.207, abs=1e-3)
    # Planar depth on the road below a camera about 1.1 m above it: a few meters to tens of meters.
    road = (sample["labels"] == 0) & (sample["depth"] > 0)
    assert 2 < np.median(sample["depth"][road]) < 60
    # The camera rides 0.8-1.7 m above the scene origin's height, upright (y down in the camera, y up in the world).
    pose = sample["pose"]
    assert 0.7 < pose[1, 3] < 1.8 and pose[1, 1] < -0.95


def test_depth_reprojects_between_frames(prepared):
    """Frames 5920 and 5977 see the same street: back-projected with K and moved
    with the two poses, the first frame's static points land at the second's depth."""
    out, _ = prepared
    cpu = pytest.importorskip("euler_loading.loaders.cpu.urbansyn")
    pytest.importorskip("OpenEXR")
    K = cpu.read_intrinsics(str(out / "camera_metadata.json")).astype(np.float64)
    depth = {f: cpu.depth(str(out / f"depth/depth_{f}.exr")).astype(np.float64) for f in ("5920", "5977")}
    labels = {f: cpu.semantic_segmentation(str(out / f"ss/ss_{f}.png")) for f in ("5920", "5977")}
    poses = {f: cpu.camera_pose(str(out / f"depth/depth_{f}.exr")) for f in ("5920", "5977")}
    v, u = np.mgrid[1:1024:4, 1:2048:4]
    d = depth["5920"][v, u]
    keep = (d > 0) & (d < 60) & np.isin(labels["5920"][v, u], [0, 1, 2, 3, 4, 8, 9])
    points = (np.stack([u[keep], v[keep], np.ones(keep.sum())], 1) @ np.linalg.inv(K).T) * d[keep][:, None]
    to_b = np.linalg.inv(poses["5977"]) @ poses["5920"]
    inb = points @ to_b[:3, :3].T + to_b[:3, 3]
    z = inb[:, 2]
    proj = inb @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        ub, vb = proj[:, 0] / z, proj[:, 1] / z
    seen = (z > 0.5) & (ub >= 0) & (ub <= 2047) & (vb >= 0) & (vb <= 1023)
    ui, vi = np.rint(ub[seen]).astype(int), np.rint(vb[seen]).astype(int)
    agree = labels["5977"][vi, ui] == labels["5920"][v, u][keep][seen]
    measured = depth["5977"][vi, ui][agree]
    rel = np.abs(z[seen][agree] - measured) / measured
    assert agree.sum() > 10000
    assert np.median(rel) < 0.003 and np.mean(rel < 0.01) > 0.8
