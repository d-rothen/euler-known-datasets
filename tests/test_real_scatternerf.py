"""Run the ScatterNeRF recipe on the real downloads (``-m real``).

Set ``SCATTERNERF_ROOT`` to a directory holding the published
``sequence_<n>.zip`` archives (or their extracted ``<n>/`` folders). All of
it is prepared: 345 frames, about 400 MB.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.datasets import scatternerf  # noqa: F401  (registers the recipe)
from euler_known_datasets.datasets.scatternerf import BASELINE_M

pytestmark = pytest.mark.real

ROOT = os.environ.get("SCATTERNERF_ROOT")

#: Frames per sequence and camera as released (checked 2026-11-10).
FRAMES = {"0": {"left": 41, "right": 50}, "6": {"left": 69, "right": 65}, "13": {"left": 61, "right": 59}}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not ROOT or not Path(ROOT).is_dir():
        pytest.skip("SCATTERNERF_ROOT is not set")
    out = tmp_path_factory.mktemp("scatternerf")
    result = prepare("scatternerf", ROOT, out)
    return out, result.receipt


def test_sequences_and_scale(prepared):
    _, receipt = prepared
    sequences = receipt["summary"]["sequences"]
    assert {s: stats["frames"] for s, stats in sequences.items()} == FRAMES
    for stats in sequences.values():
        assert stats["incomplete_frames"] == 0
        assert stats["baseline_along_x"] > 0.99
        assert stats["baseline_spread"] < 0.05
        assert 5 < stats["scale_to_meters"] < 10
    assert sequences["0"]["recorded_utc"].startswith("2018-10-08")


def test_output_verifies_and_loads(prepared):
    out, _ = prepared
    report = verify(out)
    assert report.ok, report.format_text()

    cpu = pytest.importorskip("euler_loading.loaders.cpu.scatternerf")
    from euler_loading import Modality, MultiModalDataset

    def modality(scope, loader):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=lambda *a, **k: None)

    poses = MultiModalDataset(modalities={"pose": modality("camera_pose", cpu.camera_pose_metric)})
    assert len(poses) == sum(sum(cameras.values()) for cameras in FRAMES.values())
    by_id = {poses[i]["full_id"]: poses[i]["pose"] for i in range(len(poses))}
    baselines = [
        np.linalg.norm((np.linalg.inv(pose) @ by_id[full_id.replace("camera:left", "camera:right")])[:3, 3])
        for full_id, pose in by_id.items()
        if "camera:left" in full_id and full_id.replace("camera:left", "camera:right") in by_id
    ]
    assert len(baselines) == 29 + 37 + 30
    assert np.median(baselines) == pytest.approx(BASELINE_M, rel=1e-2)

    images = MultiModalDataset(
        modalities={
            "rgb": modality("rgb", cpu.rgb),
            "intrinsics": modality("intrinsics", cpu.read_intrinsics),
        }
    )
    sample = images[0]
    assert sample["rgb"].shape == (908, 1704, 3)
    np.testing.assert_allclose(sample["intrinsics"], [[2355.7228, 0, 852], [0, 2355.7228, 454], [0, 0, 1]], rtol=1e-6)
