"""Run the SHIFT recipe on the real data (``-m real``).

Set ``SHIFT_ROOT`` to a SHIFT download root holding ``discrete/`` with the
train frames decoded by ``slurm/tools/shift_decode.py`` (dry weather, day and
twilight, frame_step 40, 3 m / 5 degree motion filter) and
``discrete/videos/train/front/{det_2d.json,seq.csv}``. The output links the
source (``--media symlink``), so nothing large is written.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.datasets import shift  # noqa: F401  (registers the recipe)
from euler_known_datasets.datasets.shift.sources import discover

pytestmark = pytest.mark.real

ROOT = os.environ.get("SHIFT_ROOT")
OPTIONS = {"splits": ["train"], "weather": ["clear", "cloudy", "overcast"], "time_of_day": ["daytime", "dawn_dusk"],
           "frame_step": 40, "min_motion_m": 3.0, "min_rotation_deg": 5.0}


def _root():
    if not ROOT or not Path(ROOT).is_dir():
        pytest.skip("SHIFT_ROOT is not set")
    return Path(ROOT)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp("shift")
    result = prepare("shift", _root(), out, media="symlink", **OPTIONS)
    return out, result.receipt


def test_source():
    train = discover(_root()).splits["train"]
    assert train.decoded is not None and train.decoded["selection"]["frame_step"] == 40
    assert train.groups["img"].ext == "png" and train.groups["img"].extracted
    assert len(train.groups["img"].files) == len(train.groups["semseg"].files) == 11647


def test_counts(prepared):
    _, receipt = prepared
    train = receipt["summary"]["splits"]["train"]
    # 3000 sequences -> 1243 dry, day or twilight -> 16,159 frames every 4 s -> 11,647 that moved.
    assert {k: train[k] for k in ("sequences", "sequences_selected", "candidate_frames", "frames_selected")} == {
        "sequences": 3000, "sequences_selected": 1243, "candidate_frames": 16159, "frames_selected": 11647}
    missing = train.get("frames_without_file", {}).get("depth", {}).get("count", 0)
    assert train["frames"] == 11647 - missing and missing <= 4
    assert receipt["summary"]["files"]["intrinsics"] == receipt["summary"]["files"]["camera_pose"] == 1243


def test_output_verifies_and_loads(prepared):
    out, _ = prepared
    report = verify(out)
    assert report.ok, report.format_text()
    cpu = pytest.importorskip("euler_loading.loaders.cpu.shift")
    from euler_loading import Modality, MultiModalDataset

    def modality(scope, loader):
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=lambda *a, **k: None)

    dataset = MultiModalDataset(
        modalities={"rgb": modality("rgb", cpu.rgb), "depth": modality("depth", cpu.depth),
                    "sky": modality("semantic_segmentation", cpu.sky_mask)},
        hierarchical_modalities={"intrinsics": modality("intrinsics", cpu.read_intrinsics),
                                 "pose": modality("camera_pose", cpu.camera_pose)},
    )
    sample = dataset[0]
    assert sample["rgb"].shape == (800, 1280, 3) and sample["depth"].shape == (800, 1280)
    assert sample["full_id"].startswith("/split:train/weather:")
    # The sky lies at the depth's far plane (with a few distant non-sky pixels); the hood is about a metre away.
    far = sample["depth"] > 999.9
    assert not (sample["sky"] & ~far).any() and (far & ~sample["sky"]).mean() < 0.01
    assert 0.9 < sample["depth"][799, 640] < 1.3
    K = next(iter(sample["intrinsics"].values()))
    assert K.tolist() == [[640, 0, 640], [0, 640, 400], [0, 0, 1]]


def test_depth_reprojects_between_kept_frames(prepared):
    """Two kept frames of a sequence: frame a's depth, back-projected with K and moved
    with both poses, lands on frame b's depth (static scene, hood rows excluded)."""
    out, _ = prepared
    cpu = pytest.importorskip("euler_loading.loaders.cpu.shift")
    calibrations = sorted((out / "calibration/train").iterdir())
    checked = 0
    for path in calibrations:
        table = cpu.camera_pose(str(path))
        frames = sorted(table.frames)
        for a, b in zip(frames, frames[1:]):
            pa, pb = table.camera_to_world(a), table.camera_to_world(b)
            if not 3 < np.linalg.norm(pa[:3, 3] - pb[:3, 3]) < 12:
                continue
            folder = out / "discrete/images/train/front/depth" / table.sequence
            da = cpu.depth(str(folder / f"{a}_depth_front.png")).astype(np.float64)
            db = cpu.depth(str(folder / f"{b}_depth_front.png")).astype(np.float64)
            K = table.intrinsics
            v, u = np.mgrid[2:720:4, 2:1280:4]
            d = da[v, u]
            keep = d < 80
            points = (np.stack([u[keep], v[keep], np.ones(keep.sum())], 1) @ np.linalg.inv(K).T) * d[keep][:, None]
            to_b = np.linalg.inv(pb) @ pa
            q = points @ to_b[:3, :3].T + to_b[:3, 3]
            p = q @ K.T
            ub, vb = p[:, 0] / q[:, 2], p[:, 1] / q[:, 2]
            seen = (q[:, 2] > 0.5) & (ub >= 0) & (ub <= 1279) & (vb >= 0) & (vb <= 719)
            measured = db[np.rint(vb[seen]).astype(int), np.rint(ub[seen]).astype(int)]
            rel = np.abs(q[seen, 2] - measured) / measured
            assert seen.sum() > 5000 and np.median(rel) < 0.01, (path.name, a, b, float(np.median(rel)))
            checked += 1
            break
        if checked == 5:
            break
    assert checked == 5
