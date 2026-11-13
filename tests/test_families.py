"""Every scope of a prepared dataset carries the dataset family as its identity."""

from __future__ import annotations

import json

import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.artifacts import family_dataset

from .fake_acdc import build_fake_acdc
from .fake_cerema import build_fake_cerema
from .fake_driving_stereo import build_fake_driving_stereo
from .fake_hypersim import build_fake_hypersim
from .fake_kitti_carla import build_fake_kitti_carla
from .fake_muses import build_fake_muses
from .fake_nuscenes import build_fake_nuscenes
from .fake_scatternerf import build_fake_scatternerf
from .fake_shift import build_fake_shift
from .fake_urbansyn import build_fake_urbansyn

BUILDERS = {
    "nuscenes": build_fake_nuscenes,
    "driving_stereo": build_fake_driving_stereo,
    "kitti_carla": build_fake_kitti_carla,
    "muses": build_fake_muses,
    "scatternerf": build_fake_scatternerf,
    "cerema": build_fake_cerema,
    "acdc": build_fake_acdc,
    "hypersim": build_fake_hypersim,
    "urbansyn": build_fake_urbansyn,
    "shift": build_fake_shift,
}


def _heads(out):
    return {
        path.parent.name: json.loads(path.read_text())
        for path in sorted((out / ".ds_crawler").glob("*/dataset-head.json"))
    }


@pytest.mark.parametrize("recipe", sorted(BUILDERS))
def test_every_scope_carries_the_family(recipe, tmp_path):
    fake = BUILDERS[recipe](tmp_path / "downloads")
    out = tmp_path / "prepared"
    prepare(recipe, fake.source, out)
    heads = _heads(out)
    assert len(heads) > 1
    for scope, head in heads.items():
        assert head["dataset"]["id"] == recipe, scope
        assert head["dataset"]["name"] == recipe, scope
        attributes = head["dataset"]["attributes"]
        # Ground truth, or a published baseline's output shipped with the dataset.
        assert (attributes["gt"], attributes["baseline"]) in ((True, False), (False, True)), scope
    assert any(head["dataset"]["attributes"]["gt"] for head in heads.values())
    assert not [w for w in verify(out).warnings if "family" in w]


def test_family_dataset():
    assert family_dataset("nuscenes", {"cameras": ["CAM_FRONT_LEFT"]}) == {
        "id": "nuscenes",
        "name": "nuscenes",
        "attributes": {"gt": True, "baseline": False, "cameras": ["CAM_FRONT_LEFT"]},
    }


def test_verify_warns_about_scopes_without_the_family(tmp_path):
    fake = build_fake_nuscenes(tmp_path / "downloads")
    out = tmp_path / "prepared"
    prepare("nuscenes", fake.source, out, cameras=["CAM_FRONT_LEFT"])
    head_path = out / ".ds_crawler" / "lidar" / "dataset-head.json"
    head = json.loads(head_path.read_text())
    head["dataset"]["id"] = "nuscenes_lidar"
    head_path.write_text(json.dumps(head))
    assert any("family" in w and "nuscenes_lidar" in w for w in verify(out).warnings)
