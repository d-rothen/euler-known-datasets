"""Run the nuScenes recipe on the real download (``-m real``).

Set ``NUSCENES_ROOT`` to a directory holding v1.0-test extracted (and/or its
archives). The output is linked, not copied, into a temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from euler_known_datasets import prepare, verify

pytestmark = pytest.mark.real

ROOT = os.environ.get("NUSCENES_ROOT")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not ROOT or not (Path(ROOT) / "v1.0-test").is_dir():
        pytest.skip("NUSCENES_ROOT does not hold an extracted v1.0-test")
    out = tmp_path_factory.mktemp("nuscenes")
    result = prepare("nuscenes", ROOT, out, versions=["v1.0-test"], cameras=["CAM_FRONT_LEFT"])
    return out, result.receipt


def test_every_keyframe_pairs_with_its_samples_sweep(prepared):
    _, receipt = prepared
    counts = receipt["summary"]["versions"]["v1.0-test"]["cameras"]["CAM_FRONT_LEFT"]
    assert counts["keyframes"] == 6008
    assert counts["keyframes_paired_as_sample"] == counts["keyframes"]
    assert counts["frames"] == 34902
    assert counts["unsynced"] == 121  # frames whose sweep was dropped
    assert 42.5 < receipt["summary"]["nominal_lidar_offset_ms"]["CAM_FRONT_LEFT"] < 43.5


def test_output_verifies(prepared):
    out, receipt = prepared
    assert receipt["splits"]["rgb"] == ["test", "test_keyframes"]
    report = verify(out)
    assert report.ok, report.format_text()
