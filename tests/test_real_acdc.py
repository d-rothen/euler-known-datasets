"""Run the ACDC recipe on the real download (``-m real``).

Set ``ACDC_ROOT`` to a directory holding ``rgb_anon_trainvaltest.zip``, an
archive of a whole ACDC copy, or the extracted ``rgb_anon/`` tree. The fog
validation images and their references are prepared (one pass through the
archive).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from euler_known_datasets import prepare, verify
from euler_known_datasets.datasets import acdc  # noqa: F401  (registers the recipe)

pytestmark = pytest.mark.real

ROOT = os.environ.get("ACDC_ROOT")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not ROOT or not Path(ROOT).is_dir():
        pytest.skip("ACDC_ROOT is not set")
    out = tmp_path_factory.mktemp("acdc")
    result = prepare("acdc", ROOT, out, conditions=["fog"], splits=["val"])
    return out, result.receipt


def test_counts_and_pairs(prepared):
    _, receipt = prepared
    assert receipt["summary"]["images"] == {"rgb": {"fog/val": 100}, "reference_rgb": {"fog/val": 100}}
    assert receipt["summary"]["pairs"] == 100
    assert receipt["splits"] == {"rgb": ["val"], "reference_rgb": ["val"]}


def test_output_verifies_and_loads(prepared):
    out, _ = prepared
    report = verify(out)
    assert report.ok, report.format_text()

    cpu = pytest.importorskip("euler_loading.loaders.cpu.acdc")
    from euler_loading import Modality, MultiModalDataset

    def modality(scope, loader):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=lambda *a, **k: None)

    dataset = MultiModalDataset(modalities={"rgb": modality("rgb", cpu.rgb),
                                            "reference": modality("reference_rgb", cpu.reference_rgb)})
    assert len(dataset) == 100
    sample = dataset[0]
    assert sample["rgb"].shape == sample["reference"].shape == (1080, 1920, 3)
    assert sample["full_id"].startswith("/condition:fog/split:val/")
