from __future__ import annotations

import json
import logging
import os
import shutil

import numpy as np
import pytest
from ds_crawler import index_dataset_from_path, load_dataset_split

from euler_known_datasets import MediaError, detect_recipes, prepare, verify
from euler_known_datasets.datasets import acdc  # noqa: F401  (registers the recipe)
from euler_known_datasets.datasets.acdc.sources import discover

from .fake_acdc import HEIGHT, WIDTH, build_fake_acdc, frames, path

N = len(frames())


@pytest.fixture
def tarred(tmp_path):
    return build_fake_acdc(tmp_path / "downloads", layout="tar")


@pytest.fixture
def extracted(tmp_path):
    return build_fake_acdc(tmp_path / "downloads", layout="extracted")


def _images_below(root):
    return {p.relative_to(root).as_posix() for p in (root / "rgb_anon").rglob("*.png")}


def test_discover(tarred, tmp_path):
    found = discover(tarred.source)
    assert found.archive.name == "ACDC.tar.gz" and found.archive_prefix == ""
    assert found.extracted is None
    zipped = build_fake_acdc(tmp_path / "zip", layout="zip")
    assert discover(zipped.source).archive.name == "rgb_anon_trainvaltest.zip"


@pytest.mark.parametrize("layout", ["tar", "zip"])
def test_prepare_from_an_archive(tmp_path, layout):
    fake = build_fake_acdc(tmp_path / "downloads", layout=layout)
    out = tmp_path / "prepared"
    receipt = prepare("acdc", fake.source, out).receipt
    assert receipt["status"] == "complete"
    assert receipt["media"] == {"requested": "auto", "used": "extract"}
    assert receipt["scopes"] == ["rgb", "reference_rgb"]
    assert receipt["summary"]["pairs"] == N
    assert receipt["summary"]["unpaired"] == {"rgb": 0, "reference_rgb": 0}
    assert receipt["summary"]["images"]["rgb"] == {
        "fog/test": 1, "fog/train": 2, "fog/val": 1, "night/train": 1, "rain/val": 1, "snow/test": 2}
    assert receipt["splits"] == {"rgb": ["test", "train", "val"], "reference_rgb": ["test", "train", "val"]}
    # Only the images and the documentation: no gt/ or lists_file_names/.
    assert _images_below(out) == fake.images(False) | fake.images(True)
    assert (out / "License.pdf").is_file() and (out / "README.md").is_file()
    assert not (out / "gt").exists() and not (out / "lists_file_names").exists()
    report = verify(out)
    assert report.ok, report.format_text()


def test_ids_join_adverse_and_reference(tarred, tmp_path):
    out = tmp_path / "prepared"
    prepare("acdc", tarred.source, out)
    rgb = index_dataset_from_path(out, metadata_scope="rgb")["index"]
    ref = index_dataset_from_path(out, metadata_scope="reference_rgb")["index"]
    node = rgb["children"]["condition:fog"]["children"]["split:train"]["children"]["sequence:GOPR0475"]
    assert [f["id"] for f in node["files"]] == ["000041", "000049"]
    assert node["files"][0]["path"] == path("fog", "train", "GOPR0475", "000041")
    ref_node = ref["children"]["condition:fog"]["children"]["split:train"]["children"]["sequence:GOPR0475"]
    assert [f["id"] for f in ref_node["files"]] == ["000041", "000049"]
    assert ref_node["files"][0]["path"] == path("fog", "train", "GOPR0475", "000041", reference=True)

    head = json.loads((out / ".ds_crawler" / "reference_rgb" / "dataset-head.json").read_text())
    assert head["dataset"]["id"] == "acdc" and head["dataset"]["attributes"]["gt"] is True
    assert head["modality"]["key"] == "rgb"
    assert {"range": [0, 255], "anonymized": True, "condition": "normal",
            "pixel_aligned": False}.items() <= head["modality"]["meta"].items()
    assert head["addons"]["euler_loading"]["function"] == "reference_rgb"

    val = load_dataset_split(out, "val", metadata_scope="rgb")
    assert sorted(f["path"] for f in _files(val)) == sorted(
        path(c, s, seq, f) for c, s, seq, f in frames(split="val"))


def _files(output):
    from ds_crawler import get_files
    return [{"path": p} for p in get_files(output)]


def test_options_select_conditions_splits_and_scopes(tarred, tmp_path):
    out = tmp_path / "fog"
    receipt = prepare("acdc", tarred.source, out, conditions=["fog"], splits=["train", "val"],
                      modalities=["rgb"]).receipt
    assert receipt["scopes"] == ["rgb"]
    assert receipt["summary"]["images"] == {"rgb": {"fog/train": 2, "fog/val": 1}}
    assert "pairs" not in receipt["summary"]
    assert _images_below(out) == {path(c, s, seq, f) for c, s, seq, f in frames("fog") if s != "test"}


def test_symlink_from_the_extracted_tree(extracted, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("acdc", extracted.source, out).receipt
    assert receipt["media"]["used"] == "symlink"
    assert (out / "rgb_anon").is_symlink() and (out / "License.pdf").is_symlink()
    assert verify(out).ok


@pytest.mark.parametrize("mode", ["hardlink", "copy"])
def test_hardlink_and_copy_from_a_subfolder(tmp_path, mode):
    fake = build_fake_acdc(tmp_path / "downloads" / "ACDC", layout="extracted")
    source = tmp_path / "downloads"
    out = tmp_path / "prepared"
    receipt = prepare("acdc", source, out, media=mode, conditions=["snow"]).receipt
    assert receipt["media"]["used"] == mode
    assert _images_below(out) == {p for p in fake.images(False) | fake.images(True) if "/snow/" in p}
    some = path("snow", "test", "GP010606", "000853")
    same = os.stat(out / some).st_ino == os.stat(fake.source / some).st_ino
    assert same == (mode == "hardlink")
    assert verify(out).ok


def test_in_place_with_the_tree_in_a_subfolder(tmp_path):
    fake = build_fake_acdc(tmp_path / "downloads" / "ACDC", layout="extracted")
    source = tmp_path / "downloads"
    prepare("acdc", source, source, modalities=["rgb"])
    index = index_dataset_from_path(source, metadata_scope="rgb")["index"]
    node = index["children"]["condition:night"]["children"]["split:train"]["children"]["sequence:GP010376"]
    assert node["files"][0]["path"] == "ACDC/" + path("night", "train", "GP010376", "000153")
    assert verify(source).ok
    assert len(fake.images(False)) == N


def test_in_place_extracts_into_the_source(tarred):
    receipt = prepare("acdc", tarred.source, tarred.source).receipt
    assert receipt["media"]["used"] == "in_place"
    assert _images_below(tarred.source) == tarred.images(False) | tarred.images(True)
    assert verify(tarred.source).ok


def test_rerun_skips_the_archive(tarred, tmp_path, caplog):
    out = tmp_path / "prepared"
    prepare("acdc", tarred.source, out)
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("acdc", tarred.source, out)
    assert any("already extracted" in r.getMessage() for r in caplog.records)
    # Another selection extracts again (only what is missing is written).
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("acdc", tarred.source, out, conditions=["fog"])
    assert verify(out).ok


def test_extract_needs_an_archive(extracted, tmp_path):
    with pytest.raises(MediaError, match="archives"):
        prepare("acdc", extracted.source, tmp_path / "prepared", media="extract")


def test_no_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="no ACDC images"):
        prepare("acdc", tmp_path, tmp_path / "out")


def test_detect(tarred, tmp_path):
    assert "acdc" in [recipe.name for recipe in detect_recipes(tarred.source)]
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "other.zip").write_bytes(b"not a zip")
    assert "acdc" not in [recipe.name for recipe in detect_recipes(empty)]


def _unused_writer(path, value, meta=None):  # pragma: no cover - never called
    raise AssertionError("not writing")


def test_loads_in_euler_loading(tarred, tmp_path):
    cpu = pytest.importorskip("euler_loading.loaders.cpu.acdc")
    from euler_loading import Modality, MultiModalDataset

    out = tmp_path / "prepared"
    prepare("acdc", tarred.source, out)

    def modality(scope, loader, **kwargs):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=_unused_writer, **kwargs)

    dataset = MultiModalDataset(modalities={"rgb": modality("rgb", cpu.rgb),
                                            "reference": modality("reference_rgb", cpu.reference_rgb)})
    assert len(dataset) == N
    sample = dataset[0]
    assert sample["rgb"].shape == sample["reference"].shape == (HEIGHT, WIDTH, 3)
    assert not np.array_equal(sample["rgb"], sample["reference"])
    assert sample["meta"]["rgb"]["path"].replace("_rgb_anon", "") == \
        sample["meta"]["reference"]["path"].replace("_ref", "").replace("_rgb_anon", "")

    fog_train = MultiModalDataset(modalities={"rgb": modality("rgb", cpu.rgb, split="train")})
    assert len(fog_train) == len(frames(split="train"))
    shutil.rmtree(out / "rgb_anon" / "fog" / "test_ref")
    assert not verify(out).ok
