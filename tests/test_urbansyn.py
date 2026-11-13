from __future__ import annotations

import json
import logging
import os
import zipfile

import numpy as np
import pytest
from ds_crawler import index_dataset_from_path

from euler_known_datasets import MediaError, detect_recipes, prepare, verify
from euler_known_datasets.datasets import urbansyn  # noqa: F401  (registers the recipe)
from euler_known_datasets.datasets.urbansyn import SCOPES, scene_of
from euler_known_datasets.datasets.urbansyn.sources import discover

from .fake_urbansyn import (
    FRAMES,
    HEIGHT,
    INCOMPLETE,
    NO_DEPTH,
    WIDTH,
    build_fake_urbansyn,
    depth_meters,
    labels,
)

N = len(FRAMES)


@pytest.fixture
def zipped(tmp_path):
    return build_fake_urbansyn(tmp_path / "downloads")


@pytest.fixture
def extracted(tmp_path):
    return build_fake_urbansyn(tmp_path / "downloads", layout="extracted")


def _files_below(root):
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not p.relative_to(root).parts[0].startswith(".")
        and p.name not in ("euler-known-datasets.json", "downloads.json")
    }


def _head(out, scope):
    return json.loads((out / ".ds_crawler" / scope / "dataset-head.json").read_text())


def test_discover_reads_the_zips(zipped):
    found = discover(zipped.source)
    assert set(found.folders) == {"rgb", "depth", "ss"} and not found.notes
    rgb = found.folders["rgb"]
    assert rgb.archive.name == "rgb.zip" and rgb.member_prefix == "rgb/" and rgb.extracted is None
    assert len(rgb.frames) == N  # Thumbs.db is no frame
    assert NO_DEPTH not in found.folders["depth"].frames
    assert found.camera_file == "camera_metadata.json"


def test_scene_of():
    assert scene_of("0001") == ("poblenou", "Barcelona")
    assert scene_of("2688")[0] == "poblenou" and scene_of("2689")[0] == "torstrasse"
    assert scene_of("5593")[0] == "torstrasse" and scene_of("5594") == ("veteranenstrasse", "Berlin")
    assert scene_of("7539")[0] == "veteranenstrasse" and scene_of("7540") is None


def test_prepare_from_the_zips(zipped, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("urbansyn", zipped.source, out).receipt

    assert receipt["status"] == "complete"
    assert receipt["media"] == {"requested": "auto", "used": "extract"}
    assert receipt["scopes"] == list(SCOPES)
    assert receipt["splits"] == {}
    summary = receipt["summary"]
    # One frame has no depth map; another's lacks the static scene and is left out of depth.
    assert summary["files"] == {"rgb": N, "depth": N - 2, "semantic_segmentation": N, "intrinsics": 1,
                                "camera_pose": N - 1}
    assert summary["frames_without_file"] == {"depth": 1, "camera_pose": 1}
    assert summary["incomplete_depth"] == {"frames": 1, "indexed": False, "left_out": "7000-7000"}
    assert summary["frames_per_scene"] == {"poblenou": 3, "torstrasse": 2, "veteranenstrasse": 3}
    # Only the indexed files, at their released paths: no Thumbs.db, nothing of ss_color.zip.
    # The incomplete depth map is extracted all the same: its header holds the frame's pose.
    assert _files_below(out) == zipped.paths("rgb", "depth", "ss") | {"camera_metadata.json"}
    report = verify(out)
    assert report.ok, report.format_text()
    assert not [w for w in report.warnings if "family" in w]


def test_ids_hierarchy_and_heads(zipped, tmp_path):
    out = tmp_path / "prepared"
    prepare("urbansyn", zipped.source, out)

    index = index_dataset_from_path(out, metadata_scope="depth")["index"]
    assert sorted(index["children"]) == ["scene:poblenou", "scene:torstrasse", "scene:veteranenstrasse"]
    assert index["children"]["scene:torstrasse"]["files"] == [{
        "path": "depth/depth_2689.exr", "id": "2689", "path_properties": {}, "basename_properties": {},
        "attributes": {"city": "Berlin"}}]
    rgb = index_dataset_from_path(out, metadata_scope="rgb")["index"]
    assert [f["id"] for f in rgb["children"]["scene:poblenou"]["files"]] == ["0001", "0002", "2688"]
    intrinsics = index_dataset_from_path(out, metadata_scope="intrinsics")["index"]
    assert "children" not in intrinsics and intrinsics["files"][0]["path"] == "camera_metadata.json"

    depth = _head(out, "depth")
    assert depth["dataset"] == {"id": "urbansyn", "name": "urbansyn", "attributes": {
        "gt": True, "baseline": False, "source": "UrbanSyn",
        "scenes": ["poblenou", "torstrasse", "veteranenstrasse"], "cities": ["Barcelona", "Berlin"]}}
    meta = depth["modality"]["meta"]
    assert (meta["radial_depth"], meta["scale_to_meters"]) == (False, 100000.0)
    assert meta["invalid"] == {"non_finite": False, "sentinel": 0}
    pose = _head(out, "camera_pose")
    assert pose["modality"]["key"] == "camera_extrinsics"
    assert pose["modality"]["meta"]["direction"] == "target_from_source"
    assert pose["addons"]["euler_loading"]["function"] == "camera_pose"
    poses = index_dataset_from_path(out, metadata_scope="camera_pose")["index"]
    assert poses["children"]["scene:poblenou"]["files"][0]["path"] == "depth/depth_0001.exr"
    labels_head = _head(out, "semantic_segmentation")
    assert labels_head["modality"]["key"] == "semantic_segmentation"
    assert labels_head["modality"]["meta"]["skyclass"] == [10, 10, 10]
    assert labels_head["modality"]["meta"]["classes"][10] == "sky"
    assert labels_head["addons"]["euler_loading"] == {
        "version": "1.0", "loader": "urbansyn", "function": "semantic_segmentation"}


def test_options_select_scopes_and_scenes(zipped, tmp_path):
    out = tmp_path / "out"
    receipt = prepare("urbansyn", zipped.source, out, modalities=["rgb", "intrinsics"],
                      scenes=["torstrasse"]).receipt
    assert receipt["scopes"] == ["rgb", "intrinsics"]
    assert receipt["summary"]["files"] == {"rgb": 2, "intrinsics": 1}
    assert _files_below(out) == {"rgb/rgb_2689.png", "rgb/rgb_5593.png", "camera_metadata.json"}
    assert _head(out, "rgb")["dataset"]["attributes"]["scenes"] == ["torstrasse"]


def test_incomplete_depth_can_be_kept(zipped, tmp_path):
    out = tmp_path / "out"
    receipt = prepare("urbansyn", zipped.source, out, modalities=["depth"], keep_incomplete_depth=True).receipt
    assert receipt["summary"]["files"] == {"depth": N - 1}
    assert receipt["summary"]["incomplete_depth"] == {"frames": 1, "indexed": True}
    files = index_dataset_from_path(out, metadata_scope="depth")["index"]["children"]["scene:veteranenstrasse"]["files"]
    marked = {f["id"]: f["attributes"].get("incomplete_depth", False) for f in files}
    assert marked == {"5594": False, INCOMPLETE: True, "7539": False}


def test_symlink_media_from_extracted_folders(extracted, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("urbansyn", extracted.source, out).receipt
    assert receipt["media"]["used"] == "symlink"
    assert (out / "depth").is_symlink() and (out / "camera_metadata.json").is_symlink()
    assert verify(out).ok


@pytest.mark.parametrize("mode", ["hardlink", "copy"])
def test_hardlink_and_copy_from_a_nested_folder(tmp_path, mode):
    fake = build_fake_urbansyn(tmp_path / "downloads", layout="nested")
    out = tmp_path / "prepared"
    receipt = prepare("urbansyn", fake.source, out, media=mode, modalities=["depth", "intrinsics"]).receipt
    assert receipt["media"]["used"] == mode
    assert _files_below(out) == fake.paths("depth") - {f"depth/depth_{INCOMPLETE}.exr"} | {"camera_metadata.json"}
    some = "depth/depth_0001.exr"
    same = os.stat(out / some).st_ino == os.stat(fake.source / "UrbanSyn" / some).st_ino
    assert same == (mode == "hardlink")
    assert verify(out).ok


def test_in_place_extracts_next_to_the_zips(zipped):
    prepare("urbansyn", zipped.source, zipped.source, modalities=["semantic_segmentation"])
    index = index_dataset_from_path(zipped.source, metadata_scope="semantic_segmentation")["index"]
    assert index["children"]["scene:poblenou"]["files"][0]["path"] == "ss/ss_0001.png"
    assert (zipped.source / "ss" / "ss_0001.png").is_file() and not (zipped.source / "rgb").exists()
    assert verify(zipped.source).ok


def test_requested_scope_must_exist(zipped, tmp_path):
    (zipped.source / "depth.zip").unlink()
    with pytest.raises(FileNotFoundError, match="depth"):
        prepare("urbansyn", zipped.source, tmp_path / "out", modalities=["depth"])
    receipt = prepare("urbansyn", zipped.source, tmp_path / "out").receipt
    assert "depth" not in receipt["scopes"]


def test_partial_extraction_falls_back_to_the_zip(zipped):
    (zipped.source / "rgb").mkdir()
    (zipped.source / "rgb" / "rgb_0001.png").write_bytes(b"partial")
    found = discover(zipped.source)
    assert found.folders["rgb"].extracted is None
    assert any("rgb/ lacks" in note for note in found.notes)


def test_extract_needs_the_zips(extracted, tmp_path):
    with pytest.raises(MediaError, match="archives"):
        prepare("urbansyn", extracted.source, tmp_path / "prepared", media="extract")


def test_rerun_skips_finished_zips(zipped, tmp_path, caplog):
    out = tmp_path / "prepared"
    prepare("urbansyn", zipped.source, out, modalities=["rgb", "depth"])
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("urbansyn", zipped.source, out, modalities=["rgb", "depth"])
    skipped = {r.args[0] for r in caplog.records if "already extracted" in r.getMessage()}
    assert skipped == {"rgb.zip", "depth.zip"}


def test_no_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="no UrbanSyn"):
        prepare("urbansyn", tmp_path, tmp_path / "out")


def test_detect(zipped, tmp_path):
    assert "urbansyn" in [recipe.name for recipe in detect_recipes(zipped.source)]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "urbansyn" not in [recipe.name for recipe in detect_recipes(empty)]


def test_a_broken_zip_is_noted(zipped):
    (zipped.source / "ss.zip").write_bytes(b"not a zip")
    found = discover(zipped.source)
    assert any(note.startswith("ss.zip: not a readable zip") for note in found.notes)
    assert "ss" not in found.folders and zipfile.is_zipfile(zipped.source / "rgb.zip")


def _unused_writer(path, value, meta=None):  # pragma: no cover - never called
    raise AssertionError("not writing")


def test_loads_in_euler_loading(zipped, tmp_path):
    cpu = pytest.importorskip("euler_loading.loaders.cpu.urbansyn")
    pytest.importorskip("OpenEXR")
    from euler_loading import Modality, MultiModalDataset

    out = tmp_path / "prepared"
    prepare("urbansyn", zipped.source, out)

    def modality(scope, loader, **kwargs):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=_unused_writer, **kwargs)

    dataset = MultiModalDataset(
        modalities={
            "rgb": modality("rgb", cpu.rgb),
            "depth": modality("depth", cpu.depth),
            "labels": modality("semantic_segmentation", cpu.semantic_segmentation),
            "sky": modality("semantic_segmentation", cpu.sky_mask),
            "pose": modality("camera_pose", cpu.camera_pose),
        },
        hierarchical_modalities={
            "intrinsics": modality("intrinsics", cpu.read_intrinsics, collapse_single=True),
        },
    )
    # The frames without a (complete) depth map drop out of the join.
    assert len(dataset) == N - 2
    sample = next(dataset[i] for i in range(N - 2) if dataset[i]["full_id"] == "/scene:torstrasse/2689")
    assert sample["rgb"].shape == (HEIGHT, WIDTH, 3)
    np.testing.assert_allclose(sample["depth"], depth_meters("2689"), rtol=1e-6)
    np.testing.assert_array_equal(sample["labels"], labels("2689"))
    np.testing.assert_array_equal(sample["sky"], labels("2689") == 10)
    assert sample["intrinsics"][0, 0] == pytest.approx(1730.207, abs=1e-3)
    np.testing.assert_allclose(sample["pose"][:3, 3], [2689.0, 1.3, 0.0])
    # Looking along +z with y up: optical z = world z, optical y = -world y, x = y cross z = -world x.
    np.testing.assert_allclose(sample["pose"][:3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-12)
