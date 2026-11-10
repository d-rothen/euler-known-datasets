from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from ds_crawler import get_files, index_dataset_from_path, load_dataset_split

from euler_known_datasets import MediaError, prepare, verify
from euler_known_datasets.datasets import driving_stereo  # noqa: F401  (registers)
from euler_known_datasets.datasets.driving_stereo.layout import (
    STREAMS,
    classify_archive,
)

from .fake_driving_stereo import (
    FOGGY,
    FRAMES,
    K_HALF,
    TEST,
    TRAIN,
    build_fake_driving_stereo,
    frames,
)

MEDIA_SCOPES = ["rgb", "depth", "disparity", "rgb_full_size", "rgb_right", "depth_full_size",
                "disparity_full_size"]


@pytest.fixture
def zipped(tmp_path):
    return build_fake_driving_stereo(tmp_path / "downloads")


@pytest.fixture
def unzipped(tmp_path):
    return build_fake_driving_stereo(tmp_path / "downloads", archives=False, extracted=True)


def _media_below(root):
    found = set()
    for top in ("weather", "train", "test"):
        for path in (root / top).rglob("*"):
            if path.is_file() and path.suffix in (".jpg", ".png"):
                found.add(path.relative_to(root).as_posix())
    return found


def test_classify_archives():
    assert classify_archive("weather/foggy/left-image-half-size.zip").stream.scope == "rgb"
    assert classify_archive("weather/rainy/right-image-full-size.zip").stream.scope == (
        "rgb_right_full_size"
    )
    info = classify_archive(f"train/disparity-map/{TRAIN}.zip")
    assert (info.subset, info.stream.scope, info.directory) == ("train", "disparity", "train/disparity-map")
    assert classify_archive(f"test/full-size/depth-map/{TEST}.zip").stream.scope == "depth_full_size"
    assert classify_archive("weather/foggy/notes.zip") is None
    assert {s.scope for s in STREAMS} == set(MEDIA_SCOPES) | {"rgb_right_full_size"}


def test_prepare_from_zips(zipped, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("driving_stereo", zipped.source, out).receipt
    assert receipt["status"] == "complete"
    assert receipt["media"] == {"requested": "auto", "used": "extract"}
    assert receipt["scopes"] == [
        "rgb", "rgb_right", "depth", "disparity", "rgb_full_size",
        "depth_full_size", "disparity_full_size", "intrinsics",
    ]
    assert receipt["splits"]["rgb"] == ["foggy", "test", "train"]
    assert receipt["splits"]["rgb_right"] == ["foggy"]
    summary = receipt["summary"]
    assert summary["subsets"]["foggy"] == {
        "sequences": 1,
        "files": {s: FRAMES for s in ["rgb", "rgb_right", "depth", "disparity",
                                       "depth_full_size", "disparity_full_size"]},
    }
    assert summary["subsets"]["train"]["files"] == {"rgb": FRAMES, "depth": FRAMES}
    assert summary["subsets"]["test"]["files"] == {"rgb": FRAMES, "rgb_full_size": FRAMES}
    assert summary["intrinsics"] == {"subset_sequences": 3, "calibration_files": 3, "missing": []}

    # Archive members land where unzipping in place puts them; only indexed ones.
    assert _media_below(out) == set(zipped.extracted_paths())
    assert not (out / "weather/foggy/foggy/left-image-half-size/README.txt").exists()
    assert (out / f"calibration/train/half-image-calib/{FOGGY}.txt").is_file()
    assert not (out / "calibration/train/full-image-calib").exists()
    assert verify(out).ok


def test_index_ids_hierarchy_and_heads(zipped, tmp_path):
    out = tmp_path / "prepared"
    prepare("driving_stereo", zipped.source, out)
    rgb = index_dataset_from_path(out, metadata_scope="rgb")
    children = rgb["index"]["children"]
    assert sorted(children) == ["subset:foggy", "subset:test", "subset:train"]
    node = children["subset:train"]["children"][f"sequence:{TRAIN}"]
    assert [f["id"] for f in node["files"]] == [f"frame-{f}" for f in frames(TRAIN)]
    assert node["files"][0]["path"] == f"train/left-image/{TRAIN}/{TRAIN}_{frames(TRAIN)[0]}.jpg"

    config = json.loads((out / ".ds_crawler/depth/ds-crawler.json").read_text())
    assert "prebuilt_index_file" not in config["source"]  # regexes are kept
    head = json.loads((out / ".ds_crawler/depth/dataset-head.json").read_text())
    assert head["addons"]["euler_loading"] == {
        "version": "1.0", "loader": "driving_stereo", "function": "depth",
    }
    meta = head["modality"]["meta"]
    assert (meta["radial_depth"], meta["scale_to_meters"], meta["sparse"]) == (False, 1 / 256, True)
    assert meta["invalid"] == {"non_finite": False, "sentinel": 0}
    full = json.loads((out / ".ds_crawler/disparity_full_size/dataset-head.json").read_text())
    assert full["addons"]["euler_loading"]["function"] == "disparity_full_size"


def test_intrinsics_follow_each_sequence(zipped, tmp_path):
    out = tmp_path / "prepared"
    prepare("driving_stereo", zipped.source, out)
    index = index_dataset_from_path(out, metadata_scope="intrinsics")["index"]["children"]
    path = lambda subset, seq: index[f"subset:{subset}"]["children"][f"sequence:{seq}"]["files"][0]  # noqa: E731
    # The foggy subset's sequence is a training sequence: its calibration is train's.
    assert path("foggy", FOGGY)["path"] == f"calibration/train/half-image-calib/{FOGGY}.txt"
    assert path("train", TRAIN)["path"] == f"calibration/train/half-image-calib/{TRAIN}.txt"
    assert path("test", TEST)["path"] == f"calibration/test/half-image-calib/{TEST}.txt"
    assert path("test", TEST)["id"] == "intrinsics"


def test_without_calibration(tmp_path):
    fake = build_fake_driving_stereo(tmp_path / "downloads", calibration=False)
    plan = prepare("driving_stereo", fake.source, tmp_path / "dry", dry_run=True).plan
    assert any("intrinsics scope is skipped" in step for step in plan.steps)
    receipt = prepare("driving_stereo", fake.source, tmp_path / "prepared").receipt
    assert "intrinsics" not in receipt["scopes"]
    assert len(receipt["summary"]["intrinsics"]["missing"]) == 3
    assert verify(tmp_path / "prepared").ok


def test_splits(zipped, tmp_path):
    out = tmp_path / "prepared"
    prepare("driving_stereo", zipped.source, out)
    foggy = get_files(load_dataset_split(out, "foggy", metadata_scope="depth"))
    assert len(foggy) == FRAMES and all(p.startswith("weather/foggy/") for p in foggy)
    train = get_files(load_dataset_split(out, "train", metadata_scope="rgb"))
    assert len(train) == FRAMES and all(p.startswith("train/") for p in train)


@pytest.mark.parametrize("mode", ["auto", "hardlink", "copy"])
def test_extracted_source(unzipped, tmp_path, mode):
    out = tmp_path / "prepared"
    receipt = prepare("driving_stereo", unzipped.source, out, media=mode).receipt
    assert receipt["media"]["used"] == ("symlink" if mode == "auto" else mode)
    assert (out / "weather").is_symlink() == (mode == "auto")
    assert _media_below(out) == set(unzipped.extracted_paths())
    assert receipt["scopes"][-1] == "intrinsics"
    assert verify(out).ok


def test_symlink_needs_extracted_files(zipped, tmp_path):
    with pytest.raises(MediaError, match="extracted"):
        prepare("driving_stereo", zipped.source, tmp_path / "out", media="symlink")


def test_in_place(zipped):
    receipt = prepare("driving_stereo", zipped.source, zipped.source).receipt
    assert receipt["media"]["used"] == "in_place"
    assert set(zipped.extracted_paths()) <= _media_below(zipped.source)
    assert verify(zipped.source).ok


def test_rerun_skips_finished_archives(zipped, tmp_path, caplog):
    out = tmp_path / "prepared"
    prepare("driving_stereo", zipped.source, out)
    first = (out / ".ds_crawler/rgb/index.json").read_text()
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("driving_stereo", zipped.source, out)
    skipped = [r for r in caplog.records if "already extracted" in r.getMessage()]
    assert len(skipped) == len(zipped.archives)
    assert (out / ".ds_crawler/rgb/index.json").read_text() == first


def test_options(zipped, tmp_path):
    receipt = prepare(
        "driving_stereo",
        zipped.source,
        tmp_path / "a",
        subsets=["foggy"],
        modalities=["rgb", "depth", "intrinsics"],
    ).receipt
    assert receipt["scopes"] == ["rgb", "depth", "intrinsics"]
    assert receipt["splits"] == {"rgb": ["foggy"], "depth": ["foggy"]}
    assert not (tmp_path / "a" / "train").exists()
    assert not (tmp_path / "a" / "weather/foggy/foggy/right-image-half-size").exists()
    with pytest.raises(FileNotFoundError, match="sunny"):
        prepare("driving_stereo", zipped.source, tmp_path / "b", subsets=["sunny"])
    with pytest.raises(FileNotFoundError, match="no files of the requested modalities"):
        prepare("driving_stereo", zipped.source, tmp_path / "c", subsets=["train"],
                modalities=["rgb_right"])


def test_no_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="no DrivingStereo downloads"):
        prepare("driving_stereo", tmp_path, tmp_path / "out")


def test_loads_in_euler_loading(zipped, tmp_path):
    pytest.importorskip("euler_loading")
    from euler_loading import Modality, MultiModalDataset
    from euler_loading.loaders.cpu import driving_stereo as ds

    out = tmp_path / "prepared"
    prepare("driving_stereo", zipped.source, out, subsets=["foggy"])
    root = str(out)

    def modality(scope, function, **kwargs):
        # Explicit writers keep euler-loading from importing the torch variant.
        return Modality(root, metadata_scope=scope, loader=getattr(ds, function),
                        writer=getattr(ds, "write_" + function.replace("read_", "", 1)), **kwargs)

    dataset = MultiModalDataset(
        modalities={
            "rgb": modality("rgb", "rgb", split="foggy"),
            "depth": modality("depth", "depth", split="foggy"),
            "depth_full": modality("depth_full_size", "depth_full_size", split="foggy"),
            "disparity_full": modality("disparity_full_size", "disparity_full_size", split="foggy"),
        },
        hierarchical_modalities={
            "intrinsics": modality("intrinsics", "read_intrinsics", collapse_single=True),
        },
    )
    assert len(dataset) == FRAMES
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["full_id"].startswith(f"/subset:foggy/sequence:{FOGGY}/frame-")
        frame = frames(FOGGY).index(sample["id"].split("-", 1)[1])
        assert sample["rgb"].shape == (4, 6, 3)
        # Every map lands on the half-size left-image grid.
        np.testing.assert_allclose(sample["depth"], np.full((4, 6), 10.0 + frame))
        np.testing.assert_allclose(sample["depth_full"], sample["depth"])
        np.testing.assert_allclose(sample["disparity_full"], np.full((4, 6), 1.0 + frame))
        np.testing.assert_allclose(sample["intrinsics"], K_HALF, rtol=1e-6)
