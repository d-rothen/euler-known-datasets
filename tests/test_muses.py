from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile

import numpy as np
import pytest
from ds_crawler import get_files, index_dataset_from_path, load_dataset_split

from euler_known_datasets import MediaError, detect_recipes, prepare, verify
from euler_known_datasets.datasets import muses  # noqa: F401  (registers the recipe)

from .fake_muses import (
    FRAMES,
    HEIGHT,
    LIDAR_TO_GNSS,
    LIDAR_TO_RGB,
    PACKAGES,
    SKY_ROWS,
    WIDTH,
    K,
    build_fake_muses,
    has_ground_truth,
    has_reference,
)

N = len(FRAMES)
N_REFERENCE = sum(has_reference(w, t) for _, w, t, _, _ in FRAMES)
N_GT = sum(has_ground_truth(split) for split, *_ in FRAMES)
SHARED = ["intrinsics", "camera_extrinsics", "lidar_to_gnss", "sensor_timing"]
FRAME_SCOPES = [
    "rgb",
    "reference_rgb",
    "lidar",
    "gnss",
    "semantic_segmentation",
    "semantic_segmentation_train_ids",
    "semantic_segmentation_color",
]


@pytest.fixture
def zipped(tmp_path):
    return build_fake_muses(tmp_path / "downloads")


@pytest.fixture
def extracted(tmp_path):
    return build_fake_muses(tmp_path / "downloads", archives=False, extracted=True)


def _files_below(root, top="muses"):
    return {
        p.relative_to(root).as_posix() for p in (root / top).rglob("*") if p.is_file()
    }


def _indexed(fake):
    """Every file the prepared scopes reference: no dataformat.txt or tag tables."""
    return {
        path
        for path in fake.files
        if path.endswith(("_frame_camera.png", "_reference_frame.png", "_lidar.bin",
                          "_gnss.txt", "_gt_labelIds.png", "_gt_labelTrainIds.png",
                          "_gt_labelColor.png"))
    } | {"muses/calib.json", "muses/meta.json"}


def test_prepare_from_packages(zipped, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("muses", zipped.source, out).receipt

    assert receipt["status"] == "complete"
    assert receipt["media"] == {"requested": "auto", "used": "extract"}
    assert receipt["scopes"] == FRAME_SCOPES + SHARED
    assert receipt["summary"]["files"] == {
        "rgb": N,
        "reference_rgb": N_REFERENCE,
        "lidar": N,
        "gnss": N,
        "semantic_segmentation": N_GT,
        "semantic_segmentation_train_ids": N_GT,
        "semantic_segmentation_color": N_GT,
    }
    assert receipt["splits"]["rgb"] == ["test", "train", "val"]
    assert receipt["splits"]["semantic_segmentation"] == ["train", "val"]
    assert "intrinsics" not in receipt["splits"]

    # Only indexed files (plus the dataset's documentation) are extracted.
    docs = {"muses/README.md", "muses/License.pdf"}
    assert _files_below(out) == _indexed(zipped) | docs
    assert not (out / ".ds_crawler" / "index.json").exists()  # the packages' stray index
    report = verify(out)
    assert report.ok, report.format_text()


def test_ids_hierarchy_and_shared_files(zipped, tmp_path):
    out = tmp_path / "prepared"
    prepare("muses", zipped.source, out)

    rgb = index_dataset_from_path(out, metadata_scope="rgb")
    assert sorted(rgb["index"]["children"]) == ["test", "train", "val"]
    node = rgb["index"]["children"]["train"]["children"]["fog"]["children"]["night"]
    assert [f["id"] for f in node["files"]] == ["000201", "000202"]
    assert node["files"][0]["path"] == "muses/frame_camera/train/fog/night/REC0002_frame_000201_frame_camera.png"
    assert node["files"][0]["basename_properties"] == {"sequence": "REC0002", "frame": "000201"}
    config = json.loads((out / ".ds_crawler" / "rgb" / "ds-crawler.json").read_text())
    assert "prebuilt_index_file" not in config["source"]  # regexes kept: re-crawlable

    for scope in SHARED:
        index = index_dataset_from_path(out, metadata_scope=scope)
        assert "children" not in index["index"]  # root level: applies to every sample
        assert [f["id"] for f in index["index"]["files"]] == [scope]
    head = json.loads((out / ".ds_crawler" / "semantic_segmentation_color" / "dataset-head.json").read_text())
    assert head["modality"]["meta"]["skyclass"] == [70, 130, 180]

    splits = load_dataset_split(out, "val", metadata_scope="rgb")
    assert sorted(p.split("/")[2] for p in get_files(splits)) == ["val", "val"]


def test_symlink_media_from_extracted_source(extracted, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("muses", extracted.source, out).receipt
    assert receipt["media"]["used"] == "symlink"
    assert (out / "muses").is_symlink()
    assert verify(out).ok


@pytest.mark.parametrize("mode", ["hardlink", "copy"])
def test_hardlink_and_copy(extracted, tmp_path, mode):
    out = tmp_path / "prepared"
    receipt = prepare("muses", extracted.source, out, media=mode).receipt
    assert receipt["media"]["used"] == mode
    assert not (out / "muses").is_symlink()
    docs = {"muses/README.md", "muses/License.pdf"}
    assert _files_below(out) == _indexed(extracted) | docs
    some = "muses/calib.json"
    same = os.stat(out / some).st_ino == os.stat(extracted.source / some).st_ino
    assert same == (mode == "hardlink")
    assert verify(out).ok


def test_extract_needs_the_packages(extracted, tmp_path):
    with pytest.raises(MediaError, match="archives"):
        prepare("muses", extracted.source, tmp_path / "prepared", media="extract")


def test_in_place(zipped):
    receipt = prepare("muses", zipped.source, zipped.source).receipt
    assert receipt["media"]["used"] == "in_place"
    assert _indexed(zipped) <= _files_below(zipped.source)
    assert verify(zipped.source).ok


def test_rerun_skips_finished_packages(zipped, tmp_path, caplog):
    out = tmp_path / "prepared"
    prepare("muses", zipped.source, out)
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("muses", zipped.source, out)
    skipped = [r for r in caplog.records if "already extracted" in r.getMessage()]
    assert len(skipped) == len(PACKAGES)


def test_options_select_modalities_and_conditions(zipped, tmp_path):
    receipt = prepare(
        "muses",
        zipped.source,
        tmp_path / "out",
        modalities=["rgb", "lidar"],
        splits=["val", "test"],
        weather=["rain", "snow"],
    ).receipt
    assert receipt["scopes"] == ["rgb", "lidar"] + SHARED
    expected = sum(1 for s, w, *_ in FRAMES if s in ("val", "test") and w in ("rain", "snow"))
    assert receipt["summary"]["files"] == {"rgb": expected, "lidar": expected}
    assert receipt["splits"]["rgb"] == ["test", "val"]
    assert not (tmp_path / "out" / "muses" / "gnss").exists()


def test_requested_modality_must_exist(zipped, tmp_path):
    with pytest.raises(FileNotFoundError, match="panoptic_segmentation"):
        prepare("muses", zipped.source, tmp_path / "out", modalities=["panoptic_segmentation"])
    assert not (tmp_path / "out").exists()


def test_no_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="no MUSES downloads"):
        prepare("muses", tmp_path, tmp_path / "out")


def test_official_package_wins_over_a_copy(zipped, tmp_path, caplog):
    # A re-zipped copy holding the same paths with other bytes must not win.
    official = zipped.source / "frame_camera_trainvaltest.zip"
    copy = zipped.source / "rgb.zip"
    with zipfile.ZipFile(official) as src, zipfile.ZipFile(copy, "w") as dst:
        for info in src.infolist():
            dst.writestr(info, b"not the official image" if info.filename.endswith(".png") else src.read(info))
    out = tmp_path / "prepared"
    with caplog.at_level(logging.WARNING):
        prepare("muses", zipped.source, out, modalities=["rgb"])
    assert "several packages" in caplog.text
    name = "muses/frame_camera/val/clear/day/REC0003_frame_000301_frame_camera.png"
    assert (out / name).read_bytes() == zipped.files[name]


def test_detect(zipped, tmp_path):
    assert "muses" in [recipe.name for recipe in detect_recipes(zipped.source)]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "muses" not in [recipe.name for recipe in detect_recipes(empty)]


def _unused_writer(path, value, meta=None):  # pragma: no cover - never called
    raise AssertionError("not writing")


def _cpu_modality(root, scope, loader, **kwargs):
    """A modality read with an explicit CPU loader.

    The explicit writer keeps euler-loading from resolving the head's writer,
    which imports its torch loaders; the tests run without torch.
    """
    from euler_loading import Modality

    return Modality(str(root), metadata_scope=scope, loader=loader, writer=_unused_writer, **kwargs)


def test_loads_in_euler_loading(zipped, tmp_path):
    pytest.importorskip("euler_loading")
    from euler_loading import MultiModalDataset
    from euler_loading.loaders.cpu import muses as cpu

    out = tmp_path / "prepared"
    prepare("muses", zipped.source, out)
    root = out
    dataset = MultiModalDataset(
        modalities={
            "rgb": _cpu_modality(root, "rgb", cpu.rgb),
            "lidar": _cpu_modality(root, "lidar", cpu.lidar_point_cloud),
            "gnss": _cpu_modality(root, "gnss", cpu.gnss),
        },
        hierarchical_modalities={
            name: _cpu_modality(root, name, loader, collapse_single=True)
            for name, loader in (
                ("intrinsics", cpu.read_intrinsics),
                ("camera_extrinsics", cpu.read_extrinsics),
                ("lidar_to_gnss", cpu.read_lidar_to_gnss),
                ("sensor_timing", cpu.sensor_timing),
            )
        },
    )
    assert len(dataset) == N
    sample = dataset[0]
    assert sample["rgb"].shape == (HEIGHT, WIDTH, 3)
    assert sample["lidar"].shape == (6, 6)
    assert sample["gnss"]["ground_speed_m_s"] == pytest.approx(5.833)
    assert np.allclose(sample["intrinsics"], K)
    assert np.allclose(sample["camera_extrinsics"], LIDAR_TO_RGB)
    assert np.allclose(sample["lidar_to_gnss"], LIDAR_TO_GNSS)
    # The timing record is the sample's own frame, found through its file names.
    frame = os.path.basename(sample["meta"]["rgb"]["path"]).rsplit("_frame_camera", 1)[0]
    assert sample["sensor_timing"]["frame"] == frame

    # Reference frames exist only for adverse conditions; joining drops clear day.
    with_reference = MultiModalDataset(
        modalities={
            "rgb": _cpu_modality(root, "rgb", cpu.rgb),
            "reference": _cpu_modality(root, "reference_rgb", cpu.reference_rgb),
        }
    )
    assert len(with_reference) == N_REFERENCE


@pytest.mark.parametrize(
    "scope",
    ["semantic_segmentation", "semantic_segmentation_train_ids", "semantic_segmentation_color"],
)
def test_sky_masks_follow_the_head_skyclass(zipped, tmp_path, scope):
    pytest.importorskip("euler_loading")
    from euler_loading import MultiModalDataset
    from euler_loading.loaders.cpu import muses as cpu

    out = tmp_path / "prepared"
    prepare("muses", zipped.source, out, modalities=[scope])
    dataset = MultiModalDataset(modalities={"sky": _cpu_modality(out, scope, cpu.sky_mask)})
    mask = np.asarray(dataset[0]["sky"]).reshape(HEIGHT, WIDTH)
    assert mask[:SKY_ROWS].all() and not mask[SKY_ROWS:].any()


def test_output_tree_matches_the_official_layout(extracted, tmp_path):
    # A prepared output is still a MUSES root for the official SDK.
    out = tmp_path / "prepared"
    prepare("muses", extracted.source, out, media="copy")
    shutil.rmtree(out / ".ds_crawler")
    assert (out / "muses" / "meta.json").is_file()
    assert (out / "muses" / "frame_camera" / "train" / "clear" / "day").is_dir()


def test_a_scope_without_matching_files_is_skipped(zipped, tmp_path):
    # MUSES has no ground truth for test.
    out = tmp_path / "out"
    receipt = prepare(
        "muses", zipped.source, out, modalities=["rgb", "semantic_segmentation"], splits=["test"]
    ).receipt
    assert receipt["scopes"] == ["rgb"] + SHARED
    assert not (out / ".ds_crawler" / "semantic_segmentation").exists()
    assert verify(out).ok
