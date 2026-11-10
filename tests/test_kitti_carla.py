from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest
from ds_crawler import index_dataset_from_path

from euler_known_datasets import MediaError, prepare, verify
from euler_known_datasets.datasets import kitti_carla  # noqa: F401  (registers the recipe)

from .fake_kitti_carla import (
    FRAMES,
    POINTS,
    SPEED_M_S,
    build_fake_kitti_carla,
    depth_metres,
    needed_files,
)

ALL_SCOPES = [
    "rgb",
    "depth",
    "semantic_segmentation",
    "lidar",
    "intrinsics",
    "camera_extrinsics",
    "lidar_pose",
]


@pytest.fixture
def zips(tmp_path):
    return build_fake_kitti_carla(tmp_path / "dataset")


@pytest.fixture
def extracted(tmp_path):
    return build_fake_kitti_carla(tmp_path / "dataset", archives=False, extracted=True)


def _files_below(root, *tops):
    found = set()
    for top in tops:
        for path in (root / top).rglob("*"):
            if path.is_file():
                found.add(path.relative_to(root).as_posix())
    return found


def _ids(root, scope, *keys):
    node = index_dataset_from_path(root, metadata_scope=scope)["index"]
    for key in keys:
        node = node["children"][key]
    return [entry["id"] for entry in node.get("files", [])], node


def test_prepare_from_town_archives(zips, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("kitti_carla", zips.source, out).receipt
    assert receipt["status"] == "complete"
    assert receipt["media"] == {"requested": "auto", "used": "extract"}
    assert receipt["scopes"] == ALL_SCOPES
    assert receipt["splits"] == {}
    assert receipt["summary"]["towns"]["Town01"] == {
        "rgb": 2 * FRAMES,
        "depth": 2 * FRAMES,
        "semantic_segmentation": 2 * FRAMES,
        "lidar": FRAMES,
        "intrinsics": 2,
        "camera_extrinsics": 2,
        "lidar_pose": 1,
    }
    # Only indexed files: no screenshot, pose PLY, camera timestamps or the
    # archive's own .ds_crawler.
    assert _files_below(out, "Town01", "Town02") == needed_files(zips)
    assert not (out / "Town01" / "generated" / "screenshot.png").exists()
    assert verify(out).ok


def test_ids_and_hierarchy_match_euler_loading(zips, tmp_path):
    out = tmp_path / "prepared"
    prepare("kitti_carla", zips.source, out)
    frames = [f"frame-{frame:04d}" for frame in range(FRAMES)]
    for scope in ("rgb", "depth", "semantic_segmentation", "lidar"):
        for camera in ("0", "1"):
            ids, node = _ids(out, scope, "town:Town02", f"camera:{camera}")
            assert ids == frames, scope
    _, node = _ids(out, "lidar", "town:Town01", "camera:1")
    assert node["files"][0]["path"] == "Town01/generated/frames/frame_0000.ply"
    for scope, file_id in (("intrinsics", "intrinsics"), ("camera_extrinsics", "extrinsics")):
        ids, node = _ids(out, scope, "town:Town01", "camera:1")
        assert ids == [file_id]
        assert node["files"][0]["path"] == "Town01/generated/lidar_to_cam1.txt"
    ids, _ = _ids(out, "lidar_pose", "town:Town01")
    assert ids == ["lidar_pose"]

    # Regex scopes keep their regexes; the per-camera lidar scope is prebuilt.
    rgb_config = json.loads((out / ".ds_crawler/rgb/ds-crawler.json").read_text())
    assert "prebuilt_index_file" not in rgb_config["source"]
    assert "images_rgb" in rgb_config["indexing"]["id"]["regex"]
    lidar_config = json.loads((out / ".ds_crawler/lidar/ds-crawler.json").read_text())
    assert lidar_config["source"]["prebuilt_index_file"] == "index.json"
    depth_head = json.loads((out / ".ds_crawler/depth/dataset-head.json").read_text())
    assert depth_head["modality"]["meta"]["scale_to_meters"] == pytest.approx(1000 / (2**24 - 1))
    seg_head = json.loads((out / ".ds_crawler/semantic_segmentation/dataset-head.json").read_text())
    assert seg_head["modality"]["meta"]["skyclass"] == [70, 130, 180]


def test_options(zips, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare(
        "kitti_carla",
        zips.source,
        out,
        towns=["Town02"],
        cameras=["1"],
        modalities=["rgb", "lidar"],
        frame_step=2,
    ).receipt
    assert receipt["scopes"] == ["rgb", "lidar", "intrinsics", "camera_extrinsics", "lidar_pose"]
    assert receipt["summary"]["towns"] == {
        "Town02": {
            "rgb": FRAMES // 2,
            "lidar": FRAMES // 2,
            "intrinsics": 1,
            "camera_extrinsics": 1,
            "lidar_pose": 1,
        }
    }
    assert _files_below(out, "Town02") == needed_files(
        zips, towns=["Town02"], cameras=["1"], step=2, modalities=["rgb", "lidar"]
    )
    assert not (out / "Town01").exists()
    ids, _ = _ids(out, "lidar", "town:Town02", "camera:1")
    assert ids == ["frame-0000", "frame-0002"]
    assert verify(out).ok


@pytest.mark.parametrize("mode", ["symlink", "hardlink", "copy"])
def test_media_from_extracted_towns(extracted, tmp_path, mode):
    out = tmp_path / "prepared"
    media = "auto" if mode == "symlink" else mode
    receipt = prepare("kitti_carla", extracted.source, out, media=media).receipt
    assert receipt["media"]["used"] == mode
    assert (out / "Town01").is_symlink() == (mode == "symlink")
    some = "Town01/generated/images_rgb/0000_0.png"
    same_inode = os.stat(out / some).st_ino == os.stat(extracted.source / some).st_ino
    assert same_inode == (mode in ("symlink", "hardlink"))
    if mode != "symlink":
        assert _files_below(out, "Town01", "Town02") == needed_files(extracted)
    assert verify(out).ok


def test_partly_extracted_town_is_extracted_from_its_archive(tmp_path):
    fake = build_fake_kitti_carla(tmp_path / "dataset", extracted=True)
    (fake.source / "Town01/generated/images_rgb/0000_0.png").unlink()
    receipt = prepare("kitti_carla", fake.source, tmp_path / "out").receipt
    assert receipt["media"]["used"] == "extract"


def test_symlink_needs_extracted_towns(zips, tmp_path):
    with pytest.raises(MediaError, match="extracted"):
        prepare("kitti_carla", zips.source, tmp_path / "out", media="symlink")


def test_in_place_extracts_next_to_the_archives(zips):
    receipt = prepare("kitti_carla", zips.source, zips.source, towns=["Town01"]).receipt
    assert receipt["media"]["used"] == "in_place"
    assert needed_files(zips, towns=["Town01"]) <= _files_below(zips.source, "Town01")
    assert verify(zips.source).ok


def test_rerun_skips_finished_archives(zips, tmp_path, caplog):
    out = tmp_path / "prepared"
    prepare("kitti_carla", zips.source, out)
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("kitti_carla", zips.source, out)
    assert sum("already extracted" in r.getMessage() for r in caplog.records) == 2


def test_source_errors(tmp_path):
    nested = tmp_path / "KITTI-CARLA"
    build_fake_kitti_carla(nested / "dataset", towns=("Town01",))
    with pytest.raises(FileNotFoundError, match="pass that directory"):
        prepare("kitti_carla", nested, tmp_path / "out")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "KITTI-CARLA.zip").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="unzip it"):
        prepare("kitti_carla", bundle, tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="Town03"):
        prepare("kitti_carla", nested / "dataset", tmp_path / "out", towns=["Town03"])
    with pytest.raises(ValueError, match="frame_step"):
        prepare("kitti_carla", nested / "dataset", tmp_path / "out", frame_step=0)


def test_loads_in_euler_loading(zips, tmp_path):
    pytest.importorskip("euler_loading")
    from euler_loading import Modality, MultiModalDataset
    from euler_loading.loaders.cpu import kitti_carla as kc

    out = tmp_path / "prepared"
    prepare("kitti_carla", zips.source, out, towns=["Town01"])
    root = str(out)

    def modality(scope, loader, writer, **kwargs):
        # Explicit writers: resolving one from the head imports the torch
        # loaders, which a NumPy-only install lacks.
        return Modality(root, metadata_scope=scope, loader=loader, writer=writer, **kwargs)

    dataset = MultiModalDataset(
        modalities={
            "rgb": modality("rgb", kc.rgb, kc.write_rgb),
            "depth": modality("depth", kc.depth, kc.write_depth),
            "classes": modality(
                "semantic_segmentation", kc.class_segmentation, kc.write_class_segmentation
            ),
            "lidar": modality("lidar", kc.lidar_point_cloud, kc.write_lidar_point_cloud),
        },
        hierarchical_modalities={
            "intrinsics": modality(
                "intrinsics", kc.read_intrinsics, kc.write_intrinsics, collapse_single=True
            ),
            "extrinsics": modality(
                "camera_extrinsics", kc.read_extrinsics, kc.write_extrinsics, collapse_single=True
            ),
            "pose": modality("lidar_pose", kc.lidar_pose, lambda *a, **k: None, collapse_single=True),
        },
    )
    assert len(dataset) == 2 * FRAMES
    for index in range(len(dataset)):
        sample = dataset[index]
        town, camera, frame = sample["full_id"].strip("/").split("/")
        camera, frame = camera.split(":")[1], int(frame.split("-")[1])
        assert town == "town:Town01"
        assert sample["rgb"].shape == (2, 3, 3)
        np.testing.assert_allclose(sample["depth"], depth_metres(frame, camera), atol=1e-4)
        assert sample["classes"].tolist() == [[13, 13, 7]] * 2
        np.testing.assert_allclose(sample["lidar"][:, :3], POINTS[:, :3], atol=1e-5)
        # Each camera takes its own calibration file through the hierarchy.
        assert sample["extrinsics"][0, 3] == pytest.approx(-0.5 * int(camera))
        assert sample["intrinsics"].shape == (3, 3)
        # The pose is selected at the sample's frame time (10 Hz).
        assert sample["pose"][0, 3] == pytest.approx(SPEED_M_S * frame / 10.0)
