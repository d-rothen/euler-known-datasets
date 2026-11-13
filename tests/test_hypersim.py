from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest

pytest.importorskip("h5py")

from ds_crawler import get_files, index_dataset_from_path, load_dataset_split  # noqa: E402

from euler_known_datasets import MediaError, detect_recipes, prepare, verify  # noqa: E402
from euler_known_datasets.datasets import hypersim  # noqa: E402,F401  (registers the recipe)
from euler_known_datasets.datasets.hypersim.camera import scene_camera  # noqa: E402

from .fake_hypersim import (  # noqa: E402
    HEIGHT,
    SCENES,
    UNLISTED,
    WIDTH,
    build_fake_hypersim,
    frame_path,
)


@pytest.fixture
def extracted(tmp_path):
    return build_fake_hypersim(tmp_path / "downloads")


def _files(index, keys=()):
    node = index
    for key in keys:
        node = node["children"][key]
    return node.get("files", [])


def _unused_writer(path, value, meta=None):  # pragma: no cover - never called
    raise AssertionError("not writing")


def test_prepare_from_extracted_scenes(extracted, tmp_path):
    out = tmp_path / "prepared"
    result = prepare("hypersim", extracted.source, out)
    receipt = result.receipt
    assert receipt["status"] == "complete"
    assert receipt["media"] == {"requested": "auto", "used": "symlink"}
    assert receipt["scopes"] == ["rgb", "depth", "radial_distance", "intrinsics", "camera_pose"]
    summary = receipt["summary"]
    assert summary["by_split"] == {"test": 4, "train": 8, "val": 4}
    assert summary["scenes"] == 3 and summary["trajectories"] == 4
    assert summary["cameras"]["tilted_scenes"] == 2 and summary["cameras"]["skewed_scenes"] == 1
    assert summary["cameras"]["orientation_notes"] == {
        "ai_003_001/cam_00": {"uniformly_scaled_frames": [0]},
        "ai_001_001/cam_01": {"frames_rendered_with_another_camera": [0]}}
    assert summary["cameras"]["frames_rendered_with_another_camera"] == ["ai_001_001/cam_01/0000"]
    assert summary["skipped"] == {"frames_rendered_with_another_camera": 1}
    assert summary["tonemap"]["frames_without_entity_ids"] == 0
    assert summary["depth"]["frames_with_invalid_pixels"] == 16
    assert receipt["splits"] == {s: ["test", "train", "val"] for s in ("rgb", "depth", "radial_distance")}
    assert (out / "calibration" / "ai_002_001" / "cam_00.json").is_file()
    assert not (out / UNLISTED).exists()
    report = verify(out)
    assert report.ok, report.format_text()


def test_index_ids_attributes_and_splits(extracted, tmp_path):
    out = tmp_path / "prepared"
    prepare("hypersim", extracted.source, out)
    keys = ("scene:ai_002_001", "camera:cam_00")
    rgb = _files(index_dataset_from_path(out, metadata_scope="rgb")["index"], keys)
    depth = _files(index_dataset_from_path(out, metadata_scope="depth")["index"], keys)
    assert [f["id"] for f in rgb] == ["0000", "0001", "0002", "0003"]
    assert rgb[0]["path"] == frame_path("ai_002_001", "cam_00", 0, "color")
    assert depth[0]["path"] == frame_path("ai_002_001", "cam_00", 0, "depth_meters")
    # The tone-mapping scale is Hypersim's, over the pixels with geometry.
    color, entity = extracted.colors[("ai_002_001", "cam_00", 0)], extracted.entities[("ai_002_001", "cam_00", 0)]
    brightness = (0.3 * color[..., 0] + 0.59 * color[..., 1] + 0.11 * color[..., 2])[entity != -1]
    assert rgb[0]["attributes"]["tonemap_scale"] == pytest.approx(0.8 ** 2.2 / np.percentile(brightness, 90))
    camera = scene_camera(WIDTH, HEIGHT, SCENES["ai_002_001"][0])
    np.testing.assert_allclose(depth[0]["attributes"]["intrinsics"], camera.intrinsics)
    assert camera.tilt_deg > 5

    head = json.loads((out / ".ds_crawler" / "depth" / "dataset-head.json").read_text())
    assert head["dataset"]["id"] == "hypersim" and head["dataset"]["attributes"]["gt"] is True
    assert {"radial_depth": False, "scale_to_meters": 1.0}.items() <= head["modality"]["meta"].items()
    assert head["modality"]["meta"]["dimensions"] == {"height": HEIGHT, "width": WIDTH}
    assert head["addons"]["euler_loading"] == {"version": "1.0", "loader": "hypersim", "function": "depth"}
    val = load_dataset_split(out, "val", metadata_scope="rgb")
    assert sorted(get_files(val)) == [frame_path("ai_002_001", "cam_00", f, "color") for f in range(4)]


def test_depth_and_poses_reproduce_hypersims_world_points(extracted, tmp_path):
    """Planar depth, K and the pose put every pixel where Hypersim's own rays do."""
    cpu = pytest.importorskip("euler_loading.loaders.cpu.hypersim")
    from euler_loading import Modality, MultiModalDataset

    out = tmp_path / "prepared"
    prepare("hypersim", extracted.source, out)

    def modality(scope, loader):
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=_unused_writer)

    dataset = MultiModalDataset(
        modalities={"rgb": modality("rgb", cpu.rgb), "depth": modality("depth", cpu.depth)},
        hierarchical_modalities={"intrinsics": modality("intrinsics", cpu.read_intrinsics),
                                 "pose": modality("camera_pose", cpu.camera_pose)},
    )
    assert len(dataset) == 16
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    pixels = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    for index in range(len(dataset)):
        sample = dataset[index]
        _, scene, camera, frame = sample["full_id"].split("/")
        key = (scene.split(":")[1], camera.split(":")[1], int(frame))
        depth = sample["depth"]
        K = np.asarray(next(iter(sample["intrinsics"].values())), dtype=np.float64)
        pose = np.asarray(next(iter(sample["pose"].values())), dtype=np.float64)
        assert sample["rgb"].shape == (HEIGHT, WIDTH, 3) and 0 <= sample["rgb"].min() <= sample["rgb"].max() <= 1
        assert depth[0, 0] == 0  # no geometry
        np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-5)
        points = pose[:3, :3] @ (np.linalg.inv(K) @ pixels * depth.ravel()) + pose[:3, 3:4]
        expected = extracted.world_points[key].reshape(-1, 3).T
        valid = depth.ravel() > 0
        np.testing.assert_allclose(points[:, valid], expected[:, valid], atol=2e-3)


def test_options(extracted, tmp_path):
    out = tmp_path / "selected"
    receipt = prepare("hypersim", extracted.source, out, modalities=["rgb", "radial_distance"], splits=["train"],
                      frame_step=2, scenes=["ai_001_*"]).receipt
    assert receipt["scopes"] == ["rgb", "radial_distance"]
    assert receipt["summary"]["by_split"] == {"train": 4}  # cam_00: 0, 2, 4 (5 is not public); cam_01: 2 (0 is stretched)
    assert not (out / "calibration").exists()
    head = json.loads((out / ".ds_crawler" / "radial_distance" / "dataset-head.json").read_text())
    assert head["modality"]["meta"]["radial_depth"] is True
    assert verify(out).ok
    with pytest.raises(FileNotFoundError, match="matches"):
        prepare("hypersim", extracted.source, tmp_path / "none", scenes=["ai_999_*"])


def test_extract_from_scene_zips(tmp_path, caplog):
    fake = build_fake_hypersim(tmp_path / "downloads", layout="zip")
    out = tmp_path / "prepared"
    receipt = prepare("hypersim", fake.source, out).receipt
    assert receipt["media"]["used"] == "extract"
    assert receipt["summary"]["frames"] == 16
    # Only what the scopes index or the recipe reads: no normals.
    assert not list(out.rglob("*.normal_cam.hdf5"))
    assert (out / frame_path("ai_001_001", "cam_00", 0, "color")).is_file()
    assert verify(out).ok
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("hypersim", fake.source, out)
    assert any("already extracted" in r.getMessage() for r in caplog.records)


def test_hardlink_from_a_subfolder(tmp_path):
    fake = build_fake_hypersim(tmp_path / "downloads", layout="nested")
    out = tmp_path / "prepared"
    receipt = prepare("hypersim", fake.source, out, media="hardlink").receipt
    assert receipt["media"]["used"] == "hardlink"
    rel = "scenes/" + frame_path("ai_001_001", "cam_01", 3, "depth_meters")
    assert os.stat(out / rel).st_ino == os.stat(fake.source / rel).st_ino
    # Camera and render-entity files are read from the source, not linked.
    assert not list(out.rglob("*render_entity_id*")) and not list(out.rglob("camera_keyframe_*"))
    index = index_dataset_from_path(out, metadata_scope="depth")["index"]
    by_id = {f["id"]: f for f in _files(index, ("scene:ai_001_001", "camera:cam_01"))}
    assert by_id["0003"]["path"] == rel and "0000" not in by_id  # frame 0 is stretched
    assert verify(out).ok


def test_in_place(extracted):
    receipt = prepare("hypersim", extracted.source, extracted.source).receipt
    assert receipt["media"]["used"] == "in_place"
    assert verify(extracted.source).ok


def test_needs_the_metadata_csvs(extracted, tmp_path):
    os.remove(extracted.source / "ml-hypersim/contrib/mikeroberts3000/metadata_camera_parameters.csv")
    with pytest.raises(FileNotFoundError, match="metadata_camera_parameters.csv"):
        prepare("hypersim", extracted.source, tmp_path / "out")


def test_media_errors(extracted, tmp_path):
    with pytest.raises(MediaError, match="archives"):
        prepare("hypersim", extracted.source, tmp_path / "out", media="extract")


def test_dark_frames_are_left_out(extracted, tmp_path):
    """A render too dark to tone-map (scale 0) would load black."""
    import h5py

    path = extracted.source / frame_path("ai_002_001", "cam_00", 2, "color")
    with h5py.File(path, "r+") as handle:
        handle["dataset"][...] = handle["dataset"][()] * 1e-6
    receipt = prepare("hypersim", extracted.source, tmp_path / "out").receipt
    assert receipt["summary"]["skipped"]["dark_frames"] == 1
    assert receipt["summary"]["by_split"]["val"] == 3
    assert receipt["summary"]["tonemap"]["dark_frames_by_scene"] == {"ai_002_001": 1}
    index = index_dataset_from_path(tmp_path / "out", metadata_scope="radial_distance")["index"]
    assert [f["id"] for f in _files(index, ("scene:ai_002_001", "camera:cam_00"))] == ["0000", "0001", "0003"]
    kept = prepare("hypersim", extracted.source, tmp_path / "kept", keep_dark_frames=True).receipt
    assert kept["summary"]["by_split"]["val"] == 4 and kept["summary"]["tonemap"]["dark_frames_kept"]


def test_frames_without_files_are_skipped(extracted, tmp_path):
    os.remove(extracted.source / frame_path("ai_001_001", "cam_01", 1, "depth_meters"))
    receipt = prepare("hypersim", extracted.source, tmp_path / "out").receipt
    assert receipt["summary"]["skipped"] == {"frames_without_files": 1, "frames_rendered_with_another_camera": 1}
    assert receipt["summary"]["frames"] == 15


def test_detect(extracted, tmp_path):
    assert "hypersim" in [recipe.name for recipe in detect_recipes(extracted.source)]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "hypersim" not in [recipe.name for recipe in detect_recipes(empty)]
