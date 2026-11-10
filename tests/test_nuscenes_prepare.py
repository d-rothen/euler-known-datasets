from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest
from ds_crawler import get_files, index_dataset_from_path, load_dataset_split

from euler_known_datasets import MediaError, PrepareError, prepare, verify
from euler_known_datasets.geometry import rigid_transform

from .fake_nuscenes import (
    CAMERA_OFFSET_US,
    CAMERA_POSES,
    CAMERA_SWEEPS,
    DROPPED,
    INTRINSICS,
    KEYFRAME_SWEEPS,
    LATE,
    LATE_BY_US,
    LIDAR_POSE,
    build_fake_nuscenes,
    ego_pose,
)

FRAMES_PER_SCENE = len(CAMERA_SWEEPS)
KEYFRAMES_PER_SCENE = len(KEYFRAME_SWEEPS)


@pytest.fixture
def fake(tmp_path):
    return build_fake_nuscenes(tmp_path / "downloads")


@pytest.fixture
def extracted(tmp_path):
    return build_fake_nuscenes(tmp_path / "downloads", archives=False, extracted=True)


def _needed(fake, camera_filter=None):
    """Camera files plus the sweeps they pair with (every camera sweep but the dropped)."""
    cameras = [p for p in fake.files if "/CAM_" in p]
    if camera_filter:
        cameras = [p for p in cameras if f"/{camera_filter}/" in p]
    lidar = []
    for index, scene in enumerate(fake.scenes):
        rows = fake.rows("LIDAR_TOP", scene)
        for k in CAMERA_SWEEPS:
            if index == 1 and k == DROPPED:
                continue
            nominal = 50_000 * k + (LATE_BY_US if index == 1 and k == LATE else 0)
            t0 = rows[0]["timestamp"]
            lidar += [r["filename"] for r in rows if r["timestamp"] - t0 == nominal]
    return set(cameras) | set(lidar)


def _files_below(root, *tops):
    found = set()
    for top in tops:
        for path in (root / top).rglob("*"):
            if path.is_file():
                found.add(path.relative_to(root).as_posix())
    return found


def test_prepare_from_archives(fake, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("nuscenes", fake.source, out).receipt

    assert receipt["status"] == "complete"
    assert receipt["recipe"]["name"] == "nuscenes"
    assert receipt["media"] == {"requested": "auto", "used": "extract"}
    assert receipt["scopes"] == ["rgb", "lidar", "intrinsics", "camera_extrinsics"]
    assert receipt["splits"] == {
        "rgb": ["test", "test_keyframes"],
        "lidar": ["test", "test_keyframes"],
    }
    summary = receipt["summary"]
    assert summary["nominal_lidar_offset_ms"] == {
        camera: offset / 1e3 for camera, offset in CAMERA_OFFSET_US.items()
    }
    for camera in CAMERA_OFFSET_US:
        assert summary["versions"]["v1.0-test"]["cameras"][camera] == {
            "frames": 2 * FRAMES_PER_SCENE,
            "synced": 2 * FRAMES_PER_SCENE - 1,  # the frame of the dropped sweep
            "keyframes": 2 * KEYFRAMES_PER_SCENE,
            "keyframes_synced": 2 * KEYFRAMES_PER_SCENE,
            "keyframes_paired_as_sample": 2 * KEYFRAMES_PER_SCENE,
            "unsynced": 1,
        }

    # Only the files the index uses are extracted: no radar, no unpaired sweeps.
    assert _files_below(out, "samples", "sweeps") == _needed(fake)
    assert (out / "v1.0-test" / "sample_data.json").is_file()
    assert (out / "maps" / "fake_map.png").is_file()
    assert (out / "LICENSE").is_file()
    assert oct(os.stat(out / "euler-known-datasets.json").st_mode & 0o044) != "0o0"
    assert verify(out).ok


def test_index_ids_hierarchy_and_attributes(fake, tmp_path):
    out = tmp_path / "prepared"
    prepare("nuscenes", fake.source, out)
    scene = fake.scenes[1]
    camera_rows = fake.rows("CAM_FRONT_LEFT", scene)

    rgb = index_dataset_from_path(out, metadata_scope="rgb")
    node = rgb["index"]["children"][f"scene:{scene}"]["children"]["camera:CAM_FRONT_LEFT"]
    assert [f["id"] for f in node["files"]] == [str(r["timestamp"]) for r in camera_rows]
    assert [f["path"] for f in node["files"]] == [r["filename"] for r in camera_rows]
    assert [f["attributes"]["is_key_frame"] for f in node["files"]] == [
        r["is_key_frame"] for r in camera_rows
    ]

    lidar = index_dataset_from_path(out, metadata_scope="lidar")
    node = lidar["index"]["children"][f"scene:{scene}"]["children"]["camera:CAM_FRONT_LEFT"]
    ids = [f["id"] for f in node["files"]]
    dropped = str(fake.camera_time(1, "CAM_FRONT_LEFT", DROPPED))
    late = str(fake.camera_time(1, "CAM_FRONT_LEFT", LATE))
    assert dropped not in ids
    assert len(ids) == FRAMES_PER_SCENE - 1
    late_entry = next(f for f in node["files"] if f["id"] == late)
    assert late_entry["attributes"]["offset_ms"] == 53.0
    assert all("__LIDAR_TOP__" in f["path"] for f in node["files"])


def test_calibration_file(fake, tmp_path):
    out = tmp_path / "prepared"
    prepare("nuscenes", fake.source, out)
    scene = fake.scenes[1]
    doc = json.loads((out / "calibration" / scene / "CAM_FRONT_LEFT.json").read_text())

    assert doc["format"] == {"name": "euler_known_datasets.nuscenes.camera_frames", "version": 1}
    assert doc["dataset_version"] == "v1.0-test"
    assert doc["scene"]["name"] == scene
    assert doc["camera"]["intrinsics"] == INTRINSICS
    assert (doc["camera"]["width"], doc["camera"]["height"]) == (160, 90)
    assert doc["lidar"]["nominal_offset_ms"] == 43.0
    assert len(doc["frames"]) == FRAMES_PER_SCENE
    assert doc["frames"][str(fake.camera_time(1, "CAM_FRONT_LEFT", DROPPED))]["lidar"] is None

    # A keyframe pairs with its sample's own sweep, not the spurious record
    # 1 ms after the camera; its transform is the devkit chain through both
    # ego poses.
    t0 = fake.rows("LIDAR_TOP", scene)[0]["timestamp"]
    t_cam = fake.camera_time(1, "CAM_FRONT_LEFT", 12)
    frame = doc["frames"][str(t_cam)]
    assert frame["is_key_frame"]
    assert frame["lidar"]["path"].startswith("samples/LIDAR_TOP/")
    t_lidar = frame["lidar"]["timestamp"]
    assert t_lidar - t_cam == 43_000

    def pose(t):
        return rigid_transform(*ego_pose(t, t0))

    camera_to_ego = rigid_transform(*CAMERA_POSES["CAM_FRONT_LEFT"])
    lidar_to_ego = rigid_transform(*LIDAR_POSE)
    expected = (
        np.linalg.inv(camera_to_ego) @ np.linalg.inv(pose(t_cam)) @ pose(t_lidar) @ lidar_to_ego
    )
    assert np.allclose(frame["lidar"]["lidar_to_camera"], expected, atol=1e-9)
    assert np.allclose(frame["camera_to_global"], pose(t_cam) @ camera_to_ego, atol=1e-9)
    # The ego motion between the two timestamps matters: 0.43 m at 10 m/s.
    static = np.linalg.inv(camera_to_ego) @ lidar_to_ego
    assert np.linalg.norm(expected[:3, 3] - static[:3, 3]) > 0.4


def test_splits(fake, tmp_path):
    out = tmp_path / "prepared"
    prepare("nuscenes", fake.source, out)
    cameras = len(CAMERA_OFFSET_US)
    frames = cameras * 2 * FRAMES_PER_SCENE
    assert len(get_files(load_dataset_split(out, "test", metadata_scope="rgb"))) == frames
    lidar = get_files(load_dataset_split(out, "test", metadata_scope="lidar"))
    assert len(lidar) == frames - cameras  # one frame per camera lost its sweep
    keyframes = load_dataset_split(out, "test_keyframes", metadata_scope="rgb")
    assert len(get_files(keyframes)) == cameras * 2 * KEYFRAMES_PER_SCENE
    assert all(path.startswith("samples/") for path in get_files(keyframes))


def test_symlink_media_from_extracted_source(extracted, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("nuscenes", extracted.source, out).receipt
    assert receipt["media"]["used"] == "symlink"
    for name in ("samples", "sweeps", "v1.0-test", "maps", "LICENSE"):
        assert (out / name).is_symlink()
    assert verify(out).ok


@pytest.mark.parametrize("mode", ["hardlink", "copy"])
def test_hardlink_and_copy_media(extracted, tmp_path, mode):
    out = tmp_path / "prepared"
    receipt = prepare("nuscenes", extracted.source, out, media=mode).receipt
    assert receipt["media"]["used"] == mode
    assert not (out / "samples").is_symlink()
    assert _files_below(out, "samples", "sweeps") == _needed(extracted)
    some = sorted(_needed(extracted))[0]
    same_inode = os.stat(out / some).st_ino == os.stat(extracted.source / some).st_ino
    assert same_inode == (mode == "hardlink")
    assert (out / "v1.0-test" / "scene.json").is_file()
    assert verify(out).ok


def test_symlink_media_needs_extracted_files(fake, tmp_path):
    with pytest.raises(MediaError, match="not extracted"):
        prepare("nuscenes", fake.source, tmp_path / "prepared", media="symlink")


def test_in_place(fake):
    receipt = prepare("nuscenes", fake.source, fake.source).receipt
    assert receipt["media"]["used"] == "in_place"
    assert (fake.source / ".ds_crawler" / "rgb" / "index.json").is_file()
    assert _needed(fake) <= _files_below(fake.source, "samples", "sweeps")
    assert verify(fake.source).ok


def test_rerun_skips_finished_archives(fake, tmp_path, caplog):
    out = tmp_path / "prepared"
    prepare("nuscenes", fake.source, out)
    first = (out / ".ds_crawler" / "rgb" / "index.json").read_text()
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("nuscenes", fake.source, out)
    skipped = [r for r in caplog.records if "already extracted" in r.getMessage()]
    assert len(skipped) == 2  # the meta and the blob archive
    assert (out / ".ds_crawler" / "rgb" / "index.json").read_text() == first


def test_options(fake, tmp_path):
    receipt = prepare(
        "nuscenes",
        fake.source,
        tmp_path / "a",
        cameras=["CAM_FRONT_LEFT"],
        keyframes_only=True,
    ).receipt
    cameras = receipt["summary"]["versions"]["v1.0-test"]["cameras"]
    assert list(cameras) == ["CAM_FRONT_LEFT"]
    assert cameras["CAM_FRONT_LEFT"]["frames"] == 2 * KEYFRAMES_PER_SCENE
    assert receipt["splits"]["rgb"] == ["test"]
    assert not list((tmp_path / "a" / "calibration").rglob("CAM_FRONT.json"))

    strict = prepare(
        "nuscenes", fake.source, tmp_path / "b", cameras="CAM_FRONT_LEFT", sync_tolerance_ms=5
    ).receipt
    counts = strict["summary"]["versions"]["v1.0-test"]["cameras"]["CAM_FRONT_LEFT"]
    assert counts["unsynced"] == 2  # the dropped sweep and the late-stamped one


def test_tolerance_must_stay_below_half_a_rotation(fake, tmp_path):
    with pytest.raises(ValueError, match="sync_tolerance_ms"):
        prepare("nuscenes", fake.source, tmp_path / "out", sync_tolerance_ms=30)


def test_trainval_with_split_blob_archives(tmp_path):
    fake = build_fake_nuscenes(tmp_path / "downloads", version="v1.0-trainval", blob_parts=2)
    assert [a.name for a in fake.archives] == [
        "v1.0-trainval_meta.tgz",
        "v1.0-trainval01_blobs.tgz",
        "v1.0-trainval02_blobs.tgz",
    ]
    out = tmp_path / "prepared"
    receipt = prepare("nuscenes", fake.source, out).receipt
    assert receipt["splits"]["rgb"] == ["train", "train_keyframes", "val", "val_keyframes"]
    assert _files_below(out, "samples", "sweeps") == _needed(fake)


def test_mini_and_trainval_are_not_mixed(tmp_path):
    source = tmp_path / "downloads"
    build_fake_nuscenes(source, version="v1.0-mini")
    build_fake_nuscenes(source, version="v1.0-trainval")
    with pytest.raises(ValueError, match="subset"):
        prepare("nuscenes", source, tmp_path / "out")
    receipt = prepare("nuscenes", source, tmp_path / "mini", versions=["v1.0-mini"]).receipt
    assert receipt["splits"]["rgb"] == [
        "mini_train", "mini_train_keyframes", "mini_val", "mini_val_keyframes"
    ]


def test_missing_version(fake, tmp_path):
    with pytest.raises(FileNotFoundError, match="v1.0-trainval"):
        prepare("nuscenes", fake.source, tmp_path / "out", versions=["v1.0-trainval"])


def test_refuses_a_foreign_non_empty_output(fake, tmp_path):
    out = tmp_path / "prepared"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    with pytest.raises(PrepareError, match="not empty"):
        prepare("nuscenes", fake.source, out)
    prepare("nuscenes", fake.source, out, force=True)
    assert (out / "notes.txt").read_text() == "mine"


def test_dry_run_writes_nothing(fake, tmp_path):
    result = prepare("nuscenes", fake.source, tmp_path / "out", dry_run=True)
    assert result.dry_run
    assert {item["kind"] for item in result.plan.inputs} == {"meta_archive", "blob_archive"}
    assert not (tmp_path / "out").exists()


def test_verify_reports_missing_files(fake, tmp_path):
    out = tmp_path / "prepared"
    prepare("nuscenes", fake.source, out)
    victim = sorted(p for p in _needed(fake) if "/CAM_" in p)[0]
    (out / victim).unlink()
    report = verify(out)
    assert not report.ok
    rgb = next(scope for scope in report.scopes if scope.scope == "rgb")
    assert rgb.missing == 1 and rgb.missing_examples == [victim]


def test_loads_in_euler_loading(fake, tmp_path):
    pytest.importorskip("euler_loading")
    from euler_loading import Modality, MultiModalDataset

    try:  # euler-loading's own module, from the release after 2.39.0
        from euler_loading.loaders.cpu import nuscenes as nus
    except ImportError:
        from . import reference_loaders as nus

    out = tmp_path / "prepared"
    prepare("nuscenes", fake.source, out, cameras=["CAM_FRONT_LEFT"])
    root = str(out)
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(root, metadata_scope="rgb", split="test", loader=nus.rgb),
            "lidar": Modality(
                root, metadata_scope="lidar", split="test", loader=nus.lidar_point_cloud
            ),
        },
        hierarchical_modalities={
            "intrinsics": Modality(
                root, metadata_scope="intrinsics", loader=nus.read_intrinsics, collapse_single=True
            ),
            "extrinsics": Modality(
                root,
                metadata_scope="camera_extrinsics",
                loader=nus.read_extrinsics,
                collapse_single=True,
            ),
        },
    )
    assert len(dataset) == 2 * FRAMES_PER_SCENE - 1
    sample = dataset[0]
    scene, camera, file_id = sample["full_id"].strip("/").split("/")
    assert camera == "camera:CAM_FRONT_LEFT"
    assert sample["rgb"].shape == (90, 160, 3)
    assert sample["lidar"].shape == (8, 5)
    assert np.allclose(sample["intrinsics"], INTRINSICS)
    doc = json.loads((out / "calibration" / scene.split(":")[1] / "CAM_FRONT_LEFT.json").read_text())
    assert np.allclose(sample["extrinsics"], doc["frames"][file_id]["lidar"]["lidar_to_camera"])


def test_heads_follow_the_contract_conventions(fake, tmp_path):
    """Geometry meta uses euler-dataset-contract's vocabulary (its phase-0 reference heads)."""
    out = tmp_path / "prepared"
    prepare("nuscenes", fake.source, out, cameras=["CAM_FRONT_LEFT"])

    def meta(scope):
        return json.loads((out / ".ds_crawler" / scope / "dataset-head.json").read_text())["modality"]["meta"]

    assert meta("camera_extrinsics") == {
        "source_frame": "LIDAR_TOP",
        "target_frame": "camera_optical",
        "direction": "target_from_source",
        "dimensions": {"rows": 4, "columns": 4},
    }
    assert meta("intrinsics") == {"camera_model": "pinhole", "dimensions": {"rows": 3, "columns": 3}}
    lidar = meta("lidar")
    assert lidar["representation"] == "point_cloud" and lidar["coordinate_unit"] == "meters"
    assert lidar["columns"] == ["x", "y", "z", "intensity", "ring"]
