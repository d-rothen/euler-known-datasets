from __future__ import annotations

import json
import logging
import os
import zipfile

import numpy as np
import pytest
from ds_crawler import index_dataset_from_path

from euler_known_datasets import MediaError, detect_recipes, prepare, verify
from euler_known_datasets.datasets import scatternerf  # noqa: F401  (registers the recipe)
from euler_known_datasets.datasets.scatternerf import BASELINE_M, stereo_scale
from euler_known_datasets.datasets.scatternerf.sources import discover

from .fake_scatternerf import (
    HEIGHT,
    SEQUENCES,
    WIDTH,
    K,
    build_fake_scatternerf,
    pose,
    scale_to_meters,
)

SCOPES = ["rgb", "intrinsics", "camera_pose"]


@pytest.fixture
def zipped(tmp_path):
    return build_fake_scatternerf(tmp_path / "downloads")


@pytest.fixture
def extracted(tmp_path):
    return build_fake_scatternerf(tmp_path / "downloads", archives=False, extracted=True)


def _indexed(fake, cameras=("left", "right"), sequences=tuple(SEQUENCES)):
    """Every file a prepared output references."""
    return {
        f"{sequence}/{kind}/{fake.stem(sequence, camera, frame)}.{ext}"
        for sequence in sequences
        for camera in cameras
        for frame in fake.frames(sequence, camera)
        for kind, ext in (("rgb", "png"), ("intrinsics", "txt"), ("pose", "txt"))
    }


def _files_below(root):
    return {
        p.relative_to(root).as_posix()
        for sequence in SEQUENCES
        for p in (root / sequence).rglob("*")
        if p.is_file()
    }


def test_discover_follows_nested_archives(zipped):
    found, notes = discover(zipped.source)
    assert sorted(found, key=int) == ["0", "6", "13"]
    # The least nested copy is read; the others are recorded as duplicates.
    assert (found["0"].archive.name, found["0"].chain) == ("sequence_0.zip", ())
    assert sorted(found["0"].archives) == ["sequence_0.zip", "sequence_4.zip"]
    assert (found["6"].archive.name, found["6"].chain) == ("sequence_6.zip", ())
    assert (found["13"].archive.name, found["13"].chain) == ("sequence_14.zip", ("sequence_13.zip",))
    assert notes == [
        "sequence_4.zip holds sequence 0 inside squence_0.zip",
        "sequence_9.zip holds sequence 6 inside sequence_8.zip > sequence_6.zip",
        "sequence_14.zip holds sequence 13 inside sequence_13.zip",
    ]


def test_conflicting_copies_raise(zipped, tmp_path):
    with zipfile.ZipFile(zipped.source / "sequence_0.zip") as src, \
            zipfile.ZipFile(zipped.source / "sequence_5.zip", "w") as dst:
        for info in src.infolist():
            changed = info.filename.endswith(".png")
            dst.writestr(info, b"another image" if changed else src.read(info))
    with pytest.raises(ValueError, match="different content for sequence 0"):
        prepare("scatternerf", zipped.source, tmp_path / "prepared")


def test_prepare_from_archives(zipped, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("scatternerf", zipped.source, out).receipt

    assert receipt["status"] == "complete"
    assert receipt["media"] == {"requested": "auto", "used": "extract"}
    assert receipt["scopes"] == SCOPES
    assert receipt["splits"] == {}
    inputs = {item["sequence"]: item for item in receipt["source"]["inputs"]}
    assert inputs["0"]["duplicate_archives"] == ["sequence_4.zip"]
    assert inputs["13"]["inside"] == "sequence_13.zip"

    summary = receipt["summary"]
    assert summary["lidar"] == "not part of the ScatterNeRF release"
    assert len(summary["notes"]) == 3
    for sequence in SEQUENCES:
        stats = summary["sequences"][sequence]
        assert stats["frames"] == {c: len(zipped.frames(sequence, c)) for c in ("left", "right")}
        assert stats["scale_to_meters"] == scale_to_meters(sequence)
        assert stats["baseline_along_x"] == pytest.approx(1.0)
        assert stats["baseline_spread"] == pytest.approx(0.0, abs=1e-4)
    assert summary["sequences"]["6"]["incomplete_frames"] == 1
    assert summary["sequences"]["0"]["stereo_pairs"] == 2
    assert summary["sequences"]["0"]["recorded_utc"] == "2018-10-08 07:02:15"

    # Only indexed files are extracted; sequence 13 is read inside sequence_14.zip.
    assert _files_below(out) == _indexed(zipped)
    assert not list((out / ".euler-known-datasets").rglob("*.zip"))
    report = verify(out)
    assert report.ok, report.format_text()
    assert not [w for w in report.warnings if "family" in w]


def test_ids_hierarchy_and_attributes(zipped, tmp_path):
    out = tmp_path / "prepared"
    prepare("scatternerf", zipped.source, out)

    for scope in SCOPES:
        index = index_dataset_from_path(out, metadata_scope=scope)["index"]
        assert sorted(index["children"]) == ["sequence:0", "sequence:13", "sequence:6"]
        assert sorted(index["children"]["sequence:6"]["children"]) == ["camera:left", "camera:right"]

    poses = index_dataset_from_path(out, metadata_scope="camera_pose")["index"]
    right = poses["children"]["sequence:6"]["children"]["camera:right"]["files"]
    assert [f["id"] for f in right] == ["frame-00015", "frame-00017"]
    stem = zipped.stem("6", "right", "00015")
    assert right[0]["path"] == f"6/pose/{stem}.txt"
    assert right[0]["attributes"] == {
        "timestamp_ns": int(stem.rsplit("_", 1)[1]),
        "scale_to_meters": scale_to_meters("6"),
    }
    left = poses["children"]["sequence:6"]["children"]["camera:left"]["files"]
    assert [f["id"] for f in left] == ["frame-00015"]  # frame 00016 has no pose

    heads = {
        scope: json.loads((out / ".ds_crawler" / scope / "dataset-head.json").read_text())
        for scope in SCOPES
    }
    assert {head["dataset"]["id"] for head in heads.values()} == {"scatternerf"}
    assert heads["camera_pose"]["modality"]["key"] == "camera_extrinsics"
    assert heads["camera_pose"]["modality"]["meta"]["direction"] == "target_from_source"
    assert heads["intrinsics"]["modality"]["meta"]["camera_model"] == "pinhole"
    functions = {scope: head["addons"]["euler_loading"]["function"] for scope, head in heads.items()}
    assert functions == {"rgb": "rgb", "intrinsics": "read_intrinsics", "camera_pose": "camera_pose"}


def test_symlink_media_from_extracted_source(extracted, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("scatternerf", extracted.source, out).receipt
    assert receipt["media"]["used"] == "symlink"
    assert all((out / sequence).is_symlink() for sequence in SEQUENCES)
    assert receipt["summary"]["sequences"]["13"]["scale_to_meters"] == scale_to_meters("13")
    assert verify(out).ok


@pytest.mark.parametrize("mode", ["hardlink", "copy"])
def test_hardlink_and_copy_one_camera(extracted, tmp_path, mode):
    out = tmp_path / "prepared"
    receipt = prepare("scatternerf", extracted.source, out, media=mode, cameras=["left"]).receipt
    assert receipt["media"]["used"] == mode
    assert _files_below(out) == _indexed(extracted, cameras=("left",))
    # The scale still comes from the stereo pairs, read from the source.
    assert receipt["summary"]["sequences"]["0"]["scale_to_meters"] == scale_to_meters("0")
    some = f"0/pose/{extracted.stem('0', 'left', '06880')}.txt"
    same = os.stat(out / some).st_ino == os.stat(extracted.source / some).st_ino
    assert same == (mode == "hardlink")
    assert verify(out).ok


def test_one_camera_from_archives(zipped, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("scatternerf", zipped.source, out, sequences=["13", "6"], cameras=["right"]).receipt
    assert receipt["options"] == {"sequences": ["13", "6"], "cameras": ["right"]}
    assert list(receipt["summary"]["sequences"]) == ["6", "13"]
    assert receipt["summary"]["sequences"]["13"]["frames"] == {"right": 2}
    assert receipt["summary"]["sequences"]["13"]["scale_to_meters"] == scale_to_meters("13")
    assert _files_below(out) == _indexed(zipped, cameras=("right",), sequences=("6", "13"))


def test_extract_needs_the_archives(extracted, tmp_path):
    with pytest.raises(MediaError, match="archives"):
        prepare("scatternerf", extracted.source, tmp_path / "prepared", media="extract")


def test_in_place(zipped):
    receipt = prepare("scatternerf", zipped.source, zipped.source).receipt
    assert receipt["media"]["used"] == "in_place"
    assert _indexed(zipped) <= _files_below(zipped.source)
    assert verify(zipped.source).ok


def test_rerun_skips_finished_archives(zipped, tmp_path, caplog):
    out = tmp_path / "prepared"
    prepare("scatternerf", zipped.source, out)
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("scatternerf", zipped.source, out)
    skipped = {r.args[0] for r in caplog.records if "already extracted" in r.getMessage()}
    assert skipped == {"sequence_0.zip", "sequence_6.zip", "sequence_14.zip > sequence_13.zip"}
    assert verify(out).ok


def test_requested_sequence_must_exist(zipped, tmp_path):
    with pytest.raises(FileNotFoundError, match=r"sequence\(s\) 4 are not in"):
        prepare("scatternerf", zipped.source, tmp_path / "out", sequences=["4"])
    assert not (tmp_path / "out").exists()


def test_no_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="no ScatterNeRF downloads"):
        prepare("scatternerf", tmp_path, tmp_path / "out")


def test_detect(zipped, tmp_path):
    assert "scatternerf" in [recipe.name for recipe in detect_recipes(zipped.source)]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "scatternerf" not in [recipe.name for recipe in detect_recipes(empty)]


def test_stereo_scale():
    poses = {("left", "1", "10"): pose("6", "left", 3), ("right", "1", "10"): pose("6", "right", 3),
             ("left", "2", "20"): pose("6", "left", 4)}
    scale = stereo_scale(poses)
    assert scale["stereo_pairs"] == 1
    assert scale["baseline_units"] == pytest.approx(SEQUENCES["6"][0])
    assert scale["scale_to_meters"] == pytest.approx(BASELINE_M / SEQUENCES["6"][0], rel=1e-6)
    assert stereo_scale({("left", "1", "10"): np.eye(4)}) == {"stereo_pairs": 0}


def _unused_writer(path, value, meta=None):  # pragma: no cover - never called
    raise AssertionError("not writing")


def test_loads_in_euler_loading(zipped, tmp_path):
    cpu = pytest.importorskip("euler_loading.loaders.cpu.scatternerf")
    from euler_loading import Modality, MultiModalDataset

    out = tmp_path / "prepared"
    prepare("scatternerf", zipped.source, out)

    def modality(scope, loader):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=_unused_writer)

    dataset = MultiModalDataset(
        modalities={
            "rgb": modality("rgb", cpu.rgb),
            "intrinsics": modality("intrinsics", cpu.read_intrinsics),
            "pose": modality("camera_pose", cpu.camera_pose_metric),
        }
    )
    assert len(dataset) == sum(len(zipped.frames(s, c)) for s in SEQUENCES for c in ("left", "right"))
    samples = {dataset[i]["full_id"]: dataset[i] for i in range(len(dataset))}
    left = samples["/sequence:0/camera:left/frame-06880"]
    right = samples["/sequence:0/camera:right/frame-06880"]
    assert left["rgb"].shape == (HEIGHT, WIDTH, 3)
    np.testing.assert_allclose(left["intrinsics"], K)
    baseline = (np.linalg.inv(left["pose"]) @ right["pose"])[:3, 3]
    np.testing.assert_allclose(baseline, (BASELINE_M, 0, 0), atol=1e-5)
