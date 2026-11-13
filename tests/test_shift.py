from __future__ import annotations

import json

import numpy as np
import pytest

from euler_known_datasets import detect_recipes, prepare, verify
from euler_known_datasets.datasets import shift  # noqa: F401  (registers the recipe)
from euler_known_datasets.datasets.shift import SCOPES
from euler_known_datasets.datasets.shift.camera import camera_to_world
from euler_known_datasets.datasets.shift.labels import (
    MERGED_LABELS,
    MERGED_PAIRS,
    UNKNOWN,
    VIDEO_TRIPLES,
    labels_from_yuv,
    video_triple,
)
from euler_known_datasets.datasets.shift.selection import (
    iter_det2d,
    motion_filter,
    read_sequences,
    select,
    slug,
)
from euler_known_datasets.datasets.shift.sources import discover

from .fake_shift import FOCAL, FRAMES, HOLE, SEQUENCES, build_fake_shift, pose

TRAIN = [n for n, s in SEQUENCES.items() if s[0] == "train"]
VAL = [n for n, s in SEQUENCES.items() if s[0] == "val"]


@pytest.fixture
def zipped(tmp_path):
    return build_fake_shift(tmp_path / "downloads")


@pytest.fixture
def decoded(tmp_path):
    return build_fake_shift(tmp_path / "downloads", layout="decoded")


def _head(out, scope):
    return json.loads((out / ".ds_crawler" / scope / "dataset-head.json").read_text())


def _index(out, scope):
    return json.loads((out / ".ds_crawler" / scope / "index.json").read_text())["index"]


def _entries(node, keys=()):
    for entry in node.get("files", []):
        yield keys, entry
    for name, child in node.get("children", {}).items():
        yield from _entries(child, keys + (name,))


# -- facts and helpers ----------------------------------------------------------


def test_video_label_table():
    # Nine pairs share a triple; vehicle, sky, ground, guard rail and terrain keep their own.
    assert len(VIDEO_TRIPLES) == 14 and len(MERGED_PAIRS) == 9
    assert video_triple(13) == (19, 126, 134) and VIDEO_TRIPLES[(19, 126, 134)] == 13
    assert video_triple(6) == video_triple(7) == (18, 127, 131) and VIDEO_TRIPLES[(18, 127, 131)] == 26
    assert MERGED_LABELS[26] == "road_line|road" and MERGED_LABELS[10] == "vehicle"
    yuv = np.array([[[19, 18, 0]], [[126, 127, 0]], [[134, 131, 0]]], dtype=np.uint8)  # (3, 1, 3)
    labels, unknown = labels_from_yuv(yuv)
    assert labels.tolist() == [[13, 26, UNKNOWN]] and unknown == 1


def test_conditions_and_hierarchy(zipped):
    sequences = read_sequences(zipped.source / "discrete/images/train/front/seq.csv", "train")
    assert sequences["0bb0-2222"].weather == "cloudy" and sequences["0bb0-2222"].time_of_day == "dawn_dusk"
    assert sequences["0dd0-4444"].hierarchy == (
        "split:train", "weather:rainy", "time_of_day:night", "sequence:0dd0-4444")
    assert slug("dawn/dusk") == "dawn_dusk" and slug("slight cloudy") == "slight_cloudy"


def test_det2d_streams_without_labels(zipped):
    path = zipped.source / "discrete/images/train/front/det_2d.json"
    # A tiny chunk makes frames straddle reads.
    records = list(iter_det2d(path, chunk=257))
    assert len(records) == 500 * len(TRAIN)
    assert "labels" not in records[0] and records[0]["videoName"] == TRAIN[0]
    assert records[-1]["name"] == "00000499_img_front.jpg"


def test_motion_filter():
    poses = [(f, [0.1 * f, 0.0, 0.0], [0.0, 0.0, 0.0]) for f in range(0, 100, 10)]
    assert motion_filter(poses, 0, 0) == list(range(0, 100, 10))
    assert motion_filter(poses, 3, 0) == [0, 30, 60, 90]
    turning = [(f, [0.0, 0.0, 0.0], [0.0, 0.0, 179.0 + f / 10]) for f in range(0, 100, 10)]  # wraps at 180
    assert motion_filter(turning, 3, 5) == [0, 50]
    assert motion_filter(turning, 3, 0) == [0]


def test_selection(zipped):
    folder = zipped.source / "discrete/images/train/front"
    every = select(folder / "seq.csv", folder / "det_2d.json", "train")
    assert every.counts == {"sequences": 4, "sequences_selected": 4, "candidate_frames": 200, "frames_selected": 200}
    chosen = select(folder / "seq.csv", folder / "det_2d.json", "train", weather=["clear", "cloudy"],
                    time_of_day=["daytime", "dawn_dusk"], frame_step=40, min_motion_m=3, min_rotation_deg=5)
    assert set(chosen.sequences) == {"0aa0-1111", "0bb0-2222"}
    assert chosen.counts["candidate_frames"] == 26
    # Straight: 4 m per 40 frames, but standing still from frame 200 to 299, so 240 and 280
    # are dropped, and 320 too: only 2.1 m from frame 200.
    assert chosen.frames["0aa0-1111"] == [0, 40, 80, 120, 160, 200, 360, 400, 440, 480]
    # Turning: 10 degrees and about 7 m per 40 frames, every candidate moves enough.
    assert chosen.frames["0bb0-2222"] == list(range(0, 500, 40))
    assert set(chosen.records["0aa0-1111"]) == set(chosen.frames["0aa0-1111"])
    with pytest.raises(ValueError, match="multiple of 10"):
        select(folder / "seq.csv", folder / "det_2d.json", "train", frame_step=25)


def test_camera_to_world_convention():
    # Straight ahead along CARLA's x, level: optical z is world x, optical x (right) is
    # world -y (CARLA's y points right, the recipe's world negates it), optical y is down.
    pose_matrix = camera_to_world([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    assert np.allclose(pose_matrix[:3, :3], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    assert np.allclose(pose_matrix[:3, 3], [1.0, -2.0, 3.0])
    # Yaw +90 (turned right in CARLA's world) looks along CARLA's +y, the recipe's -y.
    assert np.allclose(camera_to_world([0, 0, 0], [0, 0, 90])[:3, 2], [0, -1, 0], atol=1e-12)
    # rotation[1] = -5: nose up 5 degrees.
    assert camera_to_world([0, 0, 0], [0, -5, 0])[2, 2] == pytest.approx(np.sin(np.radians(5)))
    rotation = camera_to_world([0, 0, 0], [2.0, -3.0, 40.0])[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3)) and np.linalg.det(rotation) == pytest.approx(1)


# -- discovery --------------------------------------------------------------------


def test_discover_zips(zipped):
    found = discover(zipped.source)
    assert set(found.splits) == {"train", "val"} and not found.notes
    train = found.splits["train"]
    assert train.groups["img"].ext == "jpg" and train.groups["img"].archive.name == "img.zip"
    assert train.groups["img"].extracted is None and train.groups["img"].member_prefix == ""
    assert len(train.groups["img"].files) == len(TRAIN) * len(FRAMES)
    assert HOLE not in train.groups["depth"].files and len(train.groups["depth"].files) == len(TRAIN) * len(FRAMES) - 1
    assert train.det2d == "discrete/images/train/front/det_2d.json" and train.decoded is None
    assert detect_recipes(zipped.source)[0].name == "shift"


def test_discover_decoded_frames(decoded):
    train = discover(decoded.source).splits["train"]
    assert train.groups["img"].ext == "png" and train.groups["img"].extracted == "discrete/images/train/front/img"
    assert train.det2d == "discrete/videos/train/front/det_2d.json"
    assert train.seq_csv == "discrete/videos/train/front/seq.csv"
    assert train.decoded["tool"] == "slurm/tools/shift_decode.py"


# -- preparing ----------------------------------------------------------------------


def test_prepare_from_the_zips(zipped, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("shift", zipped.source, out).receipt
    assert receipt["status"] == "complete"
    assert receipt["media"] == {"requested": "auto", "used": "extract"}
    assert receipt["scopes"] == list(SCOPES)
    assert receipt["splits"] == {s: ["train", "val"] for s in ("rgb", "depth", "semantic_segmentation")}
    train = receipt["summary"]["splits"]["train"]
    # The hole frame lacks depth and is left out of every scope.
    assert train["frames"] == len(TRAIN) * len(FRAMES) - 1
    assert train["frames_without_file"] == {"depth": {"count": 1, "frames": ["0aa0-1111/00000120"]}}
    assert receipt["summary"]["files"] == {
        "rgb": len(SEQUENCES) * len(FRAMES) - 1, "depth": len(SEQUENCES) * len(FRAMES) - 1,
        "semantic_segmentation": len(SEQUENCES) * len(FRAMES) - 1,
        "intrinsics": len(SEQUENCES), "camera_pose": len(SEQUENCES)}
    assert verify(out).ok, verify(out).format_text()

    entries = list(_entries(_index(out, "rgb")))
    keys, entry = entries[0]
    assert keys == ("split:train", "weather:clear", "time_of_day:daytime", "sequence:0aa0-1111")
    assert entry["id"] == "00000000"
    assert entry["path"] == "discrete/images/train/front/img/0aa0-1111/00000000_img_front.jpg"
    assert entry["attributes"]["weather_fine"] == "clear" and entry["attributes"]["fog_density"] == 1.0
    assert (out / entry["path"]).is_file() and not (out / "discrete/videos").exists()

    head = _head(out, "semantic_segmentation")
    assert head["dataset"]["id"] == "shift" and head["dataset"]["attributes"]["decoded_from_videos"] is False
    meta = head["modality"]["meta"]
    assert meta["skyclass"] == [13, 0, 0] and meta["label_ids"] == "carla_0.9.10" and meta["classes"][13] == "sky"
    depth_meta = _head(out, "depth")["modality"]["meta"]
    assert depth_meta["radial_depth"] is False and depth_meta["range"] == [0, 1000.0]
    assert _head(out, "rgb")["modality"]["meta"]["stored"] == "JPEG as released"

    calibration = json.loads((out / "calibration/train/0bb0-2222.json").read_text())
    assert calibration["camera"]["intrinsics"][0] == [FOCAL, 0.0, 32.0]
    assert len(calibration["frames"]) == len(FRAMES)
    assert calibration["frames"]["00000040"]["source"]["rotation"] == pose("turning", 40)[1]


def test_prepare_decoded_frames_with_the_selection(decoded, tmp_path):
    out = tmp_path / "prepared"
    result = prepare("shift", decoded.source, out, media="hardlink", splits=["train"],
                     weather=["clear", "cloudy"], time_of_day=["daytime", "dawn_dusk"],
                     frame_step=40, min_motion_m=3, min_rotation_deg=5)
    receipt = result.receipt
    assert receipt["media"]["used"] == "hardlink"
    train = receipt["summary"]["splits"]["train"]
    # 10 + 13 frames are selected; frame 120 of the first is the hole and has no depth.
    assert train["sequences_selected"] == 2 and train["frames_selected"] == 23
    assert train["frames"] == 22 and train["frames_without_file"]["depth"]["count"] == 1
    assert train["frames_by_condition"] == {"clear/daytime": 9, "cloudy/dawn_dusk": 13}
    assert set(receipt["splits"]["rgb"]) == {"train"}
    head = _head(out, "semantic_segmentation")
    assert head["modality"]["meta"]["label_ids"] == "shift_video_merged"
    assert head["modality"]["meta"]["classes"]["13"] == "sky"
    assert _head(out, "rgb")["modality"]["meta"]["stored"].startswith("PNG")
    assert head["dataset"]["attributes"]["selection"]["frame_step"] == 40
    entry = next(e for _, e in _entries(_index(out, "rgb")))
    assert entry["path"].endswith(".png")
    assert (out / entry["path"]).stat().st_ino == (decoded.source / entry["path"]).stat().st_ino
    assert verify(out).ok


def test_symlink_and_rerun(decoded, tmp_path):
    out = tmp_path / "prepared"
    prepare("shift", decoded.source, out, media="symlink", modalities=["rgb", "intrinsics"])
    assert (out / "discrete").is_symlink()
    again = prepare("shift", decoded.source, out, media="symlink", modalities=["rgb", "intrinsics"]).receipt
    assert again["status"] == "complete" and again["scopes"] == ["rgb", "intrinsics"]
    # rgb alone keeps the hole frame: it has its image.
    assert again["summary"]["files"]["rgb"] == len(SEQUENCES) * len(FRAMES)


def test_errors(zipped, tmp_path):
    with pytest.raises(ValueError, match="multiple of 10"):
        prepare("shift", zipped.source, tmp_path / "a", frame_step=15)
    with pytest.raises(ValueError, match="not one of"):
        prepare("shift", zipped.source, tmp_path / "b", weather=["sunny"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        prepare("shift", tmp_path / "empty", tmp_path / "c")


# -- loading ----------------------------------------------------------------------


def test_load_and_reproject_with_euler_loading(zipped, tmp_path):
    """Depth of one frame, back-projected with K and moved with the two frames' poses,
    lands on the other frame's depth: the pose convention holds through a 10 degree turn."""
    cpu = pytest.importorskip("euler_loading.loaders.cpu.shift")
    from euler_loading import Modality, MultiModalDataset

    out = tmp_path / "prepared"
    prepare("shift", zipped.source, out, splits=["train"], weather=["cloudy"])

    def modality(scope, loader, **kwargs):
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=lambda *a, **k: None, **kwargs)

    dataset = MultiModalDataset(
        modalities={"rgb": modality("rgb", cpu.rgb), "depth": modality("depth", cpu.depth),
                    "sky": modality("semantic_segmentation", cpu.sky_mask)},
        hierarchical_modalities={"intrinsics": modality("intrinsics", cpu.read_intrinsics),
                                 "pose": modality("camera_pose", cpu.camera_pose)},
    )
    assert len(dataset) == len(FRAMES)
    samples = {s["id"]: s for s in (dataset[i] for i in range(len(dataset)))}
    a, b = samples["00000040"], samples["00000080"]
    assert a["full_id"] == "/split:train/weather:cloudy/time_of_day:dawn_dusk/sequence:0bb0-2222/00000040"
    K = np.asarray(next(iter(a["intrinsics"].values())), dtype=np.float64)
    pose_a = np.asarray(next(iter(a["pose"].values())), dtype=np.float64)
    pose_b = np.asarray(next(iter(b["pose"].values())), dtype=np.float64)
    assert (a["sky"] == (a["depth"] > 999.9)).all()

    v, u = np.mgrid[0:40, 0:64]
    d = a["depth"].astype(np.float64)
    keep = d < 999
    points = (np.stack([u[keep], v[keep], np.ones(keep.sum())], 1) @ np.linalg.inv(K).T) * d[keep][:, None]
    to_b = np.linalg.inv(pose_b) @ pose_a
    q = points @ to_b[:3, :3].T + to_b[:3, 3]
    p = q @ K.T
    ub, vb = p[:, 0] / q[:, 2], p[:, 1] / q[:, 2]
    seen = (q[:, 2] > 0.5) & (ub > 0) & (ub < 63) & (vb > 0) & (vb < 39)
    # Bilinear lookup: the fake's grid is coarse.
    x0, y0 = np.floor(ub[seen]).astype(int), np.floor(vb[seen]).astype(int)
    fx, fy = ub[seen] - x0, vb[seen] - y0
    db = b["depth"].astype(np.float64)
    measured = (db[y0, x0] * (1 - fx) * (1 - fy) + db[y0, x0 + 1] * fx * (1 - fy)
                + db[y0 + 1, x0] * (1 - fx) * fy + db[y0 + 1, x0 + 1] * fx * fy)
    finite = measured < 999
    rel = np.abs(q[seen][:, 2][finite] - measured[finite]) / measured[finite]
    assert finite.sum() > 500
    assert np.median(rel) < 0.01
