from __future__ import annotations

import json
import logging
import os
import zipfile

import numpy as np
import pytest
from ds_crawler import index_dataset_from_path

from euler_known_datasets import MediaError, detect_recipes, prepare, verify
from euler_known_datasets.datasets import cerema  # noqa: F401  (registers the recipe)
from euler_known_datasets.datasets.cerema import SCOPES, condition_attributes
from euler_known_datasets.datasets.cerema.sources import FOLDERS, discover

from .fake_cerema import (
    FRAMES,
    HEIGHT,
    NO_GATED_LIDAR,
    P_RGB,
    WIDTH,
    build_fake_cerema,
    ground_truth,
)

BF = 0.202993 * 2355.722801
N = len(FRAMES)
FRAME_SCOPES = ["rgb", "rgb_right", "gated0", "gated17", "gated31", "lidar", "lidar_gated",
                "depth", "depth_gated", "sgm", "psmnet", "monodepth", "sparse2dense"]


@pytest.fixture
def zipped(tmp_path):
    return build_fake_cerema(tmp_path / "downloads")


@pytest.fixture
def extracted(tmp_path):
    return build_fake_cerema(tmp_path / "downloads", layout="extracted")


def _files_below(root):
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not p.relative_to(root).parts[0].startswith(".")
        and p.name not in ("euler-known-datasets.json", "downloads.json")
    }


def _head(out, scope):
    return json.loads((out / ".ds_crawler" / scope / "dataset-head.json").read_text())


def test_discover_reads_the_nested_release(zipped):
    found, notes = discover(zipped.source)
    assert set(found) == set(FOLDERS) and not notes
    rgb = found["rgb_left_8bit"]
    assert rgb.archive.name == "pixel_accurate_depth_benchmark.zip"
    assert rgb.inner == ("pixel_accurate_depth_benchmark/rgb_left_8bit.zip",)
    assert rgb.member_prefix == "rgb_left_8bit/" and len(rgb.names) == N
    assert found["calibration"].member_prefix == ""
    assert len(found["lidar_hdl64_gated"].names) == N - 1


def test_prepare_from_the_release_zip(zipped, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("cerema", zipped.source, out).receipt

    assert receipt["status"] == "complete"
    assert receipt["media"] == {"requested": "auto", "used": "extract"}
    assert receipt["scopes"] == list(SCOPES)
    files = receipt["summary"]["files"]
    assert files == {**{s: N for s in FRAME_SCOPES}, "lidar_gated": N - 1,
                     "intrinsics": 1, "gated_intrinsics": 1}
    assert receipt["summary"]["frames_without_file"] == {"lidar_gated": 1}
    assert receipt["summary"]["baselines"] == {"sgm": "SGM", "psmnet": "PSMNet",
                                               "monodepth": "Monodepth", "sparse2dense": "Sparse-to-Dense"}
    # Every file of the release except the unused tf tree, at its released path.
    expected = zipped.paths(*(f for f in FOLDERS if f != "calibration")) | {
        "calibration/calib_cam_stereo_left.json", "calibration/calib_gated_bwv.json"}
    assert _files_below(out) == expected
    report = verify(out)
    assert report.ok, report.format_text()
    assert not [w for w in report.warnings if "family" in w]


def test_ids_hierarchy_and_attributes(zipped, tmp_path):
    out = tmp_path / "prepared"
    prepare("cerema", zipped.source, out)

    index = index_dataset_from_path(out, metadata_scope="depth")["index"]
    assert sorted(index["children"]) == ["scene:1", "scene:2"]
    night = index["children"]["scene:2"]["children"]["time_of_day:night"]["children"]
    assert sorted(night) == ["weather:clear", "weather:fog40", "weather:rain15"]
    fog = night["weather:fog40"]["files"]
    assert fog == [{"path": "intermetric_rgb_left/scene2.npz", "id": "frame-0", "path_properties": {},
                    "basename_properties": {}, "attributes": {"fog_visibility_m": 40}}]
    rgb = index_dataset_from_path(out, metadata_scope="rgb")["index"]
    rain = rgb["children"]["scene:1"]["children"]["time_of_day:day"]["children"]["weather:rain15"]["files"]
    assert rain[0]["path"] == "rgb_left_8bit/scene1_day_rain15_0.png"
    assert rain[0]["attributes"] == {"rain_mm_h": 15}

    intrinsics = index_dataset_from_path(out, metadata_scope="intrinsics")["index"]
    assert "children" not in intrinsics
    assert intrinsics["files"][0]["path"] == "calibration/calib_cam_stereo_left.json"

    assert _head(out, "rgb")["dataset"]["attributes"]["gt"] is True
    sgm = _head(out, "sgm")
    assert sgm["dataset"]["id"] == "cerema"
    assert {k: sgm["dataset"]["attributes"][k] for k in ("gt", "baseline", "method")} == {
        "gt": False, "baseline": True, "method": "SGM"}
    assert sgm["addons"]["euler_loading"]["function"] == "sgm_depth"
    depth = _head(out, "depth")["modality"]["meta"]
    assert depth["benchmark"]["crop"] == {"top": 270, "right": 20, "bottom": 20, "left": 170}
    assert _head(out, "lidar")["modality"]["meta"]["sparse"] is True


def test_condition_attributes():
    assert condition_attributes("clear") == {}
    assert condition_attributes("fog100") == {"fog_visibility_m": 100}
    assert condition_attributes("rain55") == {"rain_mm_h": 55}


def test_inner_zips_on_disk(tmp_path):
    fake = build_fake_cerema(tmp_path / "downloads", layout="inner")
    found, _ = discover(fake.source)
    assert found["sgm"].inner == () and found["sgm"].archive.name == "sgm.zip"
    receipt = prepare("cerema", fake.source, tmp_path / "out", modalities=["rgb", "intrinsics"]).receipt
    assert receipt["summary"]["files"] == {"rgb": N, "intrinsics": 1}
    assert _files_below(tmp_path / "out") == fake.paths("rgb_left_8bit") | {"calibration/calib_cam_stereo_left.json"}


def test_symlink_media_from_extracted_folders(extracted, tmp_path):
    out = tmp_path / "prepared"
    receipt = prepare("cerema", extracted.source, out).receipt
    assert receipt["media"]["used"] == "symlink"
    assert (out / "rgb_left_8bit").is_symlink() and (out / "calibration").is_symlink()
    assert verify(out).ok


@pytest.mark.parametrize("mode", ["hardlink", "copy"])
def test_hardlink_and_copy_from_the_data_layout(tmp_path, mode):
    fake = build_fake_cerema(tmp_path / "downloads", layout="data")
    out = tmp_path / "prepared"
    receipt = prepare("cerema", fake.source, out, media=mode, modalities=["rgb", "depth"]).receipt
    assert receipt["media"]["used"] == mode
    assert _files_below(out) == fake.paths("rgb_left_8bit", "intermetric_rgb_left")
    some = "intermetric_rgb_left/scene1.npz"
    same = os.stat(out / some).st_ino == os.stat(fake.source / "data" / some).st_ino
    assert same == (mode == "hardlink")
    assert verify(out).ok


def test_in_place_keeps_the_data_layout(tmp_path):
    fake = build_fake_cerema(tmp_path / "downloads", layout="data")
    prepare("cerema", fake.source, fake.source, modalities=["rgb", "intrinsics"])
    rgb = index_dataset_from_path(fake.source, metadata_scope="rgb")["index"]
    files = rgb["children"]["scene:1"]["children"]["time_of_day:day"]["children"]["weather:clear"]["files"]
    assert files[0]["path"] == "data/rgb_left_8bit/scene1_day_clear_0.png"
    assert verify(fake.source).ok


def test_options_select_scopes_and_conditions(zipped, tmp_path):
    out = tmp_path / "out"
    receipt = prepare("cerema", zipped.source, out, modalities=["rgb", "depth", "intrinsics"],
                      scenes=["2"], weather=["fog", "rain15"], time_of_day=["night"]).receipt
    assert receipt["scopes"] == ["rgb", "depth", "intrinsics"]
    assert receipt["summary"]["files"] == {"rgb": 2, "depth": 2, "intrinsics": 1}
    assert receipt["summary"]["conditions"]["weather"] == ["fog40", "rain15"]
    assert _files_below(out) == {
        "rgb_left_8bit/scene2_night_fog40_0.png", "rgb_left_8bit/scene2_night_rain15_0.png",
        "intermetric_rgb_left/scene2.npz", "calibration/calib_cam_stereo_left.json"}


def test_requested_scope_must_exist(tmp_path):
    fake = build_fake_cerema(tmp_path / "downloads", layout="inner")
    (fake.source / "pixel_accurate_depth_benchmark" / "sgm.zip").unlink()
    with pytest.raises(FileNotFoundError, match="sgm"):
        prepare("cerema", fake.source, tmp_path / "out", modalities=["sgm"])
    receipt = prepare("cerema", fake.source, tmp_path / "out").receipt
    assert "sgm" not in receipt["scopes"]


def test_partial_extraction_falls_back_to_the_archive(zipped, tmp_path):
    (zipped.source / "rgb_left_8bit").mkdir()
    (zipped.source / "rgb_left_8bit" / f"{FRAMES[0]}.png").write_bytes(b"partial")
    found, notes = discover(zipped.source)
    assert found["rgb_left_8bit"].extracted is None
    assert any("rgb_left_8bit/ lacks" in note for note in notes)


def test_extract_needs_the_archives(extracted, tmp_path):
    with pytest.raises(MediaError, match="archives"):
        prepare("cerema", extracted.source, tmp_path / "prepared", media="extract")


def test_rerun_skips_finished_archives(zipped, tmp_path, caplog):
    out = tmp_path / "prepared"
    prepare("cerema", zipped.source, out, modalities=["rgb", "lidar"])
    with caplog.at_level(logging.INFO, logger="euler_known_datasets"):
        prepare("cerema", zipped.source, out, modalities=["rgb", "lidar"])
    skipped = {r.args[0] for r in caplog.records if "already extracted" in r.getMessage()}
    assert skipped == {
        "pixel_accurate_depth_benchmark.zip > pixel_accurate_depth_benchmark/rgb_left_8bit.zip",
        "pixel_accurate_depth_benchmark.zip > pixel_accurate_depth_benchmark/lidar_hdl64_rgb_left.zip",
    }


def test_no_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="no CEREMA"):
        prepare("cerema", tmp_path, tmp_path / "out")


def test_detect(zipped, tmp_path):
    assert "cerema" in [recipe.name for recipe in detect_recipes(zipped.source)]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "cerema" not in [recipe.name for recipe in detect_recipes(empty)]


def test_a_broken_zip_is_noted(zipped):
    (zipped.source / "notes.zip").write_bytes(b"not a zip")
    _, notes = discover(zipped.source)
    assert any(note.startswith("notes.zip: not a readable zip") for note in notes)
    assert zipfile.is_zipfile(zipped.source / "pixel_accurate_depth_benchmark.zip")


def _unused_writer(path, value, meta=None):  # pragma: no cover - never called
    raise AssertionError("not writing")


def test_loads_in_euler_loading(zipped, tmp_path):
    cpu = pytest.importorskip("euler_loading.loaders.cpu.cerema")
    from euler_loading import Modality, MultiModalDataset

    out = tmp_path / "prepared"
    prepare("cerema", zipped.source, out)

    def modality(scope, loader, **kwargs):
        # An explicit writer keeps euler-loading from importing its torch loaders.
        return Modality(str(out), metadata_scope=scope, loader=loader, writer=_unused_writer, **kwargs)

    dataset = MultiModalDataset(
        modalities={
            "rgb": modality("rgb", cpu.rgb),
            "gated": modality("gated0", cpu.gated),
            "lidar": modality("lidar", cpu.depth),
            "depth": modality("depth", cpu.depth),
            "psmnet": modality("psmnet", cpu.psmnet_depth),
        },
        hierarchical_modalities={
            "intrinsics": modality("intrinsics", cpu.read_intrinsics, collapse_single=True),
        },
    )
    assert len(dataset) == N
    sample = next(dataset[i] for i in range(N) if dataset[i]["full_id"] == "/scene:2/time_of_day:day/weather:fog40/frame-0")
    assert sample["rgb"].shape == (HEIGHT, WIDTH, 3)
    assert sample["gated"].shape == (HEIGHT, WIDTH) and 0 <= sample["gated"].min() <= sample["gated"].max() <= 1
    np.testing.assert_array_equal(sample["depth"], ground_truth("2"))
    assert sample["lidar"][2, 3] == 10.0 and np.count_nonzero(sample["lidar"]) == 1
    assert sample["psmnet"].shape == (1024, 1920) and np.allclose(sample["psmnet"], BF / 50)
    np.testing.assert_allclose(sample["intrinsics"], np.array(P_RGB).reshape(3, 4)[:, :3], rtol=1e-6)

    # The gated lidar of one frame is missing from the release: joining drops it.
    gated = MultiModalDataset(modalities={"rgb": modality("rgb", cpu.rgb), "lidar": modality("lidar_gated", cpu.depth)})
    assert len(gated) == N - 1
    assert all(NO_GATED_LIDAR not in gated[i]["meta"]["rgb"]["path"] for i in range(N - 1))
