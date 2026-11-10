from __future__ import annotations

import json

from euler_known_datasets.cli import main

from .fake_nuscenes import build_fake_nuscenes


def test_list_and_info(capsys):
    assert main(["list"]) == 0
    assert "nuscenes" in capsys.readouterr().out
    assert main(["info", "nuscenes"]) == 0
    out = capsys.readouterr().out
    assert "--sync-tolerance-ms" in out and "v1.0-test_blobs.tgz" in out


def test_detect(tmp_path, capsys):
    assert main(["detect", str(tmp_path)]) == 1
    build_fake_nuscenes(tmp_path)
    assert main(["detect", str(tmp_path)]) == 0
    assert capsys.readouterr().out.split() == ["nuscenes"]


def test_prepare_dry_run_then_prepare_and_verify(tmp_path, capsys):
    fake = build_fake_nuscenes(tmp_path / "downloads")
    out = tmp_path / "prepared"
    args = ["prepare", "nuscenes", str(fake.source), str(out), "--cameras", "CAM_FRONT_LEFT"]
    assert main(args + ["--dry-run"]) == 0
    assert "steps" in capsys.readouterr().out
    assert not out.exists()

    assert main(args + ["--keyframes-only", "--sync-tolerance-ms", "10"]) == 0
    capsys.readouterr()
    receipt = json.loads((out / "euler-known-datasets.json").read_text())
    assert receipt["options"] == {
        "versions": [],
        "cameras": ["CAM_FRONT_LEFT"],
        "keyframes_only": True,
        "sync_tolerance_ms": 10.0,
    }

    assert main(["verify", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_prepare_reports_errors(tmp_path, capsys):
    assert main(["prepare", "nuscenes", str(tmp_path), str(tmp_path / "out")]) == 2
    assert "no nuScenes downloads" in capsys.readouterr().err
