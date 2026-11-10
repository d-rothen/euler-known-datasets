from __future__ import annotations

import logging

from euler_known_datasets.datasets.nuscenes.sources import discover_sources


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_recognises_the_official_archive_names(tmp_path):
    for name in (
        "v1.0-test_meta.tgz",
        "v1.0-test_blobs.tgz",
        "v1.0-trainval_meta.tgz",
        "v1.0-trainval01_blobs.tgz",
        "v1.0-trainval10_blobs.tgz",
        "v1.0-mini.tgz",
        "nuScenes-map-expansion-v1.3.zip",
        "can_bus.zip",
    ):
        _touch(tmp_path / name)
    found = discover_sources(tmp_path)
    assert sorted(found) == ["v1.0-mini", "v1.0-test", "v1.0-trainval"]
    assert found["v1.0-test"].meta_archive.name == "v1.0-test_meta.tgz"
    assert [a.name for a in found["v1.0-trainval"].blob_archives] == [
        "v1.0-trainval01_blobs.tgz",
        "v1.0-trainval10_blobs.tgz",
    ]
    mini = found["v1.0-mini"]
    assert mini.meta_archive == mini.blob_archives[0]


def test_recognises_extracted_tables(tmp_path):
    for table in ("sensor", "calibrated_sensor", "sample", "scene", "log", "sample_data", "ego_pose"):
        _touch(tmp_path / "v1.0-test" / f"{table}.json")
    found = discover_sources(tmp_path)
    assert found["v1.0-test"].tables_dir == tmp_path / "v1.0-test"
    assert found["v1.0-test"].has_tables


def test_ignores_incomplete_tables(tmp_path, caplog):
    _touch(tmp_path / "v1.0-test" / "sensor.json")
    with caplog.at_level(logging.WARNING):
        assert discover_sources(tmp_path) == {}
    assert "missing tables" in caplog.text
