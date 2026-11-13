"""The helpers recipes build on: regex scopes, media resolution, file listing."""

from __future__ import annotations

import json

import pytest
from ds_crawler import index_dataset_from_path

from euler_known_datasets.artifacts import IndexTree, file_entry, write_regex_scope, write_scope
from euler_known_datasets.media import (
    MediaError,
    list_files,
    place_extracted,
    resolve_media,
)
from euler_known_datasets.recipe import PrepareContext


def _ctx(tmp_path, media="auto", in_place=False):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    output = source if in_place else tmp_path / "out"
    return PrepareContext(
        source=source, output=output, options={}, media=media, state_dir=output / ".state"
    )


def test_regex_scope_indexes_paths_that_need_not_exist(tmp_path):
    output = write_regex_scope(
        tmp_path,
        "rgb",
        files=["scene_a/0001.png", "scene_a/0002.png", "scene_b/0001.png", "notes.txt"],
        dataset={"id": "demo_rgb", "name": "Demo RGB"},
        modality={"key": "rgb", "meta": {"range": [0, 255]}},
        indexing={
            "id": {"regex": r"^[^/]+/(?P<frame>\d+)\.png$", "join_char": "+"},
            "hierarchy": {"regex": r"^(?P<scene>[^/]+)/", "separator": ":"},
        },
        execution={"recipe": "demo"},
    )
    assert sorted(output["index"]["children"]) == ["scene:scene_a", "scene:scene_b"]
    config = json.loads((tmp_path / ".ds_crawler/rgb/ds-crawler.json").read_text())
    assert "prebuilt_index_file" not in config["source"]
    artifact = json.loads((tmp_path / ".ds_crawler/rgb/index.json").read_text())
    assert artifact["generator"]["name"] == "euler-known-datasets"
    assert artifact["execution"] == {"recipe": "demo"}
    loaded = index_dataset_from_path(tmp_path, metadata_scope="rgb")
    files = loaded["index"]["children"]["scene:scene_a"]["files"]
    assert [f["id"] for f in files] == ["frame-0001", "frame-0002"]


def test_prebuilt_scope_declares_its_hierarchy_separator(tmp_path):
    """Hierarchy keys are name:value pairs that consumers can split, e.g. to
    select a scene or to name output folders by value."""
    tree = IndexTree()
    tree.add(("scene:a", "camera:cam_00"), file_entry("a/0000.png", "0000"))
    tree.add(("scene:b", "camera:cam_00"), file_entry("b/0000.png", "0000"))
    head = {"dataset": {"id": "demo", "name": "demo"}, "modality": {"key": "rgb", "meta": {"range": [0, 255]}}}
    write_scope(tmp_path, "rgb", tree=tree, **head)
    config = json.loads((tmp_path / ".ds_crawler/rgb/ds-crawler.json").read_text())
    assert config["indexing"] == {"hierarchy": {"separator": ":"}}
    assert config["source"]["prebuilt_index_file"]
    loaded = index_dataset_from_path(tmp_path, metadata_scope="rgb")
    assert loaded["indexing"]["hierarchy"]["separator"] == ":"
    assert sorted(loaded["index"]["children"]) == ["scene:a", "scene:b"]

    bad = IndexTree()
    bad.add(("a",), file_entry("a/0000.png", "0000"))
    with pytest.raises(ValueError, match="separator"):
        write_scope(tmp_path / "bad", "rgb", tree=bad, **head)
    write_scope(tmp_path / "plain", "rgb", tree=bad, hierarchy_separator=None, **head)
    config = json.loads((tmp_path / "plain/.ds_crawler/rgb/ds-crawler.json").read_text())
    assert "hierarchy" not in config["indexing"]


def test_regex_scope_rejects_duplicate_ids(tmp_path):
    with pytest.raises(Exception, match="(?i)duplicate"):
        write_regex_scope(
            tmp_path,
            "rgb",
            files=["a/0001.png", "b/0001.png"],
            dataset={"id": "demo", "name": "Demo"},
            modality={"key": "rgb", "meta": {"range": [0, 255]}},
            indexing={"id": {"regex": r"^[^/]+/(?P<frame>\d+)\.png$"}},
        )


def test_resolve_media(tmp_path):
    assert resolve_media(_ctx(tmp_path), extracted=True, archives=False) == "symlink"
    assert resolve_media(_ctx(tmp_path), extracted=False, archives=True) == "extract"
    assert resolve_media(_ctx(tmp_path, in_place=True), extracted=False, archives=True) == "in_place"
    with pytest.raises(MediaError):
        resolve_media(_ctx(tmp_path, media="copy"), extracted=False, archives=True)
    with pytest.raises(MediaError):
        resolve_media(_ctx(tmp_path, media="extract"), extracted=True, archives=False)


@pytest.mark.parametrize("mode", ["symlink", "hardlink", "copy"])
def test_place_extracted(tmp_path, mode):
    ctx = _ctx(tmp_path, media=mode)
    (ctx.source / "rgb").mkdir()
    (ctx.source / "rgb" / "a.png").write_bytes(b"a")
    place_extracted(ctx, ["rgb/a.png"], mode=mode, top_level=["rgb"])
    assert (ctx.output / "rgb" / "a.png").read_bytes() == b"a"
    assert (ctx.output / "rgb").is_symlink() == (mode == "symlink")
    with pytest.raises(MediaError):
        place_extracted(ctx, ["rgb/missing.png"], mode=mode, top_level=["rgb"])


def test_list_files_skips_hidden_and_follows_links(tmp_path):
    (tmp_path / "data" / "x").mkdir(parents=True)
    (tmp_path / "data" / "x" / "a.png").write_bytes(b"")
    (tmp_path / "data" / ".ds_crawler").mkdir()
    (tmp_path / "data" / ".ds_crawler" / "index.json").write_text("{}")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "linked").symlink_to(tmp_path / "data", target_is_directory=True)
    assert list_files(tmp_path / "root") == ["linked/x/a.png"]
