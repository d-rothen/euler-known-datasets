from __future__ import annotations

import io
import tarfile

import pytest

from euler_known_datasets.archives import (
    ExtractionLedger,
    UnsafeArchiveError,
    extract_archive,
    member_relpath,
    selection_fingerprint,
)


def _tar(path, members, *, links=()):
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        for name, target in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return path


def test_extracts_selected_members_only(tmp_path):
    archive = _tar(tmp_path / "a.tgz", {"keep/a.txt": b"a", "keep/b.txt": b"bb", "drop/c.txt": b"c"})
    out = tmp_path / "out"
    stats = extract_archive(archive, out, select=lambda name: name.startswith("keep/"))
    assert (out / "keep/a.txt").read_bytes() == b"a"
    assert (out / "keep/b.txt").read_bytes() == b"bb"
    assert not (out / "drop").exists()
    assert (stats.extracted, stats.skipped_unselected) == (2, 1)
    assert not list(out.rglob(".*.part"))


def test_rerun_skips_present_files_and_rewrites_truncated_ones(tmp_path):
    archive = _tar(tmp_path / "a.tgz", {"x/a.txt": b"aaaa", "x/b.txt": b"bbbb"})
    out = tmp_path / "out"
    extract_archive(archive, out)
    (out / "x/b.txt").write_bytes(b"b")  # an interrupted write
    stats = extract_archive(archive, out)
    assert (stats.skipped_existing, stats.extracted) == (1, 1)
    assert (out / "x/b.txt").read_bytes() == b"bbbb"


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/abs/evil.txt"])
def test_rejects_members_escaping_the_destination(tmp_path, name):
    archive = _tar(tmp_path / "a.tgz", {name: b"x"})
    with pytest.raises(UnsafeArchiveError):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_skips_links(tmp_path):
    archive = _tar(tmp_path / "a.tgz", {"a.txt": b"a"}, links=[("link", "/etc/passwd")])
    stats = extract_archive(archive, tmp_path / "out")
    assert stats.skipped_special == 1
    assert not (tmp_path / "out/link").exists()


def test_refuses_to_write_through_a_symlinked_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "samples").symlink_to(outside, target_is_directory=True)
    archive = _tar(tmp_path / "a.tgz", {"samples/a.jpg": b"a"})
    with pytest.raises(UnsafeArchiveError, match="symlink"):
        extract_archive(archive, out)
    assert not (outside / "a.jpg").exists()


def test_member_relpath_normalises():
    assert member_relpath("./a/b.txt").as_posix() == "a/b.txt"
    assert member_relpath("./") is None


def test_ledger_matches_archive_destination_and_selection(tmp_path):
    archive = _tar(tmp_path / "a.tgz", {"a.txt": b"a"})
    ledger = ExtractionLedger(tmp_path / "state")
    out = tmp_path / "out"
    fingerprint = selection_fingerprint(["a.txt"])
    assert not ledger.is_complete(archive, out, fingerprint)
    stats = extract_archive(archive, out)
    ledger.mark_complete(archive, out, fingerprint, stats)
    assert ledger.is_complete(archive, out, fingerprint)
    assert not ledger.is_complete(archive, out, selection_fingerprint(["b.txt"]))
    assert not ledger.is_complete(archive, tmp_path / "elsewhere", fingerprint)


def test_selection_fingerprint_ignores_order():
    assert selection_fingerprint(["b", "a"]) == selection_fingerprint(["a", "b"])


def _zip(path, members, *, symlinks=()):
    import stat
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
        for name, target in symlinks:
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, target)
    return path


def test_zip_extraction_is_selective_and_resumable(tmp_path):
    archive = _zip(tmp_path / "a.zip", {"keep/a.txt": b"a", "drop/b.txt": b"b", "keep/dir/": b""})
    out = tmp_path / "out"
    stats = extract_archive(archive, out, select=lambda name: name.startswith("keep/"))
    assert (out / "keep/a.txt").read_bytes() == b"a"
    assert not (out / "drop").exists()
    assert (stats.extracted, stats.skipped_unselected) == (1, 1)
    assert extract_archive(archive, out).skipped_existing == 1


def test_zip_symlinks_are_not_extracted(tmp_path):
    archive = _zip(tmp_path / "a.zip", {"a.txt": b"a"}, symlinks=[("link", "/etc/passwd")])
    stats = extract_archive(archive, tmp_path / "out")
    assert stats.skipped_special == 1
    assert not (tmp_path / "out/link").exists()


def test_zip_traversal_is_rejected(tmp_path):
    archive = _zip(tmp_path / "a.zip", {"../evil.txt": b"x"})
    with pytest.raises(UnsafeArchiveError):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_archive_members(tmp_path):
    from euler_known_datasets.archives import archive_members

    zipped = _zip(tmp_path / "a.zip", {"x/a.txt": b"a", "x/": b""}, symlinks=[("l", "a")])
    assert archive_members(zipped) == ["x/a.txt"]
    tarred = _tar(tmp_path / "a.tgz", {"./x/a.txt": b"a"}, links=[("l", "a")])
    assert archive_members(tarred) == ["x/a.txt"]


def _zip_bytes(members, compression):
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buffer.getvalue()


@pytest.mark.parametrize("compression", ["stored", "deflated"])
def test_nested_zips_are_read_in_place(tmp_path, compression):
    import zipfile

    from euler_known_datasets.archives import archive_members, open_nested_zip

    method = {"stored": zipfile.ZIP_STORED, "deflated": zipfile.ZIP_DEFLATED}[compression]
    payload = {"data/a.txt": b"a" * 5000, "data/b.txt": b"bb", "other/c.txt": b"c"}
    middle = _zip_bytes({"README": b"x", "pack/inner.zip": _zip_bytes(payload, zipfile.ZIP_DEFLATED)}, method)
    outer = tmp_path / "outer.zip"
    outer.write_bytes(_zip_bytes({"top/middle.zip": middle, "notes.txt": b"n"}, method))
    inner = ("top/middle.zip", "pack/inner.zip")

    assert archive_members(outer, inner) == sorted(payload)
    with open_nested_zip(outer, inner[:1]) as zf:
        assert sorted(zf.namelist()) == ["README", "pack/inner.zip"]
    out = tmp_path / "out"
    stats = extract_archive(outer, out, select=lambda name: name.startswith("data/"), inner=inner)
    assert stats.archive == "outer.zip > top/middle.zip > pack/inner.zip"
    assert (stats.extracted, stats.skipped_unselected) == (2, 1)
    assert (out / "data/a.txt").read_bytes() == b"a" * 5000
    assert extract_archive(outer, out, inner=inner).skipped_existing == 2


def test_nested_zip_corruption_is_detected(tmp_path):
    import zipfile

    from euler_known_datasets.archives import _member_data_offset

    inner = _zip_bytes({"a.txt": b"hello world" * 100}, zipfile.ZIP_STORED)
    outer = tmp_path / "outer.zip"
    outer.write_bytes(_zip_bytes({"inner.zip": inner}, zipfile.ZIP_STORED))
    with zipfile.ZipFile(outer) as zf, open(outer, "rb") as stream:
        start = _member_data_offset(stream, zf.getinfo("inner.zip"))
    data = bytearray(outer.read_bytes())
    position = start + inner.index(b"hello world") + 3
    data[position] ^= 0xFF  # flip a byte of a.txt inside the stored inner zip
    outer.write_bytes(bytes(data))
    with pytest.raises(zipfile.BadZipFile):
        extract_archive(outer, tmp_path / "out", inner=("inner.zip",))
    assert not (tmp_path / "out/a.txt").exists()


def test_ledger_tells_nested_archives_apart(tmp_path):
    import zipfile

    outer = tmp_path / "outer.zip"
    outer.write_bytes(_zip_bytes({"a.zip": _zip_bytes({"a": b"a"}, 0), "b.zip": _zip_bytes({"b": b"b"}, 0)}, 0))
    ledger = ExtractionLedger(tmp_path / "state")
    out = tmp_path / "out"
    stats = extract_archive(outer, out, inner=("a.zip",))
    ledger.mark_complete(outer, out, "all", stats, ("a.zip",))
    assert ledger.is_complete(outer, out, "all", ("a.zip",))
    assert not ledger.is_complete(outer, out, "all", ("b.zip",))
    assert not ledger.is_complete(outer, out, "all")
    assert zipfile.is_zipfile(outer)
