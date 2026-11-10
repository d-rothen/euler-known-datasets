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
