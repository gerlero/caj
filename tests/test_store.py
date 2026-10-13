import contextlib
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from caj._store import Store
from caj._utils import WriteLimitError


def test_cache_read_write(tmp_path: Path) -> None:
    cache = Store(tmp_path)

    with cache.write("key") as f:
        assert f.write(b"data") == 4

    path = tmp_path / "key.caj"
    mtime = path.stat().st_mtime_ns
    os.utime(path, ns=(1_000_000_000, mtime))

    with cache.read("key") as f:
        assert f.read() == b"data"

    st = path.stat()
    assert st.st_atime_ns > 1_000_000_000
    assert st.st_mtime_ns == mtime


@pytest.mark.parametrize("operation", ["read", "write"])
def test_error_removes_entry(tmp_path: Path, operation: str) -> None:
    cache = Store(tmp_path)
    path = tmp_path / "key.caj"
    if operation == "read":
        path.write_bytes(b"data")

    with (
        pytest.raises(ValueError, match="failed"),
        getattr(cache, operation)("key") as f,
    ):
        if operation == "write":
            f.write(b"data")
        raise ValueError("failed")

    assert not path.exists()


def test_failed_cleanup_does_not_replace_original_exception(tmp_path: Path) -> None:
    cache = Store(tmp_path)
    with pytest.raises(ValueError, match="original"), cache._delete_on_error("missing"):
        raise ValueError("original")


def test_error_does_not_delete_symlink(tmp_path: Path) -> None:
    cache = Store(tmp_path)
    target = tmp_path / "target"
    target.write_bytes(b"data")
    link = tmp_path / "link.caj"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    with pytest.raises(ValueError, match="failed"), cache.read("link") as f:
        assert f.read() == b"data"
        raise ValueError("failed")

    assert link.is_symlink()
    assert target.read_bytes() == b"data"


def test_hit_ignores_missing_entry(tmp_path: Path) -> None:
    [] = tmp_path.iterdir()
    Store(tmp_path)._hit("missing")
    [] = tmp_path.iterdir()


def test_write_limit_and_cleanup(tmp_path: Path) -> None:
    cache = Store(tmp_path, max_bytes=3)
    with cache.write("fits") as f:
        assert f.write(b"a") == 1
        assert f.write(b"bc") == 2
    assert (tmp_path / "fits.caj").read_bytes() == b"abc"

    with pytest.raises(WriteLimitError), cache.write("too-large") as f:
        f.write(b"ab")
        f.write(b"cd")

    assert not (tmp_path / "too-large.caj").exists()


def test_cull_evicts_least_recently_used(tmp_path: Path) -> None:
    cache = Store(tmp_path, max_bytes=4)
    for key, atime in [("a", 1), ("b", 2)]:
        with cache.write(key) as f:
            f.write(b"xx")
        path = tmp_path / f"{key}.caj"
        os.utime(path, ns=(atime * 1_000_000_000, path.stat().st_mtime_ns))

    with cache.read("a") as f:
        assert f.read() == b"xx"

    with cache.write("c") as f:
        f.write(b"xx")

    assert (tmp_path / "a.caj").exists()
    assert not (tmp_path / "b.caj").exists()
    assert (tmp_path / "c.caj").exists()


def test_cull_removes_multiple_entries(tmp_path: Path) -> None:
    cache = Store(tmp_path, max_bytes=2)
    for i, key in enumerate(("a", "b", "c"), start=1):
        path = tmp_path / f"{key}.caj"
        path.write_bytes(b"xx")
        os.utime(path, ns=(i * 1_000_000_000, i * 1_000_000_000))

    cache._cull()
    assert {p.name for p in tmp_path.iterdir()} == {"c.caj"}


def test_cull_ignores_other_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "unrelated.txt").write_bytes(b"x" * 100)
    (tmp_path / "directory.caj").mkdir()
    cache = Store(tmp_path, max_bytes=2)

    with cache.write("key") as f:
        f.write(b"xx")

    assert (tmp_path / "key.caj").read_bytes() == b"xx"
    assert (tmp_path / "unrelated.txt").exists()
    assert (tmp_path / "directory.caj").is_dir()


def test_cull_warns_if_directory_cannot_be_listed(tmp_path: Path) -> None:
    cache = Store(tmp_path / "missing", max_bytes=1)
    with pytest.warns(RuntimeWarning, match="unable to list files"):
        cache._cull()


@pytest.mark.parametrize("method", ["is_file", "stat"])
def test_cull_skips_unreadable_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    entry = Mock()
    entry.path = str(tmp_path / "unreadable.caj")
    entry.is_file.return_value = True
    getattr(entry, method).side_effect = OSError("inaccessible")

    with monkeypatch.context() as m:
        m.setattr(os, "scandir", lambda _: contextlib.nullcontext([entry]))
        Store(tmp_path, max_bytes=1)._cull()


def test_cull_warns_if_entry_cannot_be_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Store(tmp_path, max_bytes=2)
    path = tmp_path / "locked.caj"
    path.write_bytes(b"abcd")
    original_unlink = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self == path:
            raise PermissionError("locked")
        original_unlink(self, missing_ok=missing_ok)

    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", unlink)
        with pytest.warns(RuntimeWarning, match="current size is at least 4 bytes"):
            cache._cull()

    assert path.read_bytes() == b"abcd"
