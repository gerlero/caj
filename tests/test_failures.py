from pathlib import Path
from typing import NoReturn

import jax.numpy as jnp
import pytest

from caj import cache
from caj._cache import Cache


def test_oversized_entry(tmp_path: Path) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path, max_bytes=1)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    with pytest.warns(RuntimeWarning, match="caj:.*limit"):
        assert f(jnp.array(1.0)) == 2.0

    [] = tmp_path.glob("*.caj")


def test_incomplete_entry(tmp_path: Path) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    x = jnp.array(1.0)

    assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    size = entry.stat().st_size

    with entry.open("r+b") as c:
        c.seek(-1, 2)
        c.truncate()
    assert entry.stat().st_size == size - 1

    assert f(x) == 2.0
    [_] = tmp_path.glob("*.caj")
    assert entry.stat().st_size == size


def test_entry_created_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    x = jnp.array(1.0)

    assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    mtime_ns = entry.stat().st_mtime_ns

    def error(*args: object, **kwargs: object) -> NoReturn:
        raise FileNotFoundError

    monkeypatch.setattr(Cache, "read", error)

    assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    assert entry.stat().st_mtime_ns == mtime_ns


def test_entry_read_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    x = jnp.array(1.0)

    assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    size = entry.stat().st_size

    def error(*args: object, **kwargs: object) -> NoReturn:
        raise OSError

    monkeypatch.setattr(Cache, "read", error)

    with pytest.warns(RuntimeWarning, match="caj:.*failed to load"):
        assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    assert entry.stat().st_size == size


def test_entry_write_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    x = jnp.array(1.0)

    def error(*args: object, **kwargs: object) -> NoReturn:
        raise OSError

    monkeypatch.setattr(Cache, "write", error)

    with pytest.warns(RuntimeWarning, match="caj:.*failed to save"):
        assert f(x) == 2.0
    [] = tmp_path.glob("*.caj")


def test_entry_too_long(tmp_path: Path) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    x = jnp.array(1.0)

    assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    size = entry.stat().st_size

    entry.write_bytes(entry.read_bytes() + b"\x00")
    assert entry.stat().st_size == size + 1

    with pytest.warns(RuntimeWarning, match="caj:.*failed to load"):
        assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    assert entry.stat().st_size == size
