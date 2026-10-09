from pathlib import Path

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


def test_bad_entry(tmp_path: Path) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    x = jnp.array(1.0)

    assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    good = entry.read_bytes()
    entry.write_bytes(b"bad")

    assert f(x) == 2.0
    [_] = tmp_path.glob("*.caj")
    assert entry.read_bytes() == good


def test_incomplete_entry(tmp_path: Path) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    x = jnp.array(1.0)

    assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    good = entry.read_bytes()

    with entry.open("r+b") as c:
        c.seek(-1, 2)
        c.truncate()
    assert entry.stat().st_size == len(good) - 1

    assert f(x) == 2.0
    [_] = tmp_path.glob("*.caj")
    assert entry.read_bytes() == good


def test_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    [] = tmp_path.glob("*.caj")

    def raise_os_error(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(Cache, "write", raise_os_error)

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    with pytest.warns(RuntimeWarning, match="caj: failed to save"):
        assert f(jnp.array(1.0)) == 2.0

    [] = tmp_path.glob("*.caj")
