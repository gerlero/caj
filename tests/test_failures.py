from pathlib import Path

import jax.numpy as jnp
import pytest

from caj import cache
from caj._cache import Cache


def test_oversized_entry(tmp_path: Path) -> None:
    @cache(dir=tmp_path, max_bytes=1)
    def f(x):
        return x + 1

    with pytest.warns(RuntimeWarning, match="caj:.*limit"):
        assert f(jnp.array(1.0)) == 2.0

    assert not list(tmp_path.glob("*.caj"))


def test_bad_entry(tmp_path: Path) -> None:
    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    x = jnp.array(1.0)

    assert f(x) == 2.0

    [entry] = tmp_path.glob("*.caj")
    entry.write_bytes(b"bad")

    with pytest.warns(RuntimeWarning, match="caj: failed to load"):
        assert f(x) == 2.0


def test_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_os_error(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(Cache, "write", raise_os_error)

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    with pytest.warns(RuntimeWarning, match="caj: failed to save"):
        assert f(jnp.array(1.0)) == 2.0
