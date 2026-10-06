from pathlib import Path

import jax
import jax.numpy as jnp

from caj import cache


def test_basic(tmp_path: Path) -> None:
    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    x = jnp.array(1.0)

    assert f(x) == 2.0
    assert f(x) == 2.0
    assert len(list(tmp_path.glob("*.caj"))) == 1

    assert f(x) == 2.0
    assert len(list(tmp_path.glob("*.caj"))) == 1

    assert f(jnp.array(2.0)) == 3.0
    assert len(list(tmp_path.glob("*.caj"))) == 2


def test_different_functions_create_different_entries(tmp_path: Path) -> None:
    @cache(dir=tmp_path)
    def add_two(x):
        return x + 2

    @cache(dir=tmp_path)
    def times_two(x):
        return x * 2

    assert add_two(jnp.array(2.0)) == 4.0
    assert times_two(jnp.array(2.0)) == 4.0

    assert len(list(tmp_path.glob("*.caj"))) == 2


def test_random_keys(tmp_path: Path) -> None:
    @cache(dir=tmp_path)
    def f(key):
        return jax.random.normal(key)

    key = jax.random.key(42)

    assert jnp.allclose(f(key), f(key))
