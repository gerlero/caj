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


def test_pytree_leaf_order(tmp_path):
    @cache(dir=tmp_path)
    def f1(xs):
        a, b = xs
        return a - b

    @cache(dir=tmp_path)
    def f2(xs):
        b, a = xs
        return a - b

    a = jnp.array(1.0)
    b = jnp.array(2.0)

    assert not list(tmp_path.glob("*.caj"))
    assert f1((a, b)) == -1.0
    assert len(list(tmp_path.glob("*.caj"))) == 1
    assert f1((b, a)) == 1.0
    assert len(list(tmp_path.glob("*.caj"))) == 2
    assert f2((a, b)) == 1.0
    assert len(list(tmp_path.glob("*.caj"))) == 3
    assert f1((a, b)) == -1.0
    assert len(list(tmp_path.glob("*.caj"))) == 3


def test_nested_jaxpr_consts(tmp_path: Path) -> None:
    def make_f(value):
        a = jnp.array([value])

        @cache(dir=tmp_path)
        @jax.jit
        def f(x):
            return x + a

        return f

    f1 = make_f(1.0)
    f2 = make_f(2.0)

    x = jnp.array([10.0])

    assert f1(x) != f2(x)
