from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from caj import cache


def test_basic(tmp_path: Path) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f(x):
        return x + 1

    [] = tmp_path.glob("*.caj")

    x = jnp.array(1.0)

    assert f(x) == 2.0
    [entry] = tmp_path.glob("*.caj")
    prev_st = entry.stat()

    assert f(x) == 2.0
    [_] = tmp_path.glob("*.caj")
    st = entry.stat()
    assert st.st_size == prev_st.st_size
    assert st.st_mtime_ns == prev_st.st_mtime_ns
    assert st.st_atime_ns > prev_st.st_atime_ns
    prev_st = st

    assert f(x) == 2.0
    [_] = tmp_path.glob("*.caj")
    st = entry.stat()
    assert st.st_size == prev_st.st_size
    assert st.st_mtime_ns == prev_st.st_mtime_ns
    assert st.st_atime_ns > prev_st.st_atime_ns

    assert f(jnp.array(2.0)) == 3.0
    [_, _] = tmp_path.glob("*.caj")


def test_different_functions_create_different_entries(tmp_path: Path) -> None:
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def add_two(x):
        return x + 2

    @cache(dir=tmp_path)
    def times_two(x):
        return x * 2

    [] = tmp_path.glob("*.caj")

    assert add_two(jnp.array(2.0)) == 4.0
    [_] = tmp_path.glob("*.caj")

    assert times_two(jnp.array(2.0)) == 4.0
    [_, _] = tmp_path.glob("*.caj")


def test_random_keys(tmp_path: Path) -> None:
    @cache(dir=tmp_path)
    def f(key):
        return jax.random.normal(key)

    key = jax.random.key(42)

    assert jnp.allclose(f(key), f(key))


def test_pytree_leaf_order(tmp_path):
    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f1(xs):
        a, b = xs
        return a - b

    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f2(xs):
        b, a = xs
        return a - b

    [] = tmp_path.glob("*.caj")

    a = jnp.array(1.0)
    b = jnp.array(2.0)

    assert f1((a, b)) == -1.0
    [_] = tmp_path.glob("*.caj")
    assert f1((b, a)) == 1.0
    [_, _] = tmp_path.glob("*.caj")
    assert f2((a, b)) == 1.0
    [_, _, _] = tmp_path.glob("*.caj")
    assert f1((a, b)) == -1.0
    [_, _, _] = tmp_path.glob("*.caj")


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

    [] = tmp_path.glob("*.caj")
    assert f1(x) != f2(x)
    [_, _] = tmp_path.glob("*.caj")


def test_no_input_confusion(tmp_path: Path) -> None:
    x = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}

    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f1(x):
        return x["a"]

    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f2(x):
        return x["b"]

    [] = tmp_path.glob("*.caj")

    f1(x)
    [_] = tmp_path.glob("*.caj")

    f2(x)
    [_, _] = tmp_path.glob("*.caj")


def test_static_field_in_output(tmp_path: Path) -> None:
    class Class(eqx.Module):
        static: int = eqx.field(static=True)
        dynamic: jax.Array

    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f1(x):
        return Class(static=1, dynamic=x)

    [] = tmp_path.glob("*.caj")

    @cache(dir=tmp_path)
    def f2(x):
        return Class(static=2, dynamic=x)

    [] = tmp_path.glob("*.caj")

    y1 = f1(jnp.array([10.0]))
    assert isinstance(y1, Class)
    assert y1.dynamic == jnp.array([10.0])
    assert y1.static == 1
    [_] = tmp_path.glob("*.caj")

    y2 = f2(jnp.array([10.0]))
    assert isinstance(y2, Class)
    assert y2.dynamic == jnp.array([10.0])
    assert y2.static == 2
    [_] = tmp_path.glob("*.caj")

    y2b = f2(jnp.array([20.0]))
    assert isinstance(y2b, Class)
    assert y2b.dynamic == jnp.array([20.0])
    assert y2b.static == 2
    [_, _] = tmp_path.glob("*.caj")

    y1b = f1(jnp.array([20.0]))
    assert isinstance(y1b, Class)
    assert y1b.dynamic == jnp.array([20.0])
    assert y1b.static == 1
    [_, _] = tmp_path.glob("*.caj")


def test_cache_dir_creation(tmp_path: Path) -> None:
    dir = tmp_path / "non_existent"
    assert not dir.exists()

    @cache(dir=dir)
    def f(x):
        return x + 1

    assert not dir.exists()

    x = jnp.array(1.0)

    assert f(x) == 2.0
    assert dir.is_dir()
    [entry] = dir.glob("*.caj")
    assert entry.exists()
