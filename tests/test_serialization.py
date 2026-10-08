import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from caj._serialization import dump_jaxpr, dump_pytree, load_pytree


@pytest.mark.parametrize(
    "array",
    [
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.bfloat16),
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.float16),
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.complex64),
        jnp.array([1, 2, 3], dtype=jnp.int32),
        jnp.array([True, False, True], dtype=jnp.bool_),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0], dtype=np.float16),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([1.0, 2.0, 3.0], dtype=np.complex64),
        np.array([1.0, 2.0, 3.0], dtype=np.complex128),
        np.array([1, 2, 3], dtype=np.int32),
        np.array([1, 2, 3], dtype=np.int64),
        np.array([True, False, True], dtype=np.bool_),
    ],
)
def test_array_roundtrip(array: jax.Array | np.ndarray) -> None:
    f = io.BytesIO()
    dump_pytree(array, f)

    f.seek(0)
    result = load_pytree(f, like=(array + array))

    assert type(result) is type(array)
    assert result.dtype == array.dtype
    assert result.shape == array.shape
    assert (result == array).all()


def test_pytree_roundtrip() -> None:
    pytree = {
        "w": jnp.array([1.0, 2.0]),
        "x": "strings are not serialized",
        "y": (3, jnp.array(4.0, dtype=jnp.bfloat16)),
        "z": np.array([5.0, 6.0, 7.0]),
    }

    f = io.BytesIO()
    dump_pytree(pytree, f)

    pytree_like = {
        "w": jnp.array([0.0, 0.0]),
        "x": "some other string",
        "y": (0, jnp.array(0.0, dtype=jnp.bfloat16)),
        "z": np.array([0.0, 0.0, 0.0]),
    }
    f.seek(0)
    result = load_pytree(f, like=pytree_like)

    assert jax.tree.structure(result) == jax.tree.structure(pytree)
    assert isinstance(result["w"], jax.Array)
    assert result["w"].dtype == pytree["w"].dtype
    assert jnp.array_equal(result["w"], pytree["w"])
    assert isinstance(result["x"], str)
    assert result["x"] == pytree_like["x"]
    assert isinstance(result["y"][1], jax.Array)
    assert result["y"][0] == pytree["y"][0]
    assert result["y"][1].dtype == pytree["y"][1].dtype
    assert jnp.array_equal(result["y"][1], pytree["y"][1])
    assert isinstance(result["z"], np.ndarray)
    assert result["z"].dtype == pytree["z"].dtype
    assert np.array_equal(result["z"], pytree["z"])


def test_prng_key_roundtrip() -> None:
    key = jax.random.key(42)

    f = io.BytesIO()
    dump_pytree(key, f)

    f.seek(0)
    result = load_pytree(f, like=key)

    assert result.dtype == key.dtype
    assert jnp.array_equal(
        jax.random.key_data(result),
        jax.random.key_data(key),
    )


@pytest.mark.parametrize(
    "scalar",
    [
        3.14,
        np.float32(3.14),
        np.float64(3.14),
        jnp.float16(3.14),
        jnp.bfloat16(3.14),
        jnp.float32(3.14),
        jnp.array(3.14),
        42,
        np.int32(42),
        np.int64(42),
        jnp.int32(42),
        jnp.array(42),
        1 + 2j,
        np.complex64(1 + 2j),
        np.complex128(1 + 2j),
        jnp.complex64(1 + 2j),
        jnp.array(1 + 2j),
        True,
        np.bool_(True),
        jnp.bool_(True),
        jnp.array(True),
    ],
)
def test_scalar_roundtrip(scalar: object) -> None:
    f = io.BytesIO()
    dump_pytree(scalar, f)

    f.seek(0)
    result = load_pytree(f, like=scalar)

    assert type(result) is type(scalar)
    assert result == scalar

    if isinstance(scalar, jax.Array):
        assert isinstance(result, jax.Array)
        assert result.dtype == scalar.dtype
        assert result.ndim == 0
        assert result.weak_type == scalar.weak_type


def test_serialize_jaxpr_is_deterministic() -> None:
    jaxpr = jax.make_jaxpr(lambda x: x**2 + 1)(1.0)

    f1 = io.BytesIO()
    f2 = io.BytesIO()

    dump_jaxpr(jaxpr, f1)
    dump_jaxpr(jaxpr, f2)

    assert f1.getvalue() == f2.getvalue()


def test_serialize_jaxpr_depends_on_jaxpr() -> None:
    jaxpr1 = jax.make_jaxpr(lambda x: x + 1)(1.0)
    jaxpr2 = jax.make_jaxpr(lambda x: x * 2)(1.0)

    f1 = io.BytesIO()
    f2 = io.BytesIO()

    dump_jaxpr(jaxpr1, f1)
    dump_jaxpr(jaxpr2, f2)

    assert f1.getvalue() != f2.getvalue()


def test_serialize_jaxpr_depends_on_consts() -> None:
    a = jnp.array([1.0, 2.0])
    b = jnp.array([1.0, 3.0])

    jaxpr1 = jax.make_jaxpr(lambda x: x + a)(jnp.zeros(2))
    jaxpr2 = jax.make_jaxpr(lambda x: x + b)(jnp.zeros(2))

    assert str(jaxpr1) == str(jaxpr2)

    f1 = io.BytesIO()
    f2 = io.BytesIO()

    dump_jaxpr(jaxpr1, f1)
    dump_jaxpr(jaxpr2, f2)

    assert f1.getvalue() != f2.getvalue()


def test_deserialize_array_different_dtype() -> None:
    f = io.BytesIO()
    array = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    dump_pytree(array, f)

    f.seek(0)
    with pytest.raises(TypeError, match="dtype"):
        load_pytree(f, like=jnp.empty(3, dtype=jnp.int32))


def test_deserialize_array_different_shape() -> None:
    f = io.BytesIO()
    array = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    dump_pytree(array, f)

    f.seek(0)
    with pytest.raises(ValueError, match="shape"):
        load_pytree(f, like=jnp.empty(4, dtype=jnp.float32))


def test_deserialize_scalar_different_type() -> None:
    f = io.BytesIO()
    scalar = 3.14
    dump_pytree(scalar, f)

    f.seek(0)
    with pytest.raises(TypeError, match="expected"):
        load_pytree(f, like=42)


def test_deserialize_pytree_missing_leaf() -> None:
    f = io.BytesIO()
    pytree = {
        "x": jnp.array([1.0, 2.0]),
        "y": (3, jnp.array(4.0)),
    }
    dump_pytree(pytree, f)

    pytree_like = {
        "x": jnp.array([0.0, 0.0]),
        "y": (0, jnp.array(0.0)),
        "z": jnp.array([5.0, 6.0]),
    }
    f.seek(0)
    with pytest.raises(TypeError, match="missing"):
        load_pytree(f, like=pytree_like)


def test_deserialize_pytree_invalid_data() -> None:
    f = io.BytesIO(b"not a valid pytree")
    pytree_like = {
        "x": jnp.array([0.0, 0.0]),
        "y": (0, jnp.array(0.0)),
    }
    f.seek(0)
    with pytest.raises(ValueError):
        load_pytree(f, like=pytree_like)


def test_unsupported_weak_type_jax_array() -> None:
    f = io.BytesIO()
    weak_array = jnp.broadcast_to(jnp.array(1), (3,))
    assert weak_array.weak_type
    with pytest.raises(NotImplementedError, match="weak_type=True"):
        dump_pytree(weak_array, f)

    f.seek(0)
    f.truncate()
    dump_pytree(jnp.arange(3), f)

    f.seek(0)
    with pytest.raises(NotImplementedError, match="weak_type=True"):
        load_pytree(f, like=weak_array)
