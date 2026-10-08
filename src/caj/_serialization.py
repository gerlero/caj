import struct
import sys
from typing import IO

from caj._utils import readexact

if sys.version_info >= (3, 14):
    from io import Writer
else:
    from typing_extensions import Writer

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.extend.core import Jaxpr


def dump_pytree(pytree: object, f: Writer[bytes], /) -> None:
    leaves, _ = jax.tree.flatten(pytree)

    for leaf in leaves:
        match leaf:
            case jax.ShapeDtypeStruct():
                raise jax.errors.ConcretizationTypeError(
                    leaf, "jax.ShapeDtypeStruct is not supported for serialization"
                )

            case jax.Array(ndim=0, weak_type=True):
                f.write(leaf.tobytes())

            case jax.Array(weak_type=True):
                raise NotImplementedError(
                    "non-scalar JAX arrays with weak_type=True are not supported"
                )

            case jax.Array():
                if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                    leaf = jax.random.key_data(leaf)

                f.write(leaf.tobytes())

            case np.ndarray() | np.generic():
                f.write(leaf.tobytes())

            case bool():
                f.write(struct.pack("?", leaf))

            case int():
                f.write(struct.pack("q", leaf))

            case float():
                f.write(struct.pack("d", leaf))

            case complex():
                f.write(struct.pack("dd", leaf.real, leaf.imag))


def load_pytree[T](f: IO[bytes], /, *, like: T) -> T:
    leaves_with_path, treedef = jax.tree.flatten_with_path(like)
    leaves = []

    for path, like_leaf in leaves_with_path:
        match like_leaf:
            case (
                jax.Array(ndim=0, weak_type=True)
                | jax.ShapeDtypeStruct(ndim=0, weak_type=True)
            ):
                leaf = jnp.frombuffer(
                    readexact(f, like_leaf.dtype.itemsize), dtype=like_leaf.dtype
                )
                leaf = jnp.array(leaf.item())
                assert leaf.ndim == 0
                assert leaf.dtype == like_leaf.dtype
                assert leaf.weak_type

            case jax.Array() | jax.ShapeDtypeStruct():
                if jnp.issubdtype(like_leaf.dtype, jax.dtypes.prng_key):
                    key_data_like = jax.eval_shape(jax.random.key_data, like_leaf)
                    leaf = jnp.frombuffer(
                        readexact(f, key_data_like.size * key_data_like.dtype.itemsize),
                        dtype=key_data_like.dtype,
                    )
                    leaf = leaf.reshape(key_data_like.shape)
                    leaf = jax.random.wrap_key_data(
                        leaf,
                        impl=jax.random.key_impl(jnp.zeros((), dtype=like_leaf.dtype)),
                    )
                    leaf = leaf.reshape(like_leaf.shape)
                    assert isinstance(leaf, jax.Array)
                    assert jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
                else:
                    leaf = jnp.frombuffer(
                        readexact(f, like_leaf.size * like_leaf.dtype.itemsize),
                        dtype=like_leaf.dtype,
                    )
                    leaf = leaf.reshape(like_leaf.shape)
                    if like_leaf.weak_type and not leaf.weak_type:
                        assert leaf.ndim != 0
                        raise NotImplementedError(
                            f"non-scalar JAX array with weak_type=True at {path} is not supported"
                        )
                assert isinstance(leaf, jax.Array)
                assert leaf.dtype == like_leaf.dtype
                assert leaf.shape == like_leaf.shape

            case np.ndarray():
                leaf = np.frombuffer(
                    readexact(f, like_leaf.size * like_leaf.dtype.itemsize),
                    dtype=like_leaf.dtype,
                )
                leaf = leaf.reshape(like_leaf.shape)
                assert isinstance(leaf, np.ndarray)
                assert leaf.dtype == like_leaf.dtype
                assert leaf.shape == like_leaf.shape

            case np.generic():
                assert like_leaf.ndim == 0
                leaf = np.frombuffer(
                    readexact(f, like_leaf.nbytes), dtype=like_leaf.dtype
                )
                leaf = leaf.reshape(())
                leaf = leaf.flat[0]
                assert isinstance(leaf, type(like_leaf))
                assert leaf.dtype == like_leaf.dtype
                assert leaf.ndim == 0

            case bool():
                (leaf,) = struct.unpack("?", readexact(f, 1))
                assert isinstance(leaf, bool)

            case int():
                (leaf,) = struct.unpack("q", readexact(f, 8))
                assert isinstance(leaf, int)

            case float():
                (leaf,) = struct.unpack("d", readexact(f, 8))
                assert isinstance(leaf, float)

            case complex():
                (real, imag) = struct.unpack("dd", readexact(f, 16))
                leaf = complex(real, imag)

            case _:
                leaf = like_leaf

        leaves.append(leaf)

    return treedef.unflatten(leaves)


def _dump_jaxpr_consts(jaxpr: Jaxpr, f: Writer[bytes], /) -> None:
    assert isinstance(jaxpr, Jaxpr)

    dump_pytree(jaxpr.consts, f)

    for eqn in jaxpr.eqns:
        leaves, _ = jax.tree.flatten(eqn.params)
        for leaf in leaves:
            if isinstance(leaf, Jaxpr):
                _dump_jaxpr_consts(leaf, f)


def dump_jaxpr(jaxpr: Jaxpr, f: Writer[bytes], /) -> None:
    assert isinstance(jaxpr, Jaxpr)

    f.write(jax.__version__.encode())
    f.write(jaxlib.__version__.encode())
    f.write(str(jaxpr).encode())
    _dump_jaxpr_consts(jaxpr, f)
