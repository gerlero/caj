import sys

from caj._utils import readexact

if sys.version_info >= (3, 14):
    from io import Reader, Writer
else:
    from typing_extensions import Reader, Writer

import jax
import jax.numpy as jnp
from jax.extend.core import Jaxpr

FORMAT = 0


def dump_pytree(pytree: object, f: Writer[bytes], /) -> None:
    leaves, _ = jax.tree.flatten(pytree)

    for leaf in leaves:
        match leaf:
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

            case _:
                raise TypeError(
                    f"non-jax.Array leaf of type {type(leaf)} is not supported"
                )


def load_pytree[T](f: Reader[bytes], /, *, like: T) -> T:
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

            case jax.Array(weak_type=True) | jax.ShapeDtypeStruct(weak_type=True):
                raise NotImplementedError(
                    f"non-scalar jax.Array with weak_type=True at {path} is not supported"
                )

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
                            f"non-scalar jax.Array with weak_type=True at {path} is not supported"
                        )
                assert isinstance(leaf, jax.Array)
                assert leaf.dtype == like_leaf.dtype
                assert leaf.shape == like_leaf.shape

            case _:
                raise TypeError(
                    f"non-jax.Array leaf of type {type(like_leaf)} at {path} is not supported"
                )

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

    f.write(str(jaxpr).encode())
    _dump_jaxpr_consts(jaxpr, f)
