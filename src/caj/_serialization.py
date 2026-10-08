import sys
from typing import IO, Any

if sys.version_info >= (3, 14):
    from io import Writer
else:
    from typing_extensions import Writer

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


def dump_pytree(pytree: object, f: Writer[bytes], /) -> None:
    leaves, _ = jax.tree.flatten(pytree)

    for leaf in leaves:
        match leaf:
            case (
                jax.Array(ndim=0, weak_type=True)
                | jax.ShapeDtypeStruct(ndim=0, weak_type=True)
            ):
                jnp.save(f, leaf, allow_pickle=False)

            case jax.Array(weak_type=True) | jax.ShapeDtypeStruct(weak_type=True):
                raise NotImplementedError(
                    "non-scalar JAX arrays with weak_type=True are not supported"
                )

            case jax.Array():
                if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                    leaf = jax.random.key_data(leaf)

                jnp.save(f, leaf, allow_pickle=False)

            case np.ndarray() | float() | int() | complex() | bool() | np.generic():
                np.save(f, leaf, allow_pickle=False)


def load_pytree[T](f: IO[bytes], /, *, like: T) -> T:
    leaves_with_path, treedef = jax.tree.flatten_with_path(like)
    leaves = []

    try:
        for path, like_leaf in leaves_with_path:
            match like_leaf:
                case (
                    jax.Array(ndim=0, weak_type=True)
                    | jax.ShapeDtypeStruct(ndim=0, weak_type=True)
                ):
                    leaf = jnp.load(f, allow_pickle=False)
                    if not isinstance(leaf, jax.Array):
                        raise TypeError(
                            f"expected JAX scalar array for {path}, got {type(leaf)}"
                        )
                    if leaf.dtype != like_leaf.dtype:
                        raise TypeError(
                            f"expected JAX scalar array with dtype {like_leaf.dtype} for {path}, got {leaf.dtype}"
                        )
                    if leaf.ndim != 0:
                        raise ValueError(
                            f"expected JAX scalar array for {path}, got array with shape {leaf.shape}"
                        )
                    leaf = jnp.array(leaf.item())
                    assert leaf.ndim == 0
                    assert leaf.dtype == like_leaf.dtype
                    assert leaf.weak_type

                case jax.Array() | jax.ShapeDtypeStruct():
                    leaf = jnp.load(f, allow_pickle=False)
                    if not isinstance(leaf, jax.Array):
                        raise TypeError(
                            f"expected JAX array for {path}, got {type(leaf)}"
                        )

                    if jnp.issubdtype(like_leaf.dtype, jax.dtypes.prng_key):
                        leaf = jax.random.wrap_key_data(
                            leaf,
                            impl=jax.random.key_impl(
                                jnp.zeros((), dtype=like_leaf.dtype)
                            ),
                        )
                        assert isinstance(leaf, jax.Array)
                        assert jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
                    elif like_leaf.weak_type and not leaf.weak_type:
                        assert leaf.ndim != 0
                        raise NotImplementedError(
                            f"non-scalar JAX array with weak_type=True at {path} is not supported"
                        )

                    if leaf.dtype != like_leaf.dtype:
                        raise TypeError(
                            f"expected JAX array with dtype {like_leaf.dtype} for {path}, got {leaf.dtype}"
                        )
                    if leaf.shape != like_leaf.shape:
                        raise ValueError(
                            f"expected JAX array with shape {like_leaf.shape} for {path}, got {leaf.shape}"
                        )

                case np.ndarray():
                    leaf = np.load(f, allow_pickle=False)
                    if not isinstance(leaf, np.ndarray):
                        raise TypeError(
                            f"expected NumPy array for {path}, got {type(leaf)}"
                        )
                    if leaf.dtype != like_leaf.dtype:
                        raise TypeError(
                            f"expected NumPy array with dtype {like_leaf.dtype} for {path}, got {leaf.dtype}"
                        )
                    if leaf.shape != like_leaf.shape:
                        raise ValueError(
                            f"expected NumPy array with shape {like_leaf.shape} for {path}, got {leaf.shape}"
                        )

                case np.generic():
                    leaf = np.load(f, allow_pickle=False)
                    if not isinstance(leaf, np.ndarray):
                        raise TypeError(
                            f"expected {type(like_leaf)} for {path}, got {type(leaf)}"
                        )
                    if leaf.ndim != 0:
                        raise ValueError(
                            f"expected scalar {type(like_leaf)} for {path}, got array with shape {leaf.shape}"
                        )
                    leaf = leaf.flat[0]
                    if type(leaf) is not type(like_leaf):
                        raise TypeError(
                            f"expected {type(like_leaf)} for {path}, got {type(leaf)}"
                        )

                case float() | int() | complex() | bool():
                    leaf = np.load(f, allow_pickle=False)
                    if not isinstance(leaf, np.ndarray):
                        raise TypeError(
                            f"expected {type(like_leaf)} for {path}, got {type(leaf)}"
                        )
                    if leaf.ndim != 0:
                        raise ValueError(
                            f"expected scalar {type(like_leaf)} for {path}, got array with shape {leaf.shape}"
                        )
                    leaf = leaf.item()
                    if type(leaf) is not type(like_leaf):
                        raise TypeError(
                            f"expected {type(like_leaf)} for {path}, got {type(leaf)}"
                        )

                case _:
                    leaf = like_leaf

            leaves.append(leaf)
    except EOFError as e:
        raise TypeError(f"missing leaf for {path} (expected {type(like_leaf)})") from e

    return treedef.unflatten(leaves)


def dump_jaxpr(jaxpr: Any, f: Writer[bytes], /) -> None:
    f.write(jax.__version__.encode())
    f.write(jaxlib.__version__.encode())
    f.write(str(jaxpr).encode())
    dump_pytree(jaxpr.consts, f)
