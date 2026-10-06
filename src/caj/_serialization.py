import sys
from typing import Any

if sys.version_info >= (3, 14):
    from io import Writer
else:
    from typing_extensions import Writer

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxlib

from ._typing import SupportsReadSeek


def serialize_jaxpr(f: Writer[bytes], jaxpr: Any, /) -> None:
    f.write(jax.__version__.encode())
    f.write(jaxlib.__version__.encode())
    f.write(str(jaxpr).encode())
    serialize_pytree(f, jaxpr.consts)


# Workaround for https://github.com/patrick-kidger/equinox/issues/1255
def serialize_filter_spec(f: Writer[bytes], x: object, /) -> None:
    if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
        x = jax.random.key_data(x)
    return eqx.default_serialise_filter_spec(f, x)


def serialize_pytree(
    f: Writer[bytes],
    pytree: object,
    /,
    *,
    exc_workaround: type[Exception] | None = None,
) -> None:
    if exc_workaround is not None:
        # Workaround for https://github.com/patrick-kidger/equinox/issues/1255
        assert issubclass(exc_workaround, Exception)
        assert not issubclass(exc_workaround, RuntimeError)

        try:
            eqx.tree_serialise_leaves(f, pytree, filter_spec=serialize_filter_spec)
        except RuntimeError as e:
            cause = e.__cause__
            while isinstance(cause, RuntimeError):
                cause = cause.__cause__
            if isinstance(cause, exc_workaround):
                raise cause from e  # ty: ignore[invalid-raise]
            raise
    else:
        eqx.tree_serialise_leaves(f, pytree, filter_spec=serialize_filter_spec)


# Workaround for https://github.com/patrick-kidger/equinox/issues/1255
def deserialize_filter_spec(f: SupportsReadSeek[bytes], x: object, /) -> object:
    ret = eqx.default_deserialise_filter_spec(f, x)
    if isinstance(x, (jax.Array, jax.ShapeDtypeStruct)) and jnp.issubdtype(
        x.dtype, jax.dtypes.prng_key
    ):
        return jax.random.wrap_key_data(
            ret, impl=jax.random.key_impl(jnp.zeros((), x.dtype))
        )
    return ret


def deserialize_pytree[T](f: SupportsReadSeek[bytes], /, *, like: T) -> T:
    return eqx.tree_deserialise_leaves(
        f, like=like, filter_spec=deserialize_filter_spec
    )
