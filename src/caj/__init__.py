import functools
import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import overload
from warnings import warn

if sys.version_info < (3, 15):
    from typing_extensions import sentinel

import jax
from platformdirs import user_cache_path

from ._cache import Cache
from ._serialization import dump_jaxpr, dump_pytree, load_pytree
from ._utils import HashWriter, WriteLimitError

DEFAULT_CACHE_DIR = user_cache_path(__package__)
DEFAULT_CACHE_MAX_BYTES = 1_000_000_000

_MISSING = sentinel("_MISSING")


@overload
def cache[**P, R](func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def cache[**P, R](
    func: Callable[P, R],
    /,
    *,
    dir: str | os.PathLike[str],
    max_bytes: int | None = ...,
) -> Callable[P, R]: ...


@overload
def cache[**P, R](
    *, dir: str | os.PathLike[str], max_bytes: int | None = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def cache[**P, R](
    func: Callable[P, R] | _MISSING = _MISSING,
    /,
    *,
    dir: str | os.PathLike[str] | _MISSING = _MISSING,
    max_bytes: int | None | _MISSING = _MISSING,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    if dir is not _MISSING:
        dir = Path(dir).absolute()

        if max_bytes is _MISSING:
            max_bytes = DEFAULT_CACHE_MAX_BYTES
        elif max_bytes is not None and max_bytes <= 0:
            raise ValueError(f"max_bytes must be None or positive, got {max_bytes!r}")

    elif max_bytes is not _MISSING:
        raise TypeError("max_bytes is only allowed if dir is also specified")

    else:
        dir = DEFAULT_CACHE_DIR
        max_bytes = DEFAULT_CACHE_MAX_BYTES

    cache = Cache(dir, max_bytes=max_bytes)

    def decorator(func: Callable[P, R], /) -> Callable[P, R]:
        jaxpr_func = jax.make_jaxpr(func, return_shape=True)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            jaxpr, ret_shape = jaxpr_func(*args, **kwargs)

            h = hashlib.blake2b(digest_size=16)

            f = HashWriter(h)
            dump_jaxpr(jaxpr, f)
            dump_pytree(args, f)
            dump_pytree(kwargs, f)

            key = h.hexdigest()

            try:
                with cache.read(key) as f:
                    ret = load_pytree(f, like=ret_shape)
                    if f.read(1):
                        raise EOFError("extra data at end of cache entry")
            except KeyError:
                pass
            except (OSError, EOFError) as e:
                warn(
                    f"{__package__}: failed to load cached entry: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                return ret

            ret = func(*args, **kwargs)

            try:
                with cache.write(key) as f:
                    dump_pytree(ret, f)
            except (OSError, WriteLimitError) as e:
                warn(
                    f"{__package__}: failed to save cache to {cache.dir}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            return ret

        return wrapper

    return decorator(func) if func is not _MISSING else decorator
