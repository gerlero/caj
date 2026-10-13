import functools
import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import overload
from warnings import warn

import jax
import jaxlib
from platformdirs import user_cache_path

if sys.version_info < (3, 15):
    from typing_extensions import sentinel

from ._serialization import FORMAT, dump_jaxpr, dump_pytree, load_pytree
from ._store import Store
from ._utils import HashWriter, WriteLimitError

DEFAULT_CACHE_DIR = user_cache_path(__package__)
DEFAULT_CACHE_MAX_BYTES = 1_000_000_000

_MISSING = sentinel("_MISSING")


class _InvalidCacheEntryError(Exception):
    pass


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

    store = Store(dir, max_bytes=max_bytes)

    def decorator(func: Callable[P, R], /) -> Callable[P, R]:
        jaxpr_func = jax.make_jaxpr(func, return_shape=True)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            args, kwargs = jax.device_put((args, kwargs))
            jaxpr, ret_shape = jaxpr_func(*args, **kwargs)

            base_hasher = hashlib.blake2b(digest_size=16)
            base_hasher.update(FORMAT.to_bytes())
            base_hasher.update(jax.__version__.encode())
            base_hasher.update(jaxlib.__version__.encode())
            dump_jaxpr(jaxpr, HashWriter(base_hasher))

            def callback(*args: P.args, **kwargs: P.kwargs) -> R:
                hasher = base_hasher.copy()
                f = HashWriter(hasher)
                dump_pytree(args, f)
                dump_pytree(kwargs, f)
                key = hasher.hexdigest()

                try:
                    with store.read(key) as f:
                        ret = load_pytree(f, like=ret_shape)
                        if f.read(1):
                            raise _InvalidCacheEntryError(
                                "unexpected extra data in cached entry"
                            )
                except (FileNotFoundError, EOFError):
                    pass
                except (OSError, _InvalidCacheEntryError) as e:
                    warn(
                        f"{__package__}: failed to load cached entry: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    return ret

                ret = func(*args, **kwargs)

                try:
                    with store.write(key) as f:
                        dump_pytree(ret, f)
                except FileExistsError:
                    pass
                except (OSError, WriteLimitError) as e:
                    warn(
                        f"{__package__}: failed to save cache to {store.dir}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                return ret

            return jax.pure_callback(callback, ret_shape, *args, **kwargs)

        return wrapper

    return decorator(func) if func is not _MISSING else decorator
