import contextlib
import os
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import IO
from warnings import warn

if sys.version_info >= (3, 14):
    from io import Writer
else:
    from typing_extensions import Writer

from ._utils import BoundedWriter


class Cache:
    _SUFFIX = f".{__package__}"

    def __init__(self, dir: Path, /, *, max_bytes: int | None = None) -> None:
        self.dir = dir
        self.max_bytes = max_bytes

    @contextlib.contextmanager
    def read(self, key: str, /) -> Generator[IO[bytes], None, None]:
        with self._path(key).open("rb") as f, self._delete_on_error(key):
            yield f
        self._hit(key)

    @contextlib.contextmanager
    def write(self, key: str, /) -> Generator[Writer[bytes], None, None]:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = path.open("xb")
        with self._delete_on_error(key), f:
            if self.max_bytes is not None:
                f = BoundedWriter(f, max_bytes=self.max_bytes)
            yield f
        self._cull()

    def _path(self, key: str, /) -> Path:
        return self.dir / f"{key}{self._SUFFIX}"

    def _hit(self, key: str, /) -> None:
        path = self._path(key)
        with contextlib.suppress(OSError):
            os.utime(path, ns=(time.time_ns(), path.stat().st_mtime_ns))

    @contextlib.contextmanager
    def _delete_on_error(self, key: str, /) -> Generator[None, None, None]:
        try:
            yield
        except:
            path = self._path(key)
            with contextlib.suppress(OSError):
                if not path.is_symlink():
                    path.unlink()
            raise

    def _cull(self) -> None:
        if self.max_bytes is None:
            return

        cache_entries: list[tuple[Path, int, int]] = []

        try:
            with os.scandir(self.dir) as dir_entries:
                for dir_entry in dir_entries:
                    path = Path(dir_entry.path)

                    if path.suffix != self._SUFFIX:
                        continue

                    try:
                        if not dir_entry.is_file(follow_symlinks=False):
                            continue
                        st = dir_entry.stat(follow_symlinks=False)
                    except OSError:
                        continue

                    cache_entries.append((path, st.st_atime_ns, st.st_size))
        except OSError:
            warn(
                f"{__package__}: failed to cull cache in {self.dir} to {self.max_bytes} bytes; "
                "unable to list files",
                RuntimeWarning,
                stacklevel=3,
            )
            return

        total_size = sum(entry[2] for entry in cache_entries)

        if total_size <= self.max_bytes:
            return

        cache_entries.sort(key=lambda entry: entry[1])

        for cache_entry in cache_entries:
            assert cache_entry[0].suffix == self._SUFFIX
            try:
                cache_entry[0].unlink(missing_ok=True)
            except OSError:
                continue

            total_size -= cache_entry[2]

            if total_size <= self.max_bytes:
                return

        warn(
            f"{__package__}: failed to cull cache in {self.dir} to {self.max_bytes} bytes; "
            f"current size is at least {total_size} bytes",
            RuntimeWarning,
            stacklevel=3,
        )
