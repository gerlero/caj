import sys
from collections.abc import Buffer
from typing import Protocol, override

if sys.version_info >= (3, 14):
    from io import Reader, Writer
else:
    from typing_extensions import Reader, Writer


def readexact[B: Buffer](f: Reader[B], n: int, /) -> B:
    data = f.read(n)
    if len(memoryview(data)) < n:
        raise EOFError(f"unexpected end of file while reading {n} bytes")
    return data


class _SupportsUpdate(Protocol):
    def update(self, data: Buffer, /) -> None: ...


class HashWriter[B: Buffer](Writer[B]):
    def __init__(self, hasher: _SupportsUpdate, /) -> None:
        self._hasher = hasher

    @override
    def write(self, data: Buffer, /) -> int:
        nbytes = memoryview(data).nbytes
        self._hasher.update(data)
        return nbytes


class WriteLimitError(Exception):
    pass


class BoundedWriter[B: Buffer](Writer[B]):
    def __init__(self, writer: Writer[B], /, *, max_bytes: int) -> None:
        self._writer = writer
        self._max_bytes = max_bytes
        self._bytes_written = 0

    @override
    def write(self, data: B, /) -> int:
        nbytes = memoryview(data).nbytes
        if self._bytes_written + nbytes > self._max_bytes:
            raise WriteLimitError(
                f"write would exceed limit of {self._max_bytes} bytes"
            )

        ret = self._writer.write(data)
        assert 0 <= ret <= nbytes

        self._bytes_written += ret
        return ret
