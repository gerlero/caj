import sys
from collections.abc import Buffer
from typing import override

if sys.version_info >= (3, 14):
    from io import Writer
else:
    from typing_extensions import Writer


from ._typing import SupportsUpdate


class HashWriter(Writer[Buffer]):
    def __init__(self, hasher: SupportsUpdate, /) -> None:
        self._hasher = hasher

    @override
    def write(self, data: Buffer, /) -> int:
        data = memoryview(data)
        self._hasher.update(data)
        return data.nbytes


class WriteLimitError(Exception):
    pass


class BoundedWriter(Writer[Buffer]):
    def __init__(self, writer: Writer[Buffer], /, *, max_bytes: int) -> None:
        self._writer = writer
        self._max_bytes = max_bytes
        self._bytes_written = 0

    @override
    def write(self, data: Buffer, /) -> int:
        data = memoryview(data)
        if self._bytes_written + data.nbytes > self._max_bytes:
            raise WriteLimitError(
                f"write would exceed limit of {self._max_bytes} bytes"
            )

        ret = self._writer.write(data)
        assert 0 <= ret <= data.nbytes

        self._bytes_written += ret
        return ret
