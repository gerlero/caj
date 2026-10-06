import sys
from collections.abc import Buffer
from typing import Protocol

if sys.version_info >= (3, 14):
    from io import Reader
else:
    from typing_extensions import Reader


class SupportsReadSeek[T](Reader[T], Protocol):
    def seek(self, offset: int, whence: int, /) -> object: ...


class SupportsUpdate(Protocol):
    def update(self, data: Buffer, /) -> None: ...
