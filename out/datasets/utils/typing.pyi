import os
from typing import TypeVar

T = TypeVar('T')
ListLike = list[T] | tuple[T, ...]
NestedDataStructureLike = T | list[T] | dict[str, T]
PathLike = str | bytes | os.PathLike
