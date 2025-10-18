import abc
from .. import config as config
from ._filelock import FileLock as FileLock
from .logging import get_logger as get_logger
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path

logger: Incomplete

class ExtractManager:
    extract_dir: Incomplete
    extractor: Incomplete
    def __init__(self, cache_dir: str | None = None) -> None: ...
    def extract(self, input_path: str, force_extract: bool = False) -> str: ...

class BaseExtractor(ABC, metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def is_extractable(cls, path: Path | str, **kwargs) -> bool: ...
    @staticmethod
    @abstractmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class MagicNumberBaseExtractor(BaseExtractor, ABC, metaclass=abc.ABCMeta):
    magic_numbers: list[bytes]
    @staticmethod
    def read_magic_number(path: Path | str, magic_number_length: int): ...
    @classmethod
    def is_extractable(cls, path: Path | str, magic_number: bytes = b'') -> bool: ...

class TarExtractor(BaseExtractor):
    @classmethod
    def is_extractable(cls, path: Path | str, **kwargs) -> bool: ...
    @staticmethod
    def safemembers(members, output_path) -> Generator[Incomplete, None, Incomplete]: ...
    @staticmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class GzipExtractor(MagicNumberBaseExtractor):
    magic_numbers: Incomplete
    @staticmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class ZipExtractor(MagicNumberBaseExtractor):
    magic_numbers: Incomplete
    @classmethod
    def is_extractable(cls, path: Path | str, magic_number: bytes = b'') -> bool: ...
    @staticmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class XzExtractor(MagicNumberBaseExtractor):
    magic_numbers: Incomplete
    @staticmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class RarExtractor(MagicNumberBaseExtractor):
    magic_numbers: Incomplete
    @staticmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class ZstdExtractor(MagicNumberBaseExtractor):
    magic_numbers: Incomplete
    @staticmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class Bzip2Extractor(MagicNumberBaseExtractor):
    magic_numbers: Incomplete
    @staticmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class SevenZipExtractor(MagicNumberBaseExtractor):
    magic_numbers: Incomplete
    @staticmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class Lz4Extractor(MagicNumberBaseExtractor):
    magic_numbers: Incomplete
    @staticmethod
    def extract(input_path: Path | str, output_path: Path | str) -> None: ...

class Extractor:
    extractors: dict[str, type[BaseExtractor]]
    @classmethod
    def is_extractable(cls, path: Path | str, return_extractor: bool = False) -> bool: ...
    @classmethod
    def infer_extractor_format(cls, path: Path | str) -> str | None: ...
    @classmethod
    def extract(cls, input_path: Path | str, output_path: Path | str, extractor_format: str) -> None: ...
