from _typeshed import Incomplete
from fsspec.archive import AbstractArchiveFileSystem

class BaseCompressedFileFileSystem(AbstractArchiveFileSystem):
    root_marker: str
    protocol: str
    compression: str
    extensions: list[str]
    fo: Incomplete
    compressed_name: Incomplete
    uncompressed_name: Incomplete
    dir_cache: Incomplete
    def __init__(self, fo: str = '', target_protocol: str | None = None, target_options: dict | None = None, **kwargs) -> None: ...
    def cat(self, path: str): ...

class Bz2FileSystem(BaseCompressedFileFileSystem):
    protocol: str
    compression: str
    extensions: Incomplete

class GzipFileSystem(BaseCompressedFileFileSystem):
    protocol: str
    compression: str
    extensions: Incomplete

class Lz4FileSystem(BaseCompressedFileFileSystem):
    protocol: str
    compression: str
    extensions: Incomplete

class XzFileSystem(BaseCompressedFileFileSystem):
    protocol: str
    compression: str
    extensions: Incomplete

class ZstdFileSystem(BaseCompressedFileFileSystem):
    protocol: str
    compression: str
    extensions: Incomplete
