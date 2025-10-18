import fsspec
from . import compression as compression

COMPRESSION_FILESYSTEMS: list[compression.BaseCompressedFileFileSystem]

def is_remote_filesystem(fs: fsspec.AbstractFileSystem) -> bool: ...
def rename(fs: fsspec.AbstractFileSystem, src: str, dst: str): ...
