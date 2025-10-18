from .. import Dataset as Dataset, Features as Features, NamedSplit as NamedSplit, config as config
from ..arrow_writer import get_writer_batch_size_from_data_size as get_writer_batch_size_from_data_size, get_writer_batch_size_from_features as get_writer_batch_size_from_features
from ..formatting import query_table as query_table
from ..packaged_modules.parquet.parquet import Parquet as Parquet
from ..utils.typing import NestedDataStructureLike as NestedDataStructureLike, PathLike as PathLike
from .abc import AbstractDatasetReader as AbstractDatasetReader
from _typeshed import Incomplete
from typing import BinaryIO

class ParquetDatasetReader(AbstractDatasetReader):
    builder: Incomplete
    def __init__(self, path_or_paths: NestedDataStructureLike[PathLike], split: NamedSplit | None = None, features: Features | None = None, cache_dir: str = None, keep_in_memory: bool = False, streaming: bool = False, num_proc: int | None = None, **kwargs) -> None: ...
    def read(self): ...

class ParquetDatasetWriter:
    dataset: Incomplete
    path_or_buf: Incomplete
    batch_size: Incomplete
    storage_options: Incomplete
    parquet_writer_kwargs: Incomplete
    use_content_defined_chunking: Incomplete
    write_page_index: Incomplete
    def __init__(self, dataset: Dataset, path_or_buf: PathLike | BinaryIO, batch_size: int | None = None, storage_options: dict | None = None, use_content_defined_chunking: bool | dict = True, write_page_index: bool = True, **parquet_writer_kwargs) -> None: ...
    def write(self) -> int: ...
