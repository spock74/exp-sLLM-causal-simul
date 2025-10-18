import abc
from . import config as config, utils as utils
from .arrow_dataset import Dataset as Dataset
from .arrow_reader import ArrowReader as ArrowReader, ReadInstruction as ReadInstruction
from .arrow_writer import ArrowWriter as ArrowWriter, ParquetWriter as ParquetWriter, SchemaInferenceError as SchemaInferenceError
from .data_files import DataFilesDict as DataFilesDict, DataFilesPatternsDict as DataFilesPatternsDict, sanitize_patterns as sanitize_patterns
from .dataset_dict import DatasetDict as DatasetDict, IterableDatasetDict as IterableDatasetDict
from .download.download_config import DownloadConfig as DownloadConfig
from .download.download_manager import DownloadManager as DownloadManager, DownloadMode as DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager as StreamingDownloadManager, xjoin as xjoin
from .exceptions import DatasetGenerationCastError as DatasetGenerationCastError, DatasetGenerationError as DatasetGenerationError, FileFormatError as FileFormatError, ManualDownloadError as ManualDownloadError
from .features import Features as Features
from .filesystems import is_remote_filesystem as is_remote_filesystem, rename as rename
from .fingerprint import Hasher as Hasher
from .info import DatasetInfo as DatasetInfo, PostProcessedInfo as PostProcessedInfo
from .iterable_dataset import ArrowExamplesIterable as ArrowExamplesIterable, ExamplesIterable as ExamplesIterable, IterableDataset as IterableDataset
from .keyhash import DuplicatedKeysError as DuplicatedKeysError
from .load import DatasetModule as DatasetModule
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH as INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase as camelcase_to_snakecase
from .splits import Split as Split, SplitDict as SplitDict, SplitGenerator as SplitGenerator, SplitInfo as SplitInfo
from .streaming import extend_dataset_builder_for_streaming as extend_dataset_builder_for_streaming
from .table import CastError as CastError
from .utils import logging as logging
from .utils._filelock import FileLock as FileLock
from .utils.file_utils import is_remote_url as is_remote_url
from .utils.info_utils import VerificationMode as VerificationMode, get_size_checksum_dict as get_size_checksum_dict, verify_checksums as verify_checksums, verify_splits as verify_splits
from .utils.py_utils import classproperty as classproperty, convert_file_size_to_int as convert_file_size_to_int, has_sufficient_disk_space as has_sufficient_disk_space, iflatmap_unordered as iflatmap_unordered, map_nested as map_nested, memoize as memoize, size_str as size_str, temporary_assignment as temporary_assignment
from .utils.track import tracked_list as tracked_list
from _typeshed import Incomplete
from dataclasses import dataclass

logger: Incomplete

class InvalidConfigName(ValueError): ...

@dataclass
class BuilderConfig:
    name: str = ...
    version: utils.Version | str | None = ...
    data_dir: str | None = ...
    data_files: DataFilesDict | DataFilesPatternsDict | None = ...
    description: str | None = ...
    def __post_init__(self) -> None: ...
    def __eq__(self, o): ...
    def create_config_id(self, config_kwargs: dict, custom_features: Features | None = None) -> str: ...

class DatasetBuilder(metaclass=abc.ABCMeta):
    VERSION: Incomplete
    BUILDER_CONFIG_CLASS = BuilderConfig
    BUILDER_CONFIGS: Incomplete
    DEFAULT_CONFIG_NAME: Incomplete
    DEFAULT_WRITER_BATCH_SIZE: Incomplete
    name: str
    hash: str | None
    base_path: Incomplete
    token: Incomplete
    repo_id: Incomplete
    storage_options: Incomplete
    dataset_name: Incomplete
    config_kwargs: Incomplete
    info: Incomplete
    dl_manager: Incomplete
    def __init__(self, cache_dir: str | None = None, dataset_name: str | None = None, config_name: str | None = None, hash: str | None = None, base_path: str | None = None, info: DatasetInfo | None = None, features: Features | None = None, token: bool | str | None = None, repo_id: str | None = None, data_files: str | list | dict | DataFilesDict | None = None, data_dir: str | None = None, storage_options: dict | None = None, writer_batch_size: int | None = None, **config_kwargs) -> None: ...
    @property
    def manual_download_instructions(self) -> str | None: ...
    @classproperty
    @classmethod
    def builder_configs(cls) -> dict[str, BuilderConfig]: ...
    @property
    def cache_dir(self): ...
    @classmethod
    def get_imported_module_dir(cls): ...
    def download_and_prepare(self, output_dir: str | None = None, download_config: DownloadConfig | None = None, download_mode: DownloadMode | str | None = None, verification_mode: VerificationMode | str | None = None, dl_manager: DownloadManager | None = None, base_path: str | None = None, file_format: str = 'arrow', max_shard_size: int | str | None = None, num_proc: int | None = None, storage_options: dict | None = None, **download_and_prepare_kwargs): ...
    def download_post_processing_resources(self, dl_manager) -> None: ...
    def as_dataset(self, split: str | Split | list[str] | list[Split] | None = None, run_post_process: bool = True, verification_mode: VerificationMode | str | None = None, in_memory: bool = False) -> Dataset | DatasetDict: ...
    def as_streaming_dataset(self, split: str | None = None, base_path: str | None = None) -> dict[str, IterableDataset] | IterableDataset: ...

class GeneratorBasedBuilder(DatasetBuilder, metaclass=abc.ABCMeta): ...
class ArrowBasedBuilder(DatasetBuilder, metaclass=abc.ABCMeta): ...
