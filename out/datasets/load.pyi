from . import __version__ as __version__, config as config
from .arrow_dataset import Dataset as Dataset
from .builder import BuilderConfig as BuilderConfig, DatasetBuilder as DatasetBuilder
from .data_files import DataFilesDict as DataFilesDict, DataFilesList as DataFilesList, DataFilesPatternsDict as DataFilesPatternsDict, EmptyDatasetError as EmptyDatasetError, get_data_patterns as get_data_patterns, sanitize_patterns as sanitize_patterns
from .dataset_dict import DatasetDict as DatasetDict, IterableDatasetDict as IterableDatasetDict
from .download.download_config import DownloadConfig as DownloadConfig
from .download.download_manager import DownloadMode as DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager as StreamingDownloadManager, xbasename as xbasename, xglob as xglob, xjoin as xjoin
from .exceptions import DataFilesNotFoundError as DataFilesNotFoundError, DatasetNotFoundError as DatasetNotFoundError
from .features import Features as Features
from .fingerprint import Hasher as Hasher
from .info import DatasetInfo as DatasetInfo, DatasetInfosDict as DatasetInfosDict
from .iterable_dataset import IterableDataset as IterableDataset
from .naming import camelcase_to_snakecase as camelcase_to_snakecase, snakecase_to_camelcase as snakecase_to_camelcase
from .packaged_modules.folder_based_builder.folder_based_builder import FolderBasedBuilder as FolderBasedBuilder
from .splits import Split as Split
from .utils.file_utils import cached_path as cached_path, get_datasets_user_agent as get_datasets_user_agent, is_relative_path as is_relative_path, relative_to_absolute_path as relative_to_absolute_path
from .utils.hub import hf_dataset_url as hf_dataset_url
from .utils.info_utils import VerificationMode as VerificationMode, is_small_dataset as is_small_dataset
from .utils.logging import get_logger as get_logger
from .utils.metadata import MetadataConfigs as MetadataConfigs
from .utils.typing import PathLike as PathLike
from .utils.version import Version as Version
from _typeshed import Incomplete
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger: Incomplete
ALL_ALLOWED_EXTENSIONS: Incomplete

class _InitializeConfiguredDatasetBuilder:
    def __call__(self, builder_cls, metadata_configs, default_config_name, name): ...

def configure_builder_class(builder_cls: type[DatasetBuilder], builder_configs: list[BuilderConfig], default_config_name: str | None, dataset_name: str) -> type[DatasetBuilder]: ...
def import_main_class(module_path) -> type[DatasetBuilder] | None: ...
def get_dataset_builder_class(dataset_module: DatasetModule, dataset_name: str | None = None) -> type[DatasetBuilder]: ...
def increase_load_count(name: str): ...
def infer_module_for_data_files_list(data_files_list: DataFilesList, download_config: DownloadConfig | None = None) -> tuple[str | None, dict]: ...
def infer_module_for_data_files_list_in_archives(data_files_list: DataFilesList, download_config: DownloadConfig | None = None) -> tuple[str | None, dict]: ...
def infer_module_for_data_files(data_files: DataFilesDict, path: str | None = None, download_config: DownloadConfig | None = None) -> tuple[str | None, dict[str, Any]]: ...
def create_builder_configs_from_metadata_configs(module_path: str, metadata_configs: MetadataConfigs, base_path: str | None = None, default_builder_kwargs: dict[str, Any] = None, download_config: DownloadConfig | None = None) -> tuple[list[BuilderConfig], str]: ...

@dataclass
class BuilderConfigsParameters:
    metadata_configs: MetadataConfigs | None = ...
    builder_configs: list[BuilderConfig] | None = ...
    default_config_name: str | None = ...

@dataclass
class DatasetModule:
    module_path: str
    hash: str
    builder_kwargs: dict
    builder_configs_parameters: BuilderConfigsParameters = field(default_factory=BuilderConfigsParameters)
    dataset_infos: DatasetInfosDict | None = ...

class _DatasetModuleFactory:
    def get_module(self) -> DatasetModule: ...

class LocalDatasetModuleFactory(_DatasetModuleFactory):
    path: Incomplete
    name: Incomplete
    data_files: Incomplete
    data_dir: Incomplete
    download_mode: Incomplete
    def __init__(self, path: str, data_dir: str | None = None, data_files: str | list | dict | None = None, download_mode: DownloadMode | str | None = None) -> None: ...
    def get_module(self) -> DatasetModule: ...

class PackagedDatasetModuleFactory(_DatasetModuleFactory):
    name: Incomplete
    data_files: Incomplete
    data_dir: Incomplete
    download_config: Incomplete
    download_mode: Incomplete
    def __init__(self, name: str, data_dir: str | None = None, data_files: str | list | dict | None = None, download_config: DownloadConfig | None = None, download_mode: DownloadMode | str | None = None) -> None: ...
    def get_module(self) -> DatasetModule: ...

class HubDatasetModuleFactory(_DatasetModuleFactory):
    name: Incomplete
    commit_hash: Incomplete
    data_files: Incomplete
    data_dir: Incomplete
    download_config: Incomplete
    download_mode: Incomplete
    use_exported_dataset_infos: Incomplete
    def __init__(self, name: str, commit_hash: str, data_dir: str | None = None, data_files: str | list | dict | None = None, download_config: DownloadConfig | None = None, download_mode: DownloadMode | str | None = None, use_exported_dataset_infos: bool = False) -> None: ...
    def get_module(self) -> DatasetModule: ...

class HubDatasetModuleFactoryWithParquetExport(_DatasetModuleFactory):
    name: Incomplete
    commit_hash: Incomplete
    download_config: Incomplete
    def __init__(self, name: str, commit_hash: str, download_config: DownloadConfig | None = None) -> None: ...
    def get_module(self) -> DatasetModule: ...

class CachedDatasetModuleFactory(_DatasetModuleFactory):
    name: Incomplete
    cache_dir: Incomplete
    def __init__(self, name: str, cache_dir: str | None = None) -> None: ...
    def get_module(self) -> DatasetModule: ...

def dataset_module_factory(path: str, revision: str | Version | None = None, download_config: DownloadConfig | None = None, download_mode: DownloadMode | str | None = None, data_dir: str | None = None, data_files: dict | list | str | DataFilesDict | None = None, cache_dir: str | None = None, **download_kwargs) -> DatasetModule: ...
def load_dataset_builder(path: str, name: str | None = None, data_dir: str | None = None, data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None, cache_dir: str | None = None, features: Features | None = None, download_config: DownloadConfig | None = None, download_mode: DownloadMode | str | None = None, revision: str | Version | None = None, token: bool | str | None = None, storage_options: dict | None = None, **config_kwargs) -> DatasetBuilder: ...
def load_dataset(path: str, name: str | None = None, data_dir: str | None = None, data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None, split: str | Split | list[str] | list[Split] | None = None, cache_dir: str | None = None, features: Features | None = None, download_config: DownloadConfig | None = None, download_mode: DownloadMode | str | None = None, verification_mode: VerificationMode | str | None = None, keep_in_memory: bool | None = None, save_infos: bool = False, revision: str | Version | None = None, token: bool | str | None = None, streaming: bool = False, num_proc: int | None = None, storage_options: dict | None = None, **config_kwargs) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset: ...
def load_from_disk(dataset_path: PathLike, keep_in_memory: bool | None = None, storage_options: dict | None = None) -> Dataset | DatasetDict: ...
