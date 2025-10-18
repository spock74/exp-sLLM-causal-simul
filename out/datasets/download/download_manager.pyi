import enum
import io
from .. import config as config
from ..utils.file_utils import ArchiveIterable as ArchiveIterable, FilesIterable as FilesIterable, cached_path as cached_path, is_relative_path as is_relative_path, stack_multiprocessing_download_progress_bars as stack_multiprocessing_download_progress_bars, url_or_path_join as url_or_path_join
from ..utils.info_utils import get_size_checksum_dict as get_size_checksum_dict
from ..utils.logging import get_logger as get_logger, tqdm as tqdm
from ..utils.py_utils import NestedDataStructure as NestedDataStructure, map_nested as map_nested
from ..utils.track import tracked_str as tracked_str
from .download_config import DownloadConfig as DownloadConfig
from _typeshed import Incomplete

logger: Incomplete

class DownloadMode(enum.Enum):
    REUSE_DATASET_IF_EXISTS = 'reuse_dataset_if_exists'
    REUSE_CACHE_IF_EXISTS = 'reuse_cache_if_exists'
    FORCE_REDOWNLOAD = 'force_redownload'

class DownloadManager:
    is_streaming: bool
    record_checksums: Incomplete
    download_config: Incomplete
    downloaded_paths: Incomplete
    extracted_paths: Incomplete
    def __init__(self, dataset_name: str | None = None, data_dir: str | None = None, download_config: DownloadConfig | None = None, base_path: str | None = None, record_checksums: bool = True) -> None: ...
    @property
    def manual_dir(self): ...
    @property
    def downloaded_size(self): ...
    def download(self, url_or_urls): ...
    def iter_archive(self, path_or_buf: str | io.BufferedReader): ...
    def iter_files(self, paths: str | list[str]): ...
    def extract(self, path_or_paths): ...
    def download_and_extract(self, url_or_urls): ...
    def get_recorded_sizes_checksums(self): ...
    def delete_extracted_files(self) -> None: ...
    def manage_extracted_files(self) -> None: ...
