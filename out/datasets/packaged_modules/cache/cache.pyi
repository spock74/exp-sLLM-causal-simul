import datasets
import datasets.data_files
from _typeshed import Incomplete
from datasets.naming import camelcase_to_snakecase as camelcase_to_snakecase, filenames_for_dataset_split as filenames_for_dataset_split

logger: Incomplete

class Cache(datasets.ArrowBasedBuilder):
    def __init__(self, cache_dir: str | None = None, dataset_name: str | None = None, config_name: str | None = None, version: str | None = '0.0.0', hash: str | None = None, base_path: str | None = None, info: datasets.DatasetInfo | None = None, features: datasets.Features | None = None, token: bool | str | None = None, repo_id: str | None = None, data_files: str | list | dict | datasets.data_files.DataFilesDict | None = None, data_dir: str | None = None, storage_options: dict | None = None, writer_batch_size: int | None = None, **config_kwargs) -> None: ...
    def download_and_prepare(self, output_dir: str | None = None, *args, **kwargs): ...
