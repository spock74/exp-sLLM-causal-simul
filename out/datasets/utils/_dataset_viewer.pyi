from .. import config as config
from ..exceptions import DatasetsError as DatasetsError
from .file_utils import get_authentication_headers_for_url as get_authentication_headers_for_url
from .logging import get_logger as get_logger
from _typeshed import Incomplete
from typing import Any

logger: Incomplete

class DatasetViewerError(DatasetsError): ...

def get_exported_parquet_files(dataset: str, commit_hash: str, token: str | bool | None) -> list[dict[str, Any]]: ...
def get_exported_dataset_infos(dataset: str, commit_hash: str, token: str | bool | None) -> dict[str, dict[str, Any]]: ...
