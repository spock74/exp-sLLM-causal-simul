import datasets
import pyarrow.dataset as ds
from _typeshed import Incomplete
from dataclasses import dataclass
from datasets import config as config
from datasets.features.features import FeatureType as FeatureType, require_storage_cast as require_storage_cast
from datasets.utils.file_utils import readline as readline

logger: Incomplete

def count_path_segments(path): ...

@dataclass
class FolderBasedBuilderConfig(datasets.BuilderConfig):
    features: datasets.Features | None = ...
    drop_labels: bool = ...
    drop_metadata: bool = ...
    metadata_filenames: list[str] = ...
    filters: ds.Expression | list[tuple] | list[list[tuple]] | None = ...
    def __post_init__(self) -> None: ...

class FolderBasedBuilder(datasets.GeneratorBasedBuilder):
    BASE_FEATURE: type[FeatureType]
    BASE_COLUMN_NAME: str
    BUILDER_CONFIG_CLASS: FolderBasedBuilderConfig
    EXTENSIONS: list[str]
    METADATA_FILENAMES: list[str]
