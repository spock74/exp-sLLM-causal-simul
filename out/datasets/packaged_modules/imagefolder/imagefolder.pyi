import datasets
from ..folder_based_builder import folder_based_builder as folder_based_builder
from _typeshed import Incomplete

logger: Incomplete

class ImageFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    drop_labels: bool
    drop_metadata: bool
    def __post_init__(self) -> None: ...

class ImageFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Image
    BASE_COLUMN_NAME: str
    BUILDER_CONFIG_CLASS = ImageFolderConfig
    EXTENSIONS: list[str]

IMAGE_EXTENSIONS: Incomplete
