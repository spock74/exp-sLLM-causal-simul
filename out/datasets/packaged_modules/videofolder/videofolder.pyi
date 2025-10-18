import datasets
from ..folder_based_builder import folder_based_builder as folder_based_builder
from _typeshed import Incomplete

logger: Incomplete

class VideoFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    drop_labels: bool
    drop_metadata: bool
    def __post_init__(self) -> None: ...

class VideoFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Video
    BASE_COLUMN_NAME: str
    BUILDER_CONFIG_CLASS = VideoFolderConfig
    EXTENSIONS: list[str]

VIDEO_EXTENSIONS: Incomplete
