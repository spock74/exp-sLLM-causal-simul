import datasets
from ..folder_based_builder import folder_based_builder as folder_based_builder
from _typeshed import Incomplete

logger: Incomplete

class AudioFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    drop_labels: bool
    drop_metadata: bool
    def __post_init__(self) -> None: ...

class AudioFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Audio
    BASE_COLUMN_NAME: str
    BUILDER_CONFIG_CLASS = AudioFolderConfig
    EXTENSIONS: list[str]

AUDIO_EXTENSIONS: Incomplete
