import datasets
from ..folder_based_builder import folder_based_builder as folder_based_builder
from _typeshed import Incomplete

logger: Incomplete

class PdfFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    drop_labels: bool
    drop_metadata: bool
    def __post_init__(self) -> None: ...

class PdfFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Pdf
    BASE_COLUMN_NAME: str
    BUILDER_CONFIG_CLASS = PdfFolderConfig
    EXTENSIONS: list[str]
