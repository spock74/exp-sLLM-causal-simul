import datasets
from _typeshed import Incomplete
from dataclasses import dataclass
from datasets.features.features import require_storage_cast as require_storage_cast
from datasets.table import table_cast as table_cast

logger: Incomplete

@dataclass
class XmlConfig(datasets.BuilderConfig):
    features: datasets.Features | None = ...
    encoding: str = ...
    encoding_errors: str | None = ...

class Xml(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = XmlConfig
