import datasets
from _typeshed import Incomplete
from dataclasses import dataclass
from datasets.table import table_cast as table_cast

logger: Incomplete

@dataclass
class ArrowConfig(datasets.BuilderConfig):
    features: datasets.Features | None = ...
    def __post_init__(self) -> None: ...

class Arrow(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = ArrowConfig
