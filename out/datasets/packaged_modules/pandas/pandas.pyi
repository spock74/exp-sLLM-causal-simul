import datasets
from dataclasses import dataclass
from datasets.table import table_cast as table_cast

@dataclass
class PandasConfig(datasets.BuilderConfig):
    features: datasets.Features | None = ...
    def __post_init__(self) -> None: ...

class Pandas(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = PandasConfig
