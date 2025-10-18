import yaml
from ..config import METADATA_CONFIGS_FIELD as METADATA_CONFIGS_FIELD
from ..features import Features as Features
from ..info import DatasetInfo as DatasetInfo, DatasetInfosDict as DatasetInfosDict
from ..utils.logging import get_logger as get_logger
from _typeshed import Incomplete
from huggingface_hub import DatasetCardData as DatasetCardData
from typing import Any, ClassVar

logger: Incomplete

class _NoDuplicateSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep: bool = False): ...

class MetadataConfigs(dict[str, dict[str, Any]]):
    FIELD_NAME: ClassVar[str]
    @classmethod
    def from_dataset_card_data(cls, dataset_card_data: DatasetCardData) -> MetadataConfigs: ...
    def to_dataset_card_data(self, dataset_card_data: DatasetCardData) -> None: ...
    def get_default_config_name(self) -> str | None: ...

known_task_ids: Incomplete
