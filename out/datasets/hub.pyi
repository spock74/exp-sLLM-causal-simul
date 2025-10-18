from datasets.info import DatasetInfosDict as DatasetInfosDict
from datasets.load import load_dataset_builder as load_dataset_builder
from datasets.utils.metadata import MetadataConfigs as MetadataConfigs
from huggingface_hub import CommitInfo as CommitInfo

def delete_from_hub(repo_id: str, config_name: str, revision: str | None = None, token: bool | str | None = None) -> CommitInfo: ...
