from .arrow_dataset import Dataset as Dataset
from .iterable_dataset import IterableDataset as IterableDataset
from typing import TypeVar

DatasetType = TypeVar('DatasetType', Dataset, IterableDataset)

def split_dataset_by_node(dataset: DatasetType, rank: int, world_size: int) -> DatasetType: ...
