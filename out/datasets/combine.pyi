from .arrow_dataset import Dataset as Dataset
from .dataset_dict import DatasetDict as DatasetDict, IterableDatasetDict as IterableDatasetDict
from .info import DatasetInfo as DatasetInfo
from .iterable_dataset import IterableDataset as IterableDataset
from .splits import NamedSplit as NamedSplit
from .utils import logging as logging
from .utils.py_utils import Literal as Literal
from _typeshed import Incomplete
from typing import TypeVar

logger: Incomplete
DatasetType = TypeVar('DatasetType', Dataset, IterableDataset)

def interleave_datasets(datasets: list[DatasetType], probabilities: list[float] | None = None, seed: int | None = None, info: DatasetInfo | None = None, split: NamedSplit | None = None, stopping_strategy: Literal['first_exhausted', 'all_exhausted', 'all_exhausted_without_replacement'] = 'first_exhausted') -> DatasetType: ...
def concatenate_datasets(dsets: list[DatasetType], info: DatasetInfo | None = None, split: NamedSplit | None = None, axis: int = 0) -> DatasetType: ...
