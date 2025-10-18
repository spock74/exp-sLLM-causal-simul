import datasets
import h5py
import numpy as np
from _typeshed import Incomplete
from collections.abc import Generator
from dataclasses import dataclass, field
from datasets.features.features import Array2D as Array2D, Array3D as Array3D, Array4D as Array4D, Array5D as Array5D, Features as Features, LargeList as LargeList, List as List, Value as Value
from datasets.table import cast_table_to_features as cast_table_to_features

logger: Incomplete
EXTENSIONS: Incomplete

@dataclass
class HDF5Config(datasets.BuilderConfig):
    batch_size: int | None = ...
    features: datasets.Features | None = ...

class HDF5(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = HDF5Config

@dataclass
class _CompoundGroup:
    dset: h5py.Dataset
    data: np.ndarray = ...
    def items(self) -> Generator[Incomplete]: ...

@dataclass
class _CompoundField:
    data: np.ndarray | None
    name: str
    dtype: np.dtype
    shape: tuple[int, ...] = field(init=False)
    def __post_init__(self) -> None: ...
    def __getitem__(self, key): ...
