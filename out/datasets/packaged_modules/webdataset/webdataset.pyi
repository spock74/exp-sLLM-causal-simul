import datasets
from _typeshed import Incomplete
from datasets.features.features import cast_to_python_objects as cast_to_python_objects
from datasets.utils.file_utils import SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL as SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL, xbasename as xbasename
from typing import Any, Callable

logger: Incomplete

class WebDataset(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE: int
    IMAGE_EXTENSIONS: list[str]
    AUDIO_EXTENSIONS: list[str]
    VIDEO_EXTENSIONS: list[str]
    DECODERS: dict[str, Callable[[Any], Any]]
    NUM_EXAMPLES_FOR_FEATURES_INFERENCE: int

def base_plus_ext(path): ...

IMAGE_EXTENSIONS: Incomplete
AUDIO_EXTENSIONS: Incomplete
VIDEO_EXTENSIONS: Incomplete

def text_loads(data: bytes): ...
def tenbin_loads(data: bytes): ...
def msgpack_loads(data: bytes): ...
def npy_loads(data: bytes): ...
def npz_loads(data: bytes): ...
def cbor_loads(data: bytes): ...
def torch_loads(data: bytes): ...

DECODERS: Incomplete
