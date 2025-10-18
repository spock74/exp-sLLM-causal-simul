import contextlib
from ..utils import experimental as experimental, logging as logging
from _typeshed import Incomplete

logger: Incomplete

class ParallelBackendConfig:
    backend_name: Incomplete

@experimental
def parallel_map(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func): ...
@experimental
@contextlib.contextmanager
def parallel_backend(backend_name: str): ...
