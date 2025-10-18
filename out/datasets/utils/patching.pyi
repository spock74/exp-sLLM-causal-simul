import types
from .logging import get_logger as get_logger
from _typeshed import Incomplete

logger: Incomplete

class _PatchedModuleObj:
    def __init__(self, module, attrs=None) -> None: ...

class patch_submodule:
    obj: Incomplete
    target: Incomplete
    new: Incomplete
    key: Incomplete
    original: Incomplete
    attrs: Incomplete
    def __init__(self, obj, target: str, new, attrs=None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc_info) -> None: ...
    def start(self) -> None: ...
    def stop(self): ...
