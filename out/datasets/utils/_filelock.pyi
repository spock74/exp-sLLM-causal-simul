from filelock import FileLock as FileLock_

class FileLock(FileLock_):
    MAX_FILENAME_LENGTH: int
    def __init__(self, lock_file, *args, **kwargs) -> None: ...
    @classmethod
    def hash_filename_if_too_long(cls, path: str) -> str: ...
