from _typeshed import Incomplete

class InvalidKeyError(Exception):
    prefix: str
    err_msg: Incomplete
    suffix: str
    def __init__(self, hash_data) -> None: ...

class DuplicatedKeysError(Exception):
    key: Incomplete
    duplicate_key_indices: Incomplete
    fix_msg: Incomplete
    prefix: str
    err_msg: Incomplete
    suffix: Incomplete
    def __init__(self, key, duplicate_key_indices, fix_msg: str = '') -> None: ...

class KeyHasher:
    def __init__(self, hash_salt: str) -> None: ...
    def hash(self, key: str | int | bytes) -> int: ...
