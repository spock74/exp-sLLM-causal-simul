import enum
from .. import config as config
from ..exceptions import ExpectedMoreDownloadedFilesError as ExpectedMoreDownloadedFilesError, ExpectedMoreSplitsError as ExpectedMoreSplitsError, NonMatchingChecksumError as NonMatchingChecksumError, NonMatchingSplitsSizesError as NonMatchingSplitsSizesError, UnexpectedDownloadedFileError as UnexpectedDownloadedFileError, UnexpectedSplitsError as UnexpectedSplitsError
from .logging import get_logger as get_logger
from _typeshed import Incomplete

logger: Incomplete

class VerificationMode(enum.Enum):
    ALL_CHECKS = 'all_checks'
    BASIC_CHECKS = 'basic_checks'
    NO_CHECKS = 'no_checks'

def verify_checksums(expected_checksums: dict | None, recorded_checksums: dict, verification_name=None): ...
def verify_splits(expected_splits: dict | None, recorded_splits: dict): ...
def get_size_checksum_dict(path: str, record_checksum: bool = True) -> dict: ...
def is_small_dataset(dataset_size): ...
