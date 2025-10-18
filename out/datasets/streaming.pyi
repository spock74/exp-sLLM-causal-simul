from .builder import DatasetBuilder as DatasetBuilder
from .download.download_config import DownloadConfig as DownloadConfig
from .utils.file_utils import xPath as xPath, xbasename as xbasename, xdirname as xdirname, xet_parse as xet_parse, xexists as xexists, xgetsize as xgetsize, xglob as xglob, xgzip_open as xgzip_open, xisdir as xisdir, xisfile as xisfile, xjoin as xjoin, xlistdir as xlistdir, xnumpy_load as xnumpy_load, xopen as xopen, xpandas_read_csv as xpandas_read_csv, xpandas_read_excel as xpandas_read_excel, xpyarrow_parquet_read_table as xpyarrow_parquet_read_table, xrelpath as xrelpath, xsio_loadmat as xsio_loadmat, xsplit as xsplit, xsplitext as xsplitext, xwalk as xwalk, xxml_dom_minidom_parse as xxml_dom_minidom_parse
from .utils.logging import get_logger as get_logger
from .utils.patching import patch_submodule as patch_submodule
from _typeshed import Incomplete

logger: Incomplete

def extend_module_for_streaming(module_path, download_config: DownloadConfig | None = None): ...
def extend_dataset_builder_for_streaming(builder: DatasetBuilder): ...
