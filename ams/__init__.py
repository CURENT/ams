try:
    from ams._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from ams import opt  # NOQA

from ams.main import config_logger, load, run  # NOQA
from ams.system import System  # NOQA
from ams.utils.paths import get_case, list_cases  # NOQA

__author__ = 'Jining Wang'

__all__ = ['System', 'load', 'run', 'config_logger', 'get_case', 'list_cases']
