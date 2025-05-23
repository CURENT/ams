from . import _version
__version__ = _version.get_versions()['version']

from ams import opt  # NOQA

from ams.main import config_logger, load, run  # NOQA
from ams.system import System  # NOQA
from ams.utils.paths import get_case, list_cases  # NOQA

__author__ = 'Jining Wang'

__all__ = ['System', 'get_case']
