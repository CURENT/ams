from . import _version
__version__ = _version.get_versions()['version']

from ams import routines  # NOQA
from ams import benchmarks  # NOQA

from ams.main import config_logger, load, run  # NOQA
from ams.utils.paths import get_case  # NOQA
from ams.shared import ppc2df  # NOQA

__author__ = 'Jining Wang'

__all__ = ['main', 'system', 'cli']
