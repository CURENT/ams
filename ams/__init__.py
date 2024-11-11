from . import _version
__version__ = _version.get_versions()['version']

from ams.main import config_logger, load, run  # NOQA
from ams.utils.paths import get_case  # NOQA
from ams.shared import ppc2df  # NOQA

__author__ = 'Jining Wang'

__all__ = ['system']
