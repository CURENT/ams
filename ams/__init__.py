from . import _version
__version__ = _version.get_versions()['version']

from ams.main import config_logger, load  # NOQA
from ams.utils.paths import get_case  # NOQA

__author__ = 'Jining Wang'
