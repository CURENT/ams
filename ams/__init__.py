from . import _version
__version__ = _version.get_versions()['version']

from ams import io         # NOQA
from ams import core       # NOQA
from ams import models     # NOQA
from ams import routines   # NOQA

from ams.main import load  # NOQA
from ams.system import System  # NOQA


__author__ = 'Jining Wang'

__all__ = ['main', 'system',
           'core', 'models', 'io', 'routines',
           '__version__']
