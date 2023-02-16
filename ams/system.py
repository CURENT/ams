"""
System class for power system data, methods, and routines.
"""
import configparser
import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

from andes.core import Config
from andes.system import ExistingModels as andes_ExistingModels
from andes.system import System as andes_System
from andes.system import (_config_numpy, load_config_rc)
from andes.variables import FileMan

from ams.utils.paths import (ams_root, get_config_path)
import ams.io

logger = logging.getLogger(__name__)


class ExistingModels(andes_ExistingModels):
    """
    Storage class for existing models used in dispatch routines.

    The class is revised from ``andes.system.ExistingModels``.
    """

    def __init__(self):
        self.dcpflow = OrderedDict()
        self.dcopf = OrderedDict()
        self.pflow = OrderedDict()
        self.opf = OrderedDict()

# TODO: might need a method ``add`` to register a new routine


class System(andes_System):
    """
    System contains data, models, and routines for dispatch modeling and analysis.

    """

    def __init__(self,
                 case: Optional[str] = None,
                 name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 config_path: Optional[str] = None,
                 default_config: Optional[bool] = False,
                 options: Optional[Dict] = None,
                 **kwargs
                 ):
        self.name = name
        self.options = {}
        if options is not None:
            self.options.update(options)
        if kwargs:
            self.options.update(kwargs)
        # ams.io.parse(self)

        # get and load default config file
        self._config_path = get_config_path()
        if config_path is not None:
            self._config_path = config_path
        if default_config is True:
            self._config_path = None

        self._config_object = load_config_rc(self._config_path)
        self._update_config_object()
        self.config = Config(self.__class__.__name__, dct=config)
        self.config.load(self._config_object)

        # custom configuration for system goes after this line
        self.config.add(OrderedDict((('freq', 60),
                                     ('mva', 100),
                                     ('seed', 'None'),
                                     ('np_divide', 'warn'),
                                     ('np_invalid', 'warn'),
                                     )))

        self.config.add_extra("_help",
                              freq='base frequency [Hz]',
                              mva='system base MVA',
                              seed='seed (or None) for random number generator',
                              np_divide='treatment for division by zero',
                              np_invalid='treatment for invalid floating-point ops.',
                              )

        self.config.add_extra("_alt",
                              freq="float",
                              mva="float",
                              seed='int or None',
                              np_divide={'ignore', 'warn', 'raise', 'call', 'print', 'log'},
                              np_invalid={'ignore', 'warn', 'raise', 'call', 'print', 'log'},
                              )

        self.config.check()
        _config_numpy(seed=self.config.seed,
                      divide=self.config.np_divide,
                      invalid=self.config.np_invalid,
                      )

        self.exist = ExistingModels()

        self.files = FileMan(case=case, **self.options)    # file path manager
