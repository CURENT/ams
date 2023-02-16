"""
System class for power system data, methods, and routines.
"""
import configparser
import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

from andes.core import Config
from andes.system import ExistingModels as andes_ExistingModels
from andes.system import (_config_numpy, load_config_rc)

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


class System:
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

    def _update_config_object(self):
        """
        Change config on the fly based on command-line options.

        Copy from ``andes.utils.paths._update_config_object``.
        """

        config_option = self.options.get('config_option', None)
        if config_option is None:
            return

        if len(config_option) == 0:
            return

        newobj = False
        if self._config_object is None:
            self._config_object = configparser.ConfigParser()
            newobj = True

        for item in config_option:

            # check the validity of the config field
            # each field follows the format `SECTION.FIELD = VALUE`

            if item.count('=') != 1:
                raise ValueError('config_option "{}" must be an assignment expression'.format(item))

            field, value = item.split("=")

            if field.count('.') != 1:
                raise ValueError('config_option left-hand side "{}" must use format SECTION.FIELD'.format(field))

            section, key = field.split(".")

            section = section.strip()
            key = key.strip()
            value = value.strip()

            if not newobj:
                self._config_object.set(section, key, value)
                logger.debug("Existing config option set: %s.%s=%s", section, key, value)
            else:
                self._config_object.add_section(section)
                self._config_object.set(section, key, value)
                logger.debug("New config option added: %s.%s=%s", section, key, value)
