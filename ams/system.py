"""
System class for power system data, methods, and routines.
"""
import configparser
import importlib
import inspect
import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

from andes.core import Config
from andes.system import ExistingModels as andes_ExistingModels
from andes.system import System as andes_System
from andes.system import (_config_numpy, load_config_rc)
from ams.models.group import GroupBase
from andes.variables import FileMan

from ams.utils.paths import (ams_root, get_config_path)
import ams.io
from ams.models import file_classes

logger = logging.getLogger(__name__)


def disable_method(func):
    def wrapper(*args, **kwargs):
        logger.warning(f"Method `{func.__name__}` is an ANDES System method but not supported in AMS System.")
        return None
    return wrapper


def disable_methods(methods):
    for method in methods:
        setattr(System, method, disable_method(getattr(System, method)))


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
        self.models = OrderedDict()          # model names and instances
        self.model_aliases = OrderedDict()   # alias: model instance
        self.groups = OrderedDict()          # group names and instances
        self.routines = OrderedDict()        # routine names and instances
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

        # TODO: revise the following attributes
        self._getters = dict(f=list(), g=list(), x=list(), y=list())
        self._adders = dict(f=list(), g=list(), x=list(), y=list())
        self._setters = dict(f=list(), g=list(), x=list(), y=list())

        self.files = FileMan(case=case, **self.options)    # file path manager

        # TODO: DEBUG now
        # self.import_models()

        # Disable methods not supported in AMS
        # TODO: some of the methods might be used in the future
        func_to_disable = [
            # --- not sure ---
            'set_config', 'set_dae_names', 'set_output_subidx', 'set_var_arrays',
            # --- not used in AMS ---
            '_check_group_common', '_clear_adder_setter', '_e_to_dae', '_expand_pycode', '_finalize_pycode',
            '_find_stale_models', '_get_models', '_init_numba', '_list2array', '_load_calls', '_mp_prepare',
            '_p_restore', '_store_calls', '_store_tf', '_to_orddct', '_update_config_object', '_v_to_dae',
            'call_models', 'save_config', 'collect_config', 'collect_ref', 'e_clear', 'f_update',
            'fg_to_dae', 'find_devices', 'find_models', 'from_ipysheet', 'g_islands', 'g_update', 'get_z',
            'import_routines', 'init', 'j_islands', 'j_update', 'l_update_eq',
            'l_update_var', 'link_ext_param', 'precompile', 'prepare', 'reload', 'remove_pycapsule', 'reset',
            's_update_post', 's_update_var', 'setup', 'store_adder_setter', 'store_existing', 'store_no_check_init',
            'store_sparse_pattern', 'store_switch_times', 'supported_models', 'switch_action', 'to_ipysheet',
            'undill']
        disable_methods(func_to_disable)

        func_to_revise = ['set_address', 'vars_to_dae', 'vars_to_models']
        # TODO: ``set_address``: exclude state variables
        # TODO: ``vars_to_dae``: switch from dae to ie
        # TODO: ``vars_to_models``: switch from dae to ie

    def summary(self):
        """
        Print out system summary.
        """

        island_sets = self.Bus.island_sets
        nosw_island = self.Bus.nosw_island
        msw_island = self.Bus.msw_island
        n_islanded_buses = self.Bus.n_islanded_buses

        logger.info("-> System connectivity check results:")
        if n_islanded_buses == 0:
            logger.info("  No islanded bus detected.")
        else:
            logger.info("  %d islanded bus detected.", n_islanded_buses)
            logger.debug("  Islanded Bus indices (0-based): %s", self.Bus.islanded_buses)

        if len(island_sets) == 0:
            logger.info("  No island detected.")
        elif len(island_sets) == 1:
            logger.info("  System is interconnected.")
            logger.debug("  Bus indices in interconnected system (0-based): %s", island_sets)
        else:
            logger.info("  System contains %d island(s).", len(island_sets))
            logger.debug("  Bus indices in islanded areas (0-based): %s", island_sets)

        if len(nosw_island) > 0:
            logger.warning('  Slack generator is not defined/enabled for %d island(s).',
                           len(nosw_island))
            logger.debug("  Bus indices in no-Slack areas (0-based): %s",
                         [island_sets[item] for item in nosw_island])

        if len(msw_island) > 0:
            logger.warning('  Multiple slack generators are defined/enabled for %d island(s).',
                           len(msw_island))
            logger.debug("  Bus indices in multiple-Slack areas (0-based): %s",
                         [island_sets[item] for item in msw_island])

        if len(self.Bus.nosw_island) == 0 and len(self.Bus.msw_island) == 0:
            logger.info('  Each island has a slack bus correctly defined and enabled.')

    def import_groups(self):
        module = importlib.import_module('ams.models.group')

        for m in inspect.getmembers(module, inspect.isclass):

            name, cls = m
            if name == 'GroupBase':
                continue
            elif not issubclass(cls, GroupBase):
                # skip other imported classes such as `OrderedDict`
                continue

            self.__dict__[name] = cls()
            self.groups[name] = self.__dict__[name]
