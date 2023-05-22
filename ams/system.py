"""
Module for system.
"""
import configparser
import copy
import importlib
import inspect
import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import sympy as sp

from andes.core import Config
from andes.system import System as andes_System
from andes.system import (_config_numpy, load_config_rc)
from andes.variables import FileMan

from andes.utils.misc import elapsed

from ams.models.group import GroupBase
from ams.models import file_classes
from ams.routines import all_routines, algeb_models
from ams.utils.paths import get_config_path
from ams.core import Algeb
from ams.core.matprocessor import MatProcessor

logger = logging.getLogger(__name__)


def disable_method(func):
    def wrapper(*args, **kwargs):
        logger.warning(f"Method `{func.__name__}` is included in ANDES System but not supported in AMS System.")
        return None
    return wrapper


def disable_methods(methods):
    for method in methods:
        setattr(System, method, disable_method(getattr(System, method)))


class System(andes_System):
    """
    System contains data, models, and routines for dispatch modeling and analysis.

    This class is a subclass of ``andes.system.System``.
    Some methods inherited from ``andes.system.System`` are disabled but remain in the class for now.
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

        func_to_disable = [
            # --- not sure ---
            'set_config', 'set_dae_names', 'set_output_subidx', 'set_var_arrays',
            # --- not used in AMS ---
            '_check_group_common', '_clear_adder_setter', '_e_to_dae', '_expand_pycode', '_finalize_pycode',
            '_find_stale_models', '_get_models', '_init_numba', '_load_calls', '_mp_prepare',
            '_p_restore', '_store_calls', '_store_tf', '_to_orddct', '_v_to_dae',
            'save_config', 'collect_config', 'collect_ref', 'e_clear', 'f_update',
            'fg_to_dae', 'from_ipysheet', 'g_islands', 'g_update', 'get_z',
            'init', 'j_islands', 'j_update', 'l_update_eq', 'connectivity', 'summary',
            'l_update_var', 'link_ext_param', 'precompile', 'prepare', 'reload', 'remove_pycapsule', 'reset',
            's_update_post', 's_update_var', 'store_adder_setter', 'store_no_check_init',
            'store_sparse_pattern', 'store_switch_times', 'switch_action', 'to_ipysheet',
            'undill']
        disable_methods(func_to_disable)

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
        self.mats = None                     # matrix processor
        self.mat = OrderedDict()             # common matrices
        # TODO: there should be an exit_code for each routine
        self.exit_code = 0                   # command-line exit code, 0 - normal, others - error.

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
                                     ('save_stats', 0),  # TODO: not sure what this is for
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

        # TODO: revise the following attributes, it seems that these are not used in AMS
        self._getters = dict(f=list(), g=list(), x=list(), y=list())
        self._adders = dict(f=list(), g=list(), x=list(), y=list())
        self._setters = dict(f=list(), g=list(), x=list(), y=list())

        self.files = FileMan(case=case, **self.options)    # file path manager

        # internal flags
        self.is_setup = False        # if system has been setup

        self.import_groups()
        self.import_models()
        self.import_routines()

    def summarize_groups(self):
        """
        Summarize groups and their models.
        """
        pass
        for gname, grp in self.groups.items():
            grp.summarize()

    def _collect_group_data(self, items):
        for item_name, item in items.items():
            # logger.debug(f'RVar {item_name} has owner {item.owner_name} in size {item.owner.n}')
            if item.owner_name in self.groups.keys():
                item.is_group = True
                item.owner = self.groups[item.owner_name]
            elif item.owner_name in self.models.keys():
                item.owner = self.models[item.owner_name]

    def import_routines(self):
        """
        Import routines as defined in ``routines/__init__.py``.

        Routines will be stored as instances with the name as class names.
        All routines will be stored to dictionary ``System.routines``.

        Examples
        --------
        ``System.PFlow`` is the power flow routine instance.
        """
        for file, cls_list in all_routines.items():
            for cls_name in cls_list:
                routine = importlib.import_module('ams.routines.' + file)
                the_class = getattr(routine, cls_name)
                attr_name = cls_name
                self.__dict__[attr_name] = the_class(system=self, config=self._config_object)
                self.routines[attr_name] = self.__dict__[attr_name]
                self.routines[attr_name].config.check()
                # NOTE: the following code is not used in ANDES
                for rname, rtn in self.routines.items():
                    # Collect rparams
                    rparams = getattr(rtn, 'rparams')
                    self._collect_group_data(rparams)
                    # Collect ralgebs
                    ralgebs = getattr(rtn, 'ralgebs')
                    self._collect_group_data(ralgebs)
                    # setup numerical optimziation model
                    # TODO: substitute symbolic expressions with numerical ones

    def import_groups(self):
        """
        Import all groups classes defined in ``models/group.py``.

        Groups will be stored as instances with the name as class names.
        All groups will be stored to dictionary ``System.groups``.
        """
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

    def import_models(self):
        """
        Import and instantiate models as System member attributes.

        Models defined in ``models/__init__.py`` will be instantiated `sequentially` as attributes with the same
        name as the class name.
        In addition, all models will be stored in dictionary ``System.models`` with model names as
        keys and the corresponding instances as values.

        Examples
        --------
        ``system.Bus`` stores the `Bus` object, and ``system.PV`` stores the PV generator object.

        ``system.models['Bus']`` points the same instance as ``system.Bus``.
        """
        for fname, cls_list in file_classes:
            for model_name in cls_list:
                the_module = importlib.import_module('ams.models.' + fname)
                the_class = getattr(the_module, model_name)
                self.__dict__[model_name] = the_class(system=self, config=self._config_object)
                self.models[model_name] = self.__dict__[model_name]
                self.models[model_name].config.check()

                # link to the group
                group_name = self.__dict__[model_name].group
                self.__dict__[group_name].add_model(model_name, self.__dict__[model_name])
        # NOTE: model_aliases is not used in AMS currently
        # for key, val in ams.models.model_aliases.items():
        #     self.model_aliases[key] = self.models[val]
        #     self.__dict__[key] = self.models[val]

        # NOTE: define a special model named ``G`` that combines PV and Slack
        # FIXME: seems hard coded
        # TODO: seems to be a bad design
        gen_cols = self.Slack.as_df().columns

    def setup(self):
        """
        Set up system for studies.

        This function is to be called after adding all device data.
        """
        ret = True
        t0, _ = elapsed()

        if self.is_setup:
            logger.warning('System has been setup. Calling setup twice is not allowed.')
            ret = False
            return ret

        self._list2array()     # `list2array` must come before `link_ext_param`

        if self.Line.rate_a.v.max() == 0:
            logger.warning("Line rate_a is corrected to large value automatically.")
            self.Line.rate_a.v = 99
        # === no device addition or removal after this point ===
        # TODO: double check calc_pu_coeff
        self.calc_pu_coeff()   # calculate parameters in system per units
        # self.store_existing()  # store models with routine flags

        if ret is True:
            self.is_setup = True  # set `is_setup` if no error occurred
        else:
            logger.error("System setup failed. Please resolve the reported issue(s).")
            self.exit_code += 1

        a0 = 0
        for mname, mdl in self.models.items():
            for aname, algeb in mdl.algebs.items():
                algeb.v = np.zeros(algeb.owner.n)
                algeb.a = np.arange(a0, a0 + algeb.owner.n)
                a0 += algeb.owner.n

        # set up common matrix
        self.mats = MatProcessor(self)       # matrix processor

        # FIXME: hard coded here
        # Set nuemerical values for special params
        gen_bus = self.StaticGen.get(src='bus', attr='v',
                                     idx=self.StaticGen.get_idx())
        all_bus = self.Bus.idx.v
        regBus = [int(bus) if isinstance(bus, (int, float)) else bus for bus in gen_bus]
        redBus = [int(bus) if isinstance(bus, (int, float)) else bus for bus in all_bus if bus not in gen_bus]

        # Restrucrue PQ load value to match gen bus pattern
        idx_PD1 = self.PQ.find_idx(keys="bus", values=regBus, allow_none=True, default=None)
        idx_PD2 = self.PQ.find_idx(keys="bus", values=redBus, allow_none=True, default=None)
        PD1 = self.PQ.get(src='p0', attr='v', idx=idx_PD1)
        PD2 = self.PQ.get(src='p0', attr='v', idx=idx_PD2)
        PTDF1, PTDF2 = self.mats.rePTDF()

        self.mat = OrderedDict([
            ('pd1', PD1), ('pd2', PD2),
            ('PTDF1', PTDF1), ('PTDF2', PTDF2),
        ])

        # NOTE: Set up om for all routines
        for rname, rtn in self.routines.items():
            # rtn.setup()  # not setup optimization model in system setup stage
            a0 = 0
            for raname, ralgeb in rtn.ralgebs.items():
                ralgeb.v = np.zeros(ralgeb.owner.n)
                ralgeb.a = np.arange(a0, a0 + ralgeb.owner.n)
                a0 += ralgeb.owner.n
            for rpname, rparam in rtn.rparams.items():
                if rpname in self.mat.keys():
                    rparam.is_set = True
                    rparam._v = self.mat[rpname]
            # TODO: maybe setup numrical arrays here? [rtn.c, Aub, Aeq ...]

        _, s = elapsed(t0)
        logger.info('System set up in %s.', s)

        return ret

    # FIXME: remove unused methods
    # # Disable methods not supported in AMS
    # func_to_include = [
    #     'import_models', 'import_groups', 'import_routines',
    #     'setup', 'init_algebs',
    #     '_update_config_object',
    #     ]
    # # disable_methods(func_to_disable)
    # __dict__ = {method: lambda self: self.x for method in func_to_include}
