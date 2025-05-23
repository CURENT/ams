"""
Module for system.
"""
import importlib
import inspect
import logging
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np

from andes.system import System as adSystem
from andes.system import (_config_numpy, load_config_rc)
from andes.variables import FileMan

from andes.utils.misc import elapsed
from andes.utils.tab import Tab

import ams
from ams.models.group import GroupBase
from ams.routines.type import TypeBase
from ams.models import file_classes
from ams.routines import all_routines
from ams.utils.paths import get_config_path
from ams.core import Config
from ams.core.matprocessor import MatProcessor
from ams.interface import to_andes
from ams.report import Report
from ams.shared import ad_dyn_models

from ams.io.matpower import system2mpc
from ams.io.matpower import write as wrtite_m
from ams.io.xlsx import write as write_xlsx
from ams.io.json import write as write_json
from ams.io.psse import write_raw

logger = logging.getLogger(__name__)


def disable_method(func):
    def wrapper(*args, **kwargs):
        logger.warning("This method is included in ANDES but not supported in AMS.")
        return None
    return wrapper


def disable_methods(methods):
    for method in methods:
        setattr(System, method, disable_method(getattr(System, method)))


class System(adSystem):
    """
    A subclass of ``andes.system.System``, this class encapsulates data, models,
    and routines for scheduling modeling and analysis in power systems.
    Some methods  inherited from the parent class are intentionally disabled.

    Parameters
    ----------
    case : str, optional
        The path to the case file.
    name : str, optional
        Name of the system instance.
    config : dict, optional
        Configuration options for the system. Overrides the default configuration if provided.
    config_path : str, optional
        The path to the configuration file.
    default_config : bool, optional
        If True, the default configuration file is loaded.
    options : dict, optional
        Additional configuration options for the system.
    **kwargs :
        Additional configuration options passed as keyword arguments.

    Attributes
    ----------
    name : str
        Name of the system instance.
    options : dict
        A dictionary containing configuration options for the system.
    models : OrderedDict
        An ordered dictionary holding the model names and instances.
    model_aliases : OrderedDict
        An ordered dictionary holding model aliases and their corresponding instances.
    groups : OrderedDict
        An ordered dictionary holding group names and instances.
    routines : OrderedDict
        An ordered dictionary holding routine names and instances.
    types : OrderedDict
        An ordered dictionary holding type names and instances.
    mats : MatrixProcessor, None
        A matrix processor instance, initially set to None.
    mat : OrderedDict
        An ordered dictionary holding common matrices.
    exit_code : int
        Command-line exit code. 0 indicates normal execution, while other values indicate errors.
    recent : RecentSolvedRoutines, None
        An object storing recently solved routines, initially set to None.
    dyn : ANDES System, None
        linked dynamic system, initially set to None.
        It is an instance of the ANDES system, which will be automatically
        set when using ``System.to_andes()``.
    files : FileMan
        File path manager instance.
    is_setup : bool
        Internal flag indicating if the system has been set up.

    Methods
    -------
    setup:
        Set up the system.
    to_andes:
        Convert the system to an ANDES system.
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

        # TODO: might need _check_group_common
        func_to_disable = [
            # --- not sure ---
            'set_config', 'set_dae_names', 'set_output_subidx', 'set_var_arrays',
            # --- not used in AMS ---
            '_check_group_common', '_clear_adder_setter', '_e_to_dae', '_expand_pycode', '_finalize_pycode',
            '_find_stale_models', '_get_models', '_init_numba', '_load_calls', '_mp_prepare',
            '_p_restore', '_store_calls', '_store_tf', '_to_orddct', '_v_to_dae',
            'save_config', 'collect_config', 'e_clear', 'f_update',
            'fg_to_dae', 'from_ipysheet', 'g_islands', 'g_update', 'get_z',
            'init', 'j_islands', 'j_update', 'l_update_eq',
            'l_update_var', 'precompile', 'prepare', 'reload', 'remove_pycapsule',
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
        self.types = OrderedDict()           # type names and instances
        self.mats = MatProcessor(self)       # matrix processor
        # TODO: there should be an exit_code for each routine
        self.exit_code = 0                   # command-line exit code, 0 - normal, others - error.
        self.recent = None                   # recent solved routines
        self.dyn = None                      # ANDES system

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

        self.import_types()
        self.import_groups()
        self.import_models()
        self.import_routines()

    def import_types(self):
        """
        Import all types classes defined in ``routines/type.py``.

        Types will be stored as instances with the name as class names.
        All types will be stored to dictionary ``System.types``.
        """
        module = importlib.import_module('ams.routines.type')
        for m in inspect.getmembers(module, inspect.isclass):
            name, cls = m
            if name == 'TypeBase':
                continue
            elif not issubclass(cls, TypeBase):
                # skip other imported classes such as `OrderedDict`
                continue

            self.__dict__[name] = cls()
            self.types[name] = self.__dict__[name]

    def _collect_group_data(self, items):
        """
        Set the owner for routine attributes: `RParam`, `Var`, `ExpressionCalc`, `Expression`,
        and `RBaseService`.
        """
        for item_name, item in items.items():
            if item.model is None:
                continue
            elif item.model in self.groups.keys():
                item.is_group = True
                item.owner = self.groups[item.model]
            elif item.model in self.models.keys():
                item.owner = self.models[item.model]
            elif item.model == 'mats':
                item.owner = self.mats
            else:
                logger.debug(f'item_name: {item_name}')
                msg = f'Model indicator \'{item.model}\' of <{item.rtn.class_name}.{item_name}>'
                msg += ' is not a model or group. Likely a modeling error.'
                logger.warning(msg)

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
                for vname, rtn in self.routines.items():
                    # TODO: collect routiens into types
                    type_name = getattr(rtn, 'type')
                    type_instance = self.types[type_name]
                    type_instance.routines[vname] = rtn
                    # self.types[rtn.type].routines[vname] = rtn
                    # Collect RParams
                    rparams = getattr(rtn, 'rparams')
                    self._collect_group_data(rparams)
                    # Collect routine Vars
                    r_vars = getattr(rtn, 'vars')
                    self._collect_group_data(r_vars)
                    # Collect ExpressionCalcs
                    exprc = getattr(rtn, 'exprcs')
                    self._collect_group_data(exprc)
                    # Collect Expressions
                    expr = getattr(rtn, 'exprs')
                    self._collect_group_data(expr)

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

    def collect_ref(self):
        """
        Collect indices into `BackRef` for all models.
        """
        models_and_groups = list(self.models.values()) + list(self.groups.values())

        # create an empty list of lists for all `BackRef` instances
        for model in models_and_groups:
            for ref in model.services_ref.values():
                ref.v = [list() for _ in range(model.n)]

        # `model` is the model who stores `IdxParam`s to other models
        # `BackRef` is declared at other models specified by the `model` parameter
        # of `IdxParam`s.

        for model in models_and_groups:
            if model.n == 0:
                continue

            # skip: a group is not allowed to link to other groups
            if not hasattr(model, "idx_params"):
                continue

            for idxp in model.idx_params.values():
                if (idxp.model not in self.models) and (idxp.model not in self.groups):
                    continue
                dest = self.__dict__[idxp.model]

                if dest.n == 0:
                    continue

                for name in (model.class_name, model.group):
                    # `BackRef` not requested by the linked models or groups
                    if name not in dest.services_ref:
                        continue

                    for model_idx, dest_idx in zip(model.idx.v, idxp.v):
                        if dest_idx not in dest.uid:
                            continue

                        dest.set_backref(name,
                                         from_idx=model_idx,
                                         to_idx=dest_idx)

    def reset(self, force=False):
        """
        Reset to the state after reading data and setup.
        """
        self.is_setup = False
        self.setup()

    def add(self, model, param_dict=None, **kwargs):
        """
        Add a device instance for an existing model.

        Revised from ``andes.system.System.add()``.
        """
        if model not in self.models and (model not in self.model_aliases):
            if model in ad_dyn_models:
                logger.debug("ANDES dynamic model <%s> is skipped.", model)
            else:
                logger.warning("<%s> is not an existing model.", model)
            return

        if self.is_setup:
            raise NotImplementedError("Adding devices are not allowed after setup.")

        group_name = self.__dict__[model].group
        group = self.groups[group_name]

        if param_dict is None:
            param_dict = {}
        if kwargs is not None:
            param_dict.update(kwargs)

        # remove `uid` field
        param_dict.pop('uid', None)

        idx = param_dict.pop('idx', None)
        if idx is not None and (not isinstance(idx, str) and np.isnan(idx)):
            idx = None

        idx = group.get_next_idx(idx=idx, model_name=model)
        self.__dict__[model].add(idx=idx, **param_dict)
        group.add(idx=idx, model=self.__dict__[model])

        return idx

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

        self.collect_ref()
        self._list2array()     # `list2array` must come before `link_ext_param`
        if not self.link_ext_param():
            ret = False

        # --- model parameters range check ---
        # TODO: there might be other parameters check?
        adjusted_params = []
        param_to_check = ['rate_a', 'rate_b', 'rate_c', 'amax', 'amin']
        for pname in param_to_check:
            param = self.Line.params[pname]
            if np.any(param.v == 0):
                adjusted_params.append(pname)
                param.v[param.v == 0] = param.default
        if adjusted_params:
            adjusted_params_str = ', '.join(adjusted_params)
            msg = f"Zero Line parameters detected, adjusted to default values: {adjusted_params_str}."
            logger.info(msg)
        # --- bus type correction ---
        pq_bus = self.PQ.bus.v
        pv_bus = self.PV.bus.v
        slack_bus = self.Slack.bus.v
        # TODO: how to include islanded buses?
        if self.Bus.n > 0 and np.all(self.Bus.type.v == 1):
            self.Bus.alter(src='type', idx=pq_bus, value=1)
            self.Bus.alter(src='type', idx=pv_bus, value=2)
            self.Bus.alter(src='type', idx=slack_bus, value=3)
            logger.info("All bus type are PQ, adjusted given load and generator connection status.")
        # === no device addition or removal after this point ===
        self.calc_pu_coeff()   # calculate parameters in system per units

        if ret is True:
            self.is_setup = True  # set `is_setup` if no error occurred
        else:
            logger.error("System setup failed. Please resolve the reported issue(s).")
            self.exit_code += 1

        a0 = 0
        for _, mdl in self.models.items():
            for _, algeb in mdl.algebs.items():
                algeb.v = np.zeros(algeb.owner.n)
                algeb.a = np.arange(a0, a0 + algeb.owner.n)
                a0 += algeb.owner.n

        # NOTE: this is a temporary solution for building Y matrix
        # consider refator this part if any other similar cases occur in the future
        self.Line.a1a = self.Bus.get(src='a', attr='a', idx=self.Line.bus1.v)
        self.Line.a2a = self.Bus.get(src='a', attr='a', idx=self.Line.bus2.v)

        # assign bus type as placeholder; 1=PQ, 2=PV, 3=ref, 4=isolated
        if self.Bus.type.v.sum() == self.Bus.n:  # if all type are PQ
            self.Bus.alter(src='type', idx=self.PV.bus.v,
                           value=np.ones(self.PV.n))
            self.Bus.alter(src='type', idx=self.Slack.bus.v,
                           value=np.ones(self.Slack.n))

        # --- assign column and row names ---
        self.mats.Cft.col_names = self.Line.idx.v
        self.mats.Cft.row_names = self.Bus.idx.v

        self.mats.CftT.col_names = self.Bus.idx.v
        self.mats.CftT.row_names = self.Line.idx.v

        self.mats.Cg.col_names = self.StaticGen.get_all_idxes()
        self.mats.Cg.row_names = self.Bus.idx.v

        self.mats.Cl.col_names = self.PQ.idx.v
        self.mats.Cl.row_names = self.Bus.idx.v

        self.mats.Csh.col_names = self.Shunt.idx.v
        self.mats.Csh.row_names = self.Bus.idx.v

        self.mats.Bbus.col_names = self.Bus.idx.v
        self.mats.Bbus.row_names = self.Bus.idx.v

        self.mats.Bf.col_names = self.Bus.idx.v
        self.mats.Bf.row_names = self.Line.idx.v

        self.mats.PTDF.col_names = self.Bus.idx.v
        self.mats.PTDF.row_names = self.Line.idx.v

        self.mats.LODF.col_names = self.Line.idx.v
        self.mats.LODF.row_names = self.Line.idx.v

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

    def supported_routines(self, export='plain'):
        """
        Return the support type names and routine names in a table.

        Returns
        -------
        str
            A table-formatted string for the types and routines
        """

        def rst_ref(name, export):
            """
            Refer to the model in restructuredText mode so that
            it renders as a hyperlink.
            """

            if export == 'rest':
                return ":ref:`" + name + '`'
            else:
                return name

        pairs = list()
        for g in self.types:
            routines = list()
            for m in self.types[g].routines:
                routines.append(rst_ref(m, export))
            if len(routines) > 0:
                pairs.append((rst_ref(g, export), ', '.join(routines)))

        tab = Tab(title='Supported Types and Routines',
                  header=['Type', 'Routines'],
                  data=pairs,
                  export=export,
                  )

        return tab.draw()

    def connectivity(self, info=True):
        """
        Perform connectivity check for system.

        Parameters
        ----------
        info : bool
            True to log connectivity summary.
        """

        raise NotImplementedError

    def to_andes(self, addfile=None,
                 setup=False, no_output=False,
                 default_config=True,
                 verify=False, tol=1e-3,
                 **kwargs):
        """
        Convert the AMS system to an ANDES system.

        A preferred dynamic system file to be added has following features:
        1. The file contains both power flow and dynamic models.
        2. The file can run in ANDES natively.
        3. Power flow models are in the same shape as the AMS system.
        4. Dynamic models, if any, are in the same shape as the AMS system.

        This function is wrapped as the ``System`` class method ``to_andes()``.
        Using the file conversion ``to_andes()`` will automatically
        link the AMS system instance to the converted ANDES system instance
        in the AMS system attribute ``dyn``.

        It should be noted that detailed dynamic simualtion requires extra
        dynamic models to be added to the ANDES system, which can be passed
        through the ``addfile`` argument.

        Parameters
        ----------
        system : System
            The AMS system to be converted to ANDES format.
        addfile : str, optional
            The additional file to be converted to ANDES dynamic mdoels.
        setup : bool, optional
            Whether to call `setup()` after the conversion. Default is True.
        no_output : bool, optional
            To ANDES system.
        default_config : bool, optional
            To ANDES system.
        verify : bool
            If True, the converted ANDES system will be verified with the source
            AMS system using AC power flow.
        tol : float
            The tolerance of error.

        Returns
        -------
        adsys : andes.system.System
            The converted ANDES system.

        Examples
        --------
        >>> import ams
        >>> import andes
        >>> sp = ams.load(ams.get_case('ieee14/ieee14_uced.xlsx'), setup=True)
        >>> sa = sp.to_andes(addfile=andes.get_case('ieee14/ieee14_full.xlsx'),
        ...                  setup=False, overwrite=True, no_output=True)

        Notes
        -----
        1. Power flow models in the addfile will be skipped and only dynamic models will be used.
        2. The addfile format is guessed based on the file extension. Currently only ``xlsx`` is supported.
        3. Index in the addfile is automatically adjusted when necessary.
        """
        return to_andes(system=self, addfile=addfile,
                        setup=setup, no_output=no_output,
                        default_config=default_config,
                        verify=verify, tol=tol,
                        **kwargs)

    def summary(self):
        """
        Print out system summary.
        """
        # FIXME: add system connectivity check
        # logger.info("-> System connectivity check results:")
        rtn_check = OrderedDict((key, val._data_check(info=False)) for key, val in self.routines.items())
        rtn_types = OrderedDict({tp: [] for tp in self.types.keys()})

        for name, data_pass in rtn_check.items():
            if data_pass:
                r_type = self.routines[name].type
                rtn_types[r_type].append(name)

        nb = self.Bus.n
        nl = self.Line.n
        ng = self.StaticGen.n

        pd = self.PQ.p0.v.sum()
        qd = self.PQ.q0.v.sum()

        out = list()

        out.append("-> Systen size:")
        out.append(f"Base: {self.config.mva} MVA; Frequency: {self.config.freq} Hz")
        out.append(f"{nb} Buses; {nl} Lines; {ng} Static Generators")
        out.append(f"Active load: {pd:,.2f} p.u.; Reactive load: {qd:,.2f} p.u.")

        out.append("-> Data check results:")
        for rtn_type, names in rtn_types.items():
            if len(names) == 0:
                continue
            names = ", ".join(names)
            out.append(f"{rtn_type}: {names}")

        out_str = '\n'.join(out)
        logger.info(out_str)

    def report(self):
        """
        Write system routine reports to a plain-text file.

        Returns
        -------
        bool
            True if the report is written successfully.
        """
        if self.files.no_output is False:
            r = Report(self)
            r.write()
            return True

        return False

    def to_mpc(self):
        """
        Export an AMS system to a MATPOWER dict.

        Returns
        -------
        dict
            A dictionary representing the MATPOWER case.
        """
        return system2mpc(self)

    def to_m(self, outfile: str, overwrite: bool = None):
        """
        Export an AMS system to a MATPOWER M-file.

        Parameters
        ----------
        outfile : str
            The output file name.
        overwrite : bool, optional
            If True, overwrite the existing file. Default is None.
        """
        return wrtite_m(self, outfile=outfile, overwrite=overwrite)

    def to_xlsx(self, outfile: str, overwrite: bool = None):
        """
        Export an AMS system to an Excel file.

        Parameters
        ----------
        outfile : str
            The output file name.
        overwrite : bool, optional
            If True, overwrite the existing file. Default is None.
        """
        return write_xlsx(self, outfile=outfile, overwrite=overwrite)

    def to_json(self, outfile: str, overwrite: bool = None):
        """
        Export an AMS system to a JSON file.

        Parameters
        ----------
        outfile : str
            The output file name.
        overwrite : bool, optional
            If True, overwrite the existing file. Default is None.
        """
        return write_json(self, outfile=outfile, overwrite=overwrite)

    def to_raw(self, outfile: str, overwrite: bool = None):
        """
        Export an AMS system to a v33 PSS/E RAW file.

        Parameters
        ----------
        outfile : str
            The output file name.
        overwrite : bool, optional
            If True, overwrite the existing file. Default is None.
        """
        return write_raw(self, outfile=outfile, overwrite=overwrite)

# --------------- Helper Functions ---------------
# NOTE: _config_numpy, load_config_rc are imported from andes.system


def example(setup=True, no_output=True, **kwargs):
    """
    Return an :py:class:`ams.system.System` object for the
    ``ieee14_uced.xlsx`` as an example.

    This function is useful when a user wants to quickly get a
    System object for testing.

    Returns
    -------
    System
        An example :py:class:`ams.system.System` object.
    """

    return ams.load(ams.get_case('matpower/case14.m'),
                    setup=setup, no_output=no_output, **kwargs)
