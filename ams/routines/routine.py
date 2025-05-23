"""
Module for routine data.
"""

import logging
import os
from typing import Optional, Union, Type, Iterable, Dict
from collections import OrderedDict

import numpy as np

from andes.utils.misc import elapsed

from ams.core import Config
from ams.core.param import RParam
from ams.core.symprocessor import SymProcessor
from ams.core.documenter import RDocumenter
from ams.core.service import RBaseService, ValueService
from ams.opt import OModel
from ams.opt import Param, Var, Constraint, Objective, ExpressionCalc, Expression

from ams.shared import pd

logger = logging.getLogger(__name__)


class RoutineBase:
    """
    Class to hold descriptive routine models and data mapping.

    Attributes
    ----------
    system : Optional[Type]
        The system object associated with the routine.
    config : Config
        Configuration object for the routine.
    info : Optional[str]
        Information about the routine.
    tex_names : OrderedDict
        LaTeX names for the routine parameters.
    syms : SymProcessor
        Symbolic processor for the routine.
    _syms : bool
        Flag indicating whether symbols have been generated.
    rparams : OrderedDict
        Registry for RParam objects.
    services : OrderedDict
        Registry for service objects.
    params : OrderedDict
        Registry for Param objects.
    vars : OrderedDict
        Registry for Var objects.
    constrs : OrderedDict
        Registry for Constraint objects.
    exprcs : OrderedDict
        Registry for ExpressionCalc objects.
    exprs : OrderedDict
        Registry for Expression objects.
    obj : Optional[Objective]
        Objective of the routine.
    initialized : bool
        Flag indicating whether the routine has been initialized.
    type : str
        Type of the routine.
    docum : RDocumenter
        Documentation generator for the routine.
    map1 : OrderedDict
        Mapping from ANDES.
    map2 : OrderedDict
        Mapping to ANDES.
    om : OModel
        Optimization model for the routine.
    exec_time : float
        Execution time of the routine.
    exit_code : int
        Exit code of the routine.
    converged : bool
        Flag indicating whether the routine has converged.
    converted : bool
        Flag indicating whether AC conversion has been performed.
    """

    def __init__(self, system=None, config=None):
        """
        Initialize the routine.

        Parameters
        ----------
        system : Optional[Type]
            The system object associated with the routine.
        config : Optional[dict]
            Configuration dictionary for the routine.
        """
        self.system = system
        self.config = Config(self.class_name)
        self.info = None
        self.tex_names = OrderedDict(
            (
                ("sys_f", "f_{sys}"),
                ("sys_mva", "S_{b,sys}"),
            )
        )
        self.syms = SymProcessor(self)      # symbolic processor
        self._syms = False                  # symbol generation flag

        self.rparams = OrderedDict()        # RParam registry
        self.services = OrderedDict()       # Service registry
        self.params = OrderedDict()         # Param registry
        self.vars = OrderedDict()           # Var registry
        self.constrs = OrderedDict()        # Constraint registry
        self.exprcs = OrderedDict()         # ExpressionCalc registry
        self.exprs = OrderedDict()          # Expression registry
        self.obj = None                     # Objective
        self.initialized = False            # initialization flag
        self.type = "UndefinedType"         # routine type
        self.docum = RDocumenter(self)      # documentation generator

        # --- sync mapping ---
        self.map1 = OrderedDict()  # from ANDES
        self.map2 = OrderedDict()  # to ANDES

        # --- optimization modeling ---
        self.om = OModel(routine=self)      # optimization model

        if config is not None:
            self.config.load(config)

        # NOTE: the difference between exit_code and converged is that
        # exit_code is the solver exit code, while converged is the
        # convergence flag of the routine.
        self.exec_time = 0.0        # running time
        self.exit_code = 0          # exit code
        self.converged = False      # convergence flag
        self.converted = False          # AC conversion flag

    @property
    def class_name(self):
        return self.__class__.__name__

    def get(self, src: str, idx, attr: str = 'v',
            horizon: Optional[Union[int, str, Iterable]] = None):
        """
        Get the value of a variable or parameter.

        Parameters
        ----------
        src: str
            Name of the variable or parameter.
        idx: int, str, or list
            Index of the variable or parameter.
        attr: str
            Attribute name.
        horizon: list, optional
            Horizon index.
        """
        if src not in self.__dict__.keys():
            raise ValueError(f"<{src}> does not exist in <<{self.class_name}>.")
        item = self.__dict__[src]

        if not hasattr(item, attr):
            raise ValueError(f"{attr} does not exist in {self.class_name}.{src}.")

        idx_all = item.get_all_idxes()

        if idx_all is None:
            raise ValueError(f"<{self.class_name}> item <{src}> has no idx.")

        is_format = False  # whether the idx is formatted as a list
        idx_u = None
        if isinstance(idx, (str, int)):
            idx_u = [idx]
            is_format = True
        elif isinstance(idx, (np.ndarray, pd.Series)):
            idx_u = idx.tolist()
        elif isinstance(idx, list):
            idx_u = idx.copy()

        loc = [idx_all.index(idxe) if idxe in idx_all else None for idxe in idx_u]
        if None in loc:
            idx_none = [idxe for idxe in idx_u if idxe not in idx_all]
            msg = f"Var <{self.class_name}.{src}> does not contain value with idx={idx_none}"
            raise ValueError(msg)
        out = getattr(item, attr)[loc]

        if horizon is not None:
            if item.horizon is None:
                raise ValueError(f"horizon is not defined for {self.class_name}.{src}.")
            horizon_all = item.horizon.get_all_idxes()
            if not isinstance(horizon, list):
                raise TypeError(f"horizon must be a list, not {type(horizon)}.")
            loc_h = [
                horizon_all.index(idxe) if idxe in horizon_all else None
                for idxe in horizon
            ]
            if None in loc_h:
                idx_none = [idxe for idxe in horizon if idxe not in horizon_all]
                msg = f"Var <{self.class_name}.{src}> does not contain horizon with idx={idx_none}"
                raise ValueError(msg)
            out = out[:, loc_h]
            if out.shape[1] == 1:
                out = out[:, 0]

        return out[0] if is_format else out

    def set(self, src: str, idx, attr: str = "v", value=0.0):
        """
        Set the value of an attribute of a routine parameter.

        Performs ``self.<src>.<attr>[idx] = value``. This method will not modify
        the input values from the case file that have not been converted to the
        system base. As a result, changes applied by this method will not affect
        the dumped case file.

        To alter parameters and reflect it in the case file, use :meth:`alter`
        instead.

        Parameters
        ----------
        src : str
            Name of the model property
        idx : str, int, float, array-like
            Indices of the devices
        attr : str, optional, default='v'
            The internal attribute of the property to get.
            ``v`` for values, ``a`` for address, and ``e`` for equation value.
        value : array-like
            New values to be set

        Returns
        -------
        bool
            True when successful.
        """
        if self.__dict__[src].owner is not None:
            # TODO: fit to `_v` type param in the future
            owner = self.__dict__[src].owner
            src0 = self.__dict__[src].src
            try:
                res = owner.set(src=src0, idx=idx, attr=attr, value=value)
                return res
            except KeyError as e:
                msg = f"Failed to set <{src0}> in <{owner.class_name}>. "
                msg += f"Original error: {e}"
                raise KeyError(msg)
        else:
            # FIXME: add idx for non-grouped variables
            raise TypeError(f"Variable {self.name} has no owner.")

    def doc(self, max_width=78, export="plain"):
        """
        Retrieve routine documentation as a string.
        """
        return self.docum.get(max_width=max_width, export=export)

    def _get_off_constrs(self):
        """
        Chcek if constraints are turned off.
        """
        disabled = []
        for cname, c in self.constrs.items():
            if c.is_disabled:
                disabled.append(cname)
        if len(disabled) > 0:
            msg = "Disabled constraints: "
            d_str = [f'{constr}' for constr in disabled]
            msg += ", ".join(d_str)
            logger.warning(msg)
        return disabled

    def _data_check(self, info=True):
        """
        Check if data is valid for a routine.

        Parameters
        ----------
        info: bool
            Whether to print warning messages.
        """
        logger.debug(f"Entering data check for <{self.class_name}>")
        no_input = []
        owner_list = []
        for rname, rparam in self.rparams.items():
            if rparam.owner is not None:
                # NOTE: skip checking Shunt.g
                if (rparam.owner.class_name == 'Shunt') and (rparam.src == 'g'):
                    pass
                elif rparam.owner.n == 0:
                    no_input.append(rname)
                    owner_list.append(rparam.owner.class_name)
            # TODO: add more data config check?
            if rparam.config.pos:
                if not np.all(rparam.v > 0):
                    logger.warning(f"RParam <{rname}> should have all positive values.")
        if len(no_input) > 0:
            if info:
                msg = f"Following models are missing in input: {set(owner_list)}"
                logger.error(msg)
            return False
        # TODO: add data validation for RParam, typical range, etc.
        logger.debug(" -> Data check passed")
        return True

    def init(self, **kwargs):
        """
        Initialize the routine.

        Other parameters
        ----------------
        force: bool
            Whether to force initialization regardless of the current initialization status.
        force_mats: bool
            Whether to force build the system matrices, goes to `self.system.mats.build()`.
        force_constr: bool
            Whether to turn on all constraints.
        force_om: bool
            Whether to force initialize the optimization model.
        """
        force = kwargs.pop('force', False)
        force_mats = kwargs.pop('force_mats', False)
        force_constr = kwargs.pop('force_constr', False)
        force_om = kwargs.pop('force_om', False)

        skip_all = not (force and force_mats) and self.initialized and self.om.initialized

        if skip_all:
            logger.debug(f"{self.class_name} has already been initialized.")
            return True

        t0, _ = elapsed()
        # --- data check ---
        self._data_check()

        # --- turn on all constrs ---
        if force_constr:
            for constr in self.constrs.values():
                constr.is_disabled = False

        # --- matrix build ---
        self.system.mats.build(force=force_mats)

        # --- constraint check ---
        _ = self._get_off_constrs()

        if not self.om.initialized:
            self.om.init(force=force_om)
        _, s_init = elapsed(t0)

        msg = f"<{self.class_name}> "
        if self.om.initialized:
            msg += f"initialized in {s_init}."
            self.initialized = True
        else:
            msg += "initialization failed!"
            self.initialized = False
        logger.info(msg)
        return self.initialized

    def solve(self, **kwargs):
        """
        Solve the routine optimization model.
        """
        raise NotImplementedError

    def unpack(self, res, **kwargs):
        """
        Unpack the results.
        """
        raise NotImplementedError

    def _post_solve(self):
        """
        Post-solve calculation.
        """
        raise NotImplementedError

    def run(self, **kwargs):
        """
        Run the routine.
        args and kwargs go to `self.solve()`.

        Force initialization (`force_init=True`) will do the following:
        - Rebuild the system matrices
        - Enable all constraints
        - Reinitialize the optimization model

        Parameters
        ----------
        force_init: bool
            Whether to force re-initialize the routine.
        force_mats: bool
            Whether to force build the system matrices.
        force_constr: bool
            Whether to turn on all constraints.
        force_om: bool
            Whether to force initialize the OModel.
        """
        # --- setup check ---
        force_init = kwargs.pop('force_init', False)
        force_mats = kwargs.pop('force_mats', False)
        force_constr = kwargs.pop('force_constr', False)
        force_om = kwargs.pop('force_om', False)
        self.init(force=force_init, force_mats=force_mats,
                  force_constr=force_constr, force_om=force_om)

        # --- solve optimization ---
        t0, _ = elapsed()
        _ = self.solve(**kwargs)
        status = self.om.prob.status
        self.exit_code = self.syms.status[status]
        self.converged = self.exit_code == 0
        _, s = elapsed(t0)
        self.exec_time = float(s.split(" ")[0])
        sstats = self.om.prob.solver_stats  # solver stats
        if sstats.num_iters is None:
            n_iter = -1
        else:
            n_iter = int(sstats.num_iters)
        n_iter_str = f"{n_iter} iterations " if n_iter > 1 else f"{n_iter} iteration "
        if self.exit_code == 0:
            msg = f"<{self.class_name}> solved as {status} in {s}, converged in "
            msg += n_iter_str + f"with {sstats.solver_name}."
            logger.warning(msg)
            self.unpack(res=None, **kwargs)
            self._post_solve()
            self.system.report()
            return True
        else:
            msg = f"{self.class_name} failed as {status} in "
            msg += n_iter_str + f"with {sstats.solver_name}!"
            logger.warning(msg)
            return False

    def export_csv(self, path=None):
        """
        Export scheduling results to a csv file.
        For multi-period routines, the column "Time" is the time index of
        ``timeslot.v``, which usually comes from ``EDTSlot`` or ``UCTSlot``.
        The rest columns are the variables registered in ``vars``.

        For single-period routines, the column "Time" have a pseduo value of "T1".

        Parameters
        ----------
        path : str
            path of the csv file to save

        Returns
        -------
        export_path
            The path of the exported csv file
        """
        if not self.converged:
            logger.warning("Routine did not converge, aborting export.")
            return None

        if not path:
            if self.system.files.fullname is None:
                logger.info("Input file name not detacted. Using `Untitled`.")
                file_name = f'Untitled_{self.class_name}'
            else:
                file_name = os.path.splitext(self.system.files.fullname)[0]
                file_name += f'_{self.class_name}'
            path = os.path.join(os.getcwd(), file_name + '.csv')

        data_dict = initialize_data_dict(self)

        collect_data(self, data_dict, self.vars, 'v')
        collect_data(self, data_dict, self.exprs, 'v')
        collect_data(self, data_dict, self.exprcs, 'v')

        if 'T1' in data_dict['Time']:
            data_dict = OrderedDict([(k, [v]) for k, v in data_dict.items()])

        pd.DataFrame(data_dict).to_csv(path, index=False)

        return file_name + '.csv'

    def summary(self, **kwargs):
        """
        Summary interface
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.class_name} at {hex(id(self))}"

    def dc2ac(self, kloss=1.0, **kwargs):
        """
        Convert the DC-based results with ACOPF.
        """
        raise NotImplementedError

    def _check_attribute(self, key, value):
        """
        Check the attribute pair for valid names while instantiating the class.

        This function assigns `owner` to the model itself, assigns the name and tex_name.
        """
        if key in self.__dict__:
            existing_keys = []
            for rtn_type in ["constrs", "vars", "rparams", "services"]:
                if rtn_type in self.__dict__:
                    existing_keys += list(self.__dict__[rtn_type].keys())
            if key in existing_keys:
                msg = f"Attribute <{key}> already exists in <{self.class_name}>."
                logger.warning(msg)

        # register owner routine instance of following attributes
        if isinstance(value, (RBaseService)):
            value.rtn = self

    def __setattr__(self, key, value):
        """
        Overload the setattr function to register attributes.

        Parameters
        ----------
        key: str
            name of the attribute
        value:
            value of the attribute
        """

        # NOTE: value.id is not in use yet
        if isinstance(value, Var):
            value.id = len(self.vars)
        self._check_attribute(key, value)
        self._register_attribute(key, value)

        super(RoutineBase, self).__setattr__(key, value)

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to the routine instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        """
        if isinstance(value, (Param, Var, Constraint, Objective, ExpressionCalc, Expression)):
            value.om = self.om
            value.rtn = self
        if isinstance(value, Param):
            self.params[key] = value
            self.om.params[key] = None  # cp.Parameter
        if isinstance(value, Var):
            self.vars[key] = value
            self.om.vars[key] = None  # cp.Variable
        elif isinstance(value, Constraint):
            self.constrs[key] = value
            self.om.constrs[key] = None  # cp.Constraint
        elif isinstance(value, Expression):
            self.exprs[key] = value
            self.om.exprs[key] = None  # cp.Expression
        elif isinstance(value, ExpressionCalc):
            self.exprcs[key] = value
        elif isinstance(value, RParam):
            self.rparams[key] = value
        elif isinstance(value, RBaseService):
            self.services[key] = value

    def update(self, params=None, build_mats=False):
        """
        Update the values of Parameters in the optimization model.

        This method is particularly important when some `RParams` are
        linked with system matrices.
        In such cases, setting `build_mats=True` is necessary to rebuild
        these matrices for the changes to take effect.
        This is common in scenarios involving topology changes, connection statuses,
        or load value modifications.
        If unsure, it is advisable to use `build_mats=True` as a precautionary measure.

        Parameters
        ----------
        params: Parameter, str, or list
            Parameter, Parameter name, or a list of parameter names to be updated.
            If None, all parameters will be updated.
        build_mats: bool
            True to rebuild the system matrices. Set to False to speed up the process
            if no system matrices are changed.
        """
        if not self.initialized:
            return self.init()
        t0, _ = elapsed()
        re_finalize = False
        # sanitize input
        sparams = []
        if params is None:
            sparams = [val for val in self.params.values()]
            build_mats = True
        elif isinstance(params, Param):
            sparams = [params]
        elif isinstance(params, str):
            sparams = [self.params[params]]
        elif isinstance(params, list):
            sparams = [self.params[param] for param in params if isinstance(param, str)]
            for param in sparams:
                param.update()

        for param in sparams:
            if param.optz is None:  # means no_parse=True
                re_finalize = True
                break

        self.system.mats.build(force=build_mats)

        if re_finalize:
            logger.warning(f"<{self.class_name}> reinit OModel due to non-parametric change.")
            self.om.evaluate(force=True)
            self.om.finalize(force=True)

        results = self.om.update(params=sparams)
        t0, s0 = elapsed(t0)
        logger.debug(f"Update params in {s0}.")
        return results

    def __delattr__(self, name):
        """
        Overload the delattr function to unregister attributes.

        Parameters
        ----------
        name: str
            name of the attribute
        """
        self._unregister_attribute(name)
        if name == "obj":
            self.obj = None
        else:
            super().__delattr__(name)  # Call the superclass implementation

    def _unregister_attribute(self, name):
        """
        Unregister a pair of attributes from the routine instance.

        Called within ``__delattr__``, this is where the magic happens.
        Subclass attributes are automatically unregistered based on the variable type.
        """
        if name in self.vars:
            del self.vars[name]
            if name in self.om.vars:
                del self.om.vars[name]
        elif name in self.rparams:
            del self.rparams[name]
        elif name in self.constrs:
            del self.constrs[name]
            if name in self.om.constrs:
                del self.om.constrs[name]
        elif name in self.services:
            del self.services[name]

    def enable(self, name):
        """
        Enable a constraint by name.

        Parameters
        ----------
        name: str or list
            name of the constraint to be enabled
        """
        if isinstance(name, list):
            constr_act = []
            for n in name:
                if n not in self.constrs:
                    logger.warning(f"Constraint <{n}> not found.")
                    continue
                if not self.constrs[n].is_disabled:
                    logger.warning(f"Constraint <{n}> has already been enabled.")
                    continue
                self.constrs[n].is_disabled = False
                self.om.finalized = False
                constr_act.append(n)
            if len(constr_act) > 0:
                msg = ", ".join(constr_act)
                logger.warning(f"Turn on constraints: {msg}")
            return True

        if name in self.constrs:
            if not self.constrs[name].is_disabled:
                logger.warning(f"Constraint <{name}> has already been enabled.")
            else:
                self.constrs[name].is_disabled = False
                self.om.finalized = False
                logger.warning(f"Turn on constraint <{name}>.")
            return True

    def disable(self, name):
        """
        Disable a constraint by name.

        Parameters
        ----------
        name: str or list
            name of the constraint to be disabled
        """
        if isinstance(name, list):
            constr_act = []
            for n in name:
                if n not in self.constrs:
                    logger.warning(f"Constraint <{n}> not found.")
                elif self.constrs[n].is_disabled:
                    logger.warning(f"Constraint <{n}> has already been disabled.")
                else:
                    self.constrs[n].is_disabled = True
                    self.om.finalized = False
                    constr_act.append(n)
            if len(constr_act) > 0:
                msg = ", ".join(constr_act)
                logger.warning(f"Turn off constraints: {msg}")
            return True

        if name in self.constrs:
            if self.constrs[name].is_disabled:
                logger.warning(f"Constraint <{name}> has already been disabled.")
            else:
                self.constrs[name].is_disabled = True
                self.om.finalized = False
                logger.warning(f"Turn off constraint <{name}>.")
            return True

        logger.warning(f"Constraint <{name}> not found.")

    def _post_add_check(self):
        """
        Post-addition check.
        """
        # --- reset routine status ---
        self.initialized = False
        self.exec_time = 0.0
        self.exit_code = 0
        # --- reset symprocessor status ---
        self._syms = False
        # --- reset optimization model status ---
        self.om.parsed = False
        self.om.evaluated = False
        self.om.finalized = False
        # --- reset OModel parser status ---
        self.om.parsed = False

    def addRParam(self,
                  name: str,
                  tex_name: Optional[str] = None,
                  info: Optional[str] = None,
                  src: Optional[str] = None,
                  unit: Optional[str] = None,
                  model: Optional[str] = None,
                  v: Optional[np.ndarray] = None,
                  indexer: Optional[str] = None,
                  imodel: Optional[str] = None,):
        """
        Add `RParam` to the routine.

        Parameters
        ----------
        name : str
            Name of this parameter. If not provided, `name` will be set
            to the attribute name.
        tex_name : str, optional
            LaTeX-formatted parameter name. If not provided, `tex_name`
            will be assigned the same as `name`.
        info : str, optional
            A description of this parameter
        src : str, optional
            Source name of the parameter.
        unit : str, optional
            Unit of the parameter.
        model : str, optional
            Name of the owner model or group.
        v : np.ndarray, optional
            External value of the parameter.
        indexer : str, optional
            Indexer of the parameter.
        imodel : str, optional
            Name of the owner model or group of the indexer.
        """
        item = RParam(name=name, tex_name=tex_name, info=info, src=src, unit=unit,
                      model=model, v=v, indexer=indexer, imodel=imodel)

        # add the parameter as an routine attribute
        setattr(self, name, item)

        # NOTE: manually register the owner of the parameter
        # This is skipped in ``addVars`` because of ``Var.__setattr__``
        item.rtn = self

        # check variable owner validity if given
        if model is not None:
            if item.model in self.system.groups.keys():
                item.is_group = True
                item.owner = self.system.groups[item.model]
            elif item.model in self.system.models.keys():
                item.owner = self.system.models[item.model]
            else:
                msg = f'Model indicator \'{item.model}\' of <{item.rtn.class_name}.{name}>'
                msg += ' is not a model or group. Likely a modeling error.'
                logger.warning(msg)

        self._post_add_check()
        return item

    def addService(self,
                   name: str,
                   value: np.ndarray,
                   tex_name: str = None,
                   unit: str = None,
                   info: str = None,
                   vtype: Type = None,):
        """
        Add `ValueService` to the routine.

        Parameters
        ----------
        name : str
            Instance name.
        value : np.ndarray
            Value.
        tex_name : str, optional
            TeX name.
        unit : str, optional
            Unit.
        info : str, optional
            Description.
        vtype : Type, optional
            Variable type.
        """
        item = ValueService(name=name, tex_name=tex_name,
                            unit=unit, info=info,
                            vtype=vtype, value=value)
        # add the service as an routine attribute
        setattr(self, name, item)

        self._post_add_check()

        return item

    def addConstrs(self,
                   name: str,
                   e_str: str,
                   info: Optional[str] = None,
                   is_eq: Optional[str] = False,):
        """
        Add `Constraint` to the routine. to the routine.

        Parameters
        ----------
        name : str
            Constraint name. One should typically assigning the name directly because
            it will be automatically assigned by the model. The value of ``name``
            will be the symbol name to be used in expressions.
        e_str : str
            Constraint expression string.
        info : str, optional
            Descriptive information
        is_eq : str, optional
            Flag indicating if the constraint is an equality constraint. False indicates
            an inequality constraint in the form of `<= 0`.
        """
        item = Constraint(name=name, e_str=e_str, info=info, is_eq=is_eq)
        # add the constraint as an routine attribute
        setattr(self, name, item)

        self._post_add_check()

        return item

    def addVars(self,
                name: str,
                model: Optional[str] = None,
                shape: Optional[Union[int, tuple]] = None,
                tex_name: Optional[str] = None,
                info: Optional[str] = None,
                src: Optional[str] = None,
                unit: Optional[str] = None,
                horizon: Optional[RParam] = None,
                nonneg: Optional[bool] = False,
                nonpos: Optional[bool] = False,
                cplx: Optional[bool] = False,
                imag: Optional[bool] = False,
                symmetric: Optional[bool] = False,
                diag: Optional[bool] = False,
                psd: Optional[bool] = False,
                nsd: Optional[bool] = False,
                hermitian: Optional[bool] = False,
                boolean: Optional[bool] = False,
                integer: Optional[bool] = False,
                pos: Optional[bool] = False,
                neg: Optional[bool] = False,):
        """
        Add a variable to the routine.

        Parameters
        ----------
        name : str, optional
            Variable name. One should typically assigning the name directly because
            it will be automatically assigned by the model. The value of ``name``
            will be the symbol name to be used in expressions.
        model : str, optional
            Name of the owner model or group.
        shape : int or tuple, optional
            Shape of the variable. If is None, the shape of `model` will be used.
        info : str, optional
            Descriptive information
        unit : str, optional
            Unit
        tex_name : str
            LaTeX-formatted variable symbol. If is None, the value of `name` will be
            used.
        src : str, optional
            Source variable name. If is None, the value of `name` will be used.
        lb : str, optional
            Lower bound
        ub : str, optional
            Upper bound
        horizon : ams.routines.RParam, optional
            Horizon idx.
        nonneg : bool, optional
            Non-negative variable
        nonpos : bool, optional
            Non-positive variable
        cplx : bool, optional
            Complex variable
        imag : bool, optional
            Imaginary variable
        symmetric : bool, optional
            Symmetric variable
        diag : bool, optional
            Diagonal variable
        psd : bool, optional
            Positive semi-definite variable
        nsd : bool, optional
            Negative semi-definite variable
        hermitian : bool, optional
            Hermitian variable
        bool : bool, optional
            Boolean variable
        integer : bool, optional
            Integer variable
        pos : bool, optional
            Positive variable
        neg : bool, optional
            Negative variable
        """
        if model is None and shape is None:
            raise ValueError("Either model or shape must be specified.")
        item = Var(name=name, tex_name=tex_name,
                   info=info, src=src, unit=unit,
                   model=model, shape=shape, horizon=horizon,
                   nonneg=nonneg, nonpos=nonpos,
                   cplx=cplx, imag=imag,
                   symmetric=symmetric, diag=diag,
                   psd=psd, nsd=nsd, hermitian=hermitian,
                   boolean=boolean, integer=integer,
                   pos=pos, neg=neg, )

        # add the variable as an routine attribute
        setattr(self, name, item)

        # check variable owner validity if given
        if model is not None:
            if item.model in self.system.groups.keys():
                item.is_group = True
                item.owner = self.system.groups[item.model]
            elif item.model in self.system.models.keys():
                item.owner = self.system.models[item.model]
            else:
                msg = (
                    f"Model indicator '{item.model}' of <{item.rtn.class_name}.{name}>"
                )
                msg += " is not a model or group. Likely a modeling error."
                logger.warning(msg)

        self._post_add_check()

        return item

    def _initial_guess(self):
        """
        Generate initial guess for the optimization model.
        """
        raise NotImplementedError


def initialize_data_dict(rtn: RoutineBase):
    """
    Initialize the data dictionary for export.

    Parameters
    ----------
    rtn : ams.routines.routine.RoutineBase
        The routine to collect data from

    Returns
    -------
    OrderedDict
        The initialized data dictionary.
    """
    if hasattr(rtn, 'timeslot'):
        timeslot = rtn.timeslot.v.copy()
        return OrderedDict([('Time', timeslot)])
    else:
        return OrderedDict([('Time', 'T1')])


def collect_data(rtn: RoutineBase, data_dict: Dict, items: Dict, attr: str):
    """
    Collect data for export.

    Parameters
    ----------
    rtn : ams.routines.routine.RoutineBase
        The routine to collect data from.
    data_dict : OrderedDict
        The data dictionary to populate.
    items : dict
        Dictionary of items to collect data from.
    attr : str
        Attribute to collect data for.
    """
    for key, item in items.items():
        if item.owner is None:
            continue
        idx_v = item.get_all_idxes()
        try:
            data_v = rtn.get(src=key, attr=attr, idx=idx_v,
                             horizon=rtn.timeslot.v if hasattr(rtn, 'timeslot') else None).round(6)
        except Exception as e:
            logger.debug(f"Error with collecting data for '{key}': {e}")
            data_v = [np.nan] * len(idx_v)
        data_dict.update(OrderedDict(zip([f'{key} {dev}' for dev in idx_v], data_v)))
