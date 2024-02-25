"""
Module for routine data.
"""

import logging
import os
from typing import Optional, Union, Type, Iterable
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from andes.core import Config
from andes.shared import pd
from andes.utils.misc import elapsed

from ams.core.param import RParam
from ams.core.symprocessor import SymProcessor
from ams.core.documenter import RDocumenter
from ams.core.service import RBaseService, ValueService
from ams.opt.omodel import OModel, Param, Var, Constraint, Objective

from ams.shared import igraph as ig
from ams.shared import require_igraph


logger = logging.getLogger(__name__)


class RoutineBase:
    """
    Class to hold descriptive routine models and data mapping.
    """

    def __init__(self, system=None, config=None):
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
        self.is_ac = False          # AC conversion flag

    @property
    def class_name(self):
        return self.__class__.__name__

    def _loc(self, src: str, idx, allow_none=False):
        """
        Helper function to index a variable or parameter in a routine.
        """
        src_idx = self.__dict__[src].get_idx()
        loc = [src_idx.index(idxe) if idxe in src_idx else None for idxe in idx]
        if None not in loc:
            return loc
        else:
            idx_none = [idxe for idxe in idx if idxe not in src_idx]
            msg = f"Var <{self.class_name}.{src}> does not contain value with idx={idx_none}"
            raise ValueError(msg)

    def get_load(self, horizon: Union[int, str],
                 src: str, attr: str = 'v',
                 idx=None, model: str = 'EDTSlot', factor: str = 'sd',):
        """
        Get the load value by applying zonal scaling factor defined in ``Horizon``.

        Parameters
        ----------
        idx: int, str, or list
            Index of the desired load.
        attr: str
            Attribute name.
        model: str
            Scaling factor owner, ``EDTSlot`` or ``UCTSlot``.
        factor: str
            Scaling factor name, usually ``sd``.
        horizon: int or str
            Horizon single index.
        """
        all_zone = self.system.Region.idx.v
        if idx is None:
            pq_zone = self.system.PQ.zone.v
            pq0 = self.system.PQ.get(src=src, attr=attr, idx=idx)
        else:
            pq_zone = self.system.PQ.get(src="zone", attr="v", idx=idx)
            pq0 = self.system.PQ.get(src=src, attr=attr, idx=idx)
        col = [all_zone.index(pq_z) for pq_z in pq_zone]

        mdl = self.system.__dict__[model]
        if mdl.n == 0:
            raise ValueError(f"<{model}> does not have data, check input file.")
        if factor not in mdl.__dict__.keys():
            raise ValueError(f"<{model}> does not have <{factor}>.")
        sdv = mdl.__dict__[factor].v

        horizon_all = mdl.idx.v
        try:
            row = horizon_all.index(horizon)
        except ValueError as e:
            msg = f"<{model}> does not have horizon with idx=<{horizon}>. "
            msg += f"Original error: {e}"
            raise ValueError(msg)
        pq_factor = np.array(sdv[:, col][row, :])
        pqv = np.multiply(pq0, pq_factor)
        return pqv

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

        idx_all = item.get_idx()

        if idx_all is None:
            raise ValueError(f"<{self.class_name}> item <{src}> has no idx.")

        if isinstance(idx, (str, int)):
            idx = [idx]

        if isinstance(idx, np.ndarray):
            idx = idx.tolist()

        loc = [idx_all.index(idxe) if idxe in idx_all else None for idxe in idx]
        if None in loc:
            idx_none = [idxe for idxe in idx if idxe not in idx_all]
            msg = f"Var <{self.class_name}.{src}> does not contain value with idx={idx_none}"
            raise ValueError(msg)
        out = getattr(item, attr)[loc]

        if horizon is not None:
            if item.horizon is None:
                raise ValueError(f"horizon is not defined for {self.class_name}.{src}.")
            horizon_all = item.horizon.get_idx()
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
        return out

    def set(self, src: str, idx, attr: str = "v", value=0.0):
        """
        Set the value of an attribute of a routine parameter.
        """
        if self.__dict__[src].owner is not None:
            # TODO: fit to `_v` type param in the future
            owner = self.__dict__[src].owner
            src0 = self.__dict__[src].src
            src_owner = src0 if src0 is not None else src
            try:
                res = owner.set(src=src_owner, idx=idx, attr=attr, value=value)
                return res
            except KeyError as e:
                msg = f"Failed to set <{src0}> in <{owner.class_name}>. "
                msg += f"Original error: {e}"
                raise KeyError(msg)
            else:
                logger.info(f"Failed to set <{src0}> in <{owner.class_name}>.")
                return None
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
        return disabled

    def _data_check(self, info=True):
        """
        Check if data is valid for a routine.

        Parameters
        ----------
        info: bool
            Whether to print warning messages.
        """
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
                logger.warning(msg)
            return False
        # TODO: add data validation for RParam, typical range, etc.
        return True

    def init(self, force=False, no_code=True, **kwargs):
        """
        Initialize the routine.

        Force initialization (`force=True`) will do the following:
        - Rebuild the system matrices
        - Enable all constraints
        - Reinitialize the optimization model

        Parameters
        ----------
        force: bool
            Whether to force initialization.
        no_code: bool
            Whether to show generated code.
        """
        skip_all = (not force) and self.initialized and self.om.initialized
        skip_ominit = (not force) and self.om.initialized

        if skip_all:
            logger.debug(f"{self.class_name} has already been initialized.")
            return True

        t0, _ = elapsed()
        # --- data check ---
        if self._data_check():
            logger.debug(f"{self.class_name} data check passed.")
        else:
            msg = f"{self.class_name} data check failed, setup may run into error!"
            logger.warning(msg)

        # --- matrix build ---
        if force or isinstance(self.system.mats.Cft._v, type(None)):
            self.system.mats.make()
            for constr in self.constrs.values():
                constr.is_disabled = False

        # --- constraint check ---
        disabled = self._get_off_constrs()
        if len(disabled) > 0:
            msg = "Disabled constraints: "
            d_str = [f'{constr}' for constr in disabled]
            msg += ", ".join(d_str)
            logger.warning(msg)

        if not skip_ominit:
            om_init = self.om.init(no_code=no_code)
        else:
            om_init = True
        _, s_init = elapsed(t0)

        msg = f"Routine <{self.class_name}> "
        if om_init:
            msg += f"initialized in {s_init}."
            self.initialized = True
        else:
            msg += "initialization failed!"
            self.initialized = False
        logger.info(msg)
        return self.initialized

    def prepare(self):
        """
        Prepare the routine.
        """
        logger.debug("Generating code for %s", self.class_name)
        self.syms.generate_symbols()

    def solve(self, **kwargs):
        """
        Solve the routine optimization model.
        """
        return True

    def unpack(self, **kwargs):
        """
        Unpack the results.
        """
        return None

    def _post_solve(self):
        """
        Post-solve calculation.
        """
        return None

    def run(self, force_init=False, no_code=True, **kwargs):
        """
        Run the routine.

        Force initialization (`force_init=True`) will do the following:
        - Rebuild the system matrices
        - Enable all constraints
        - Reinitialize the optimization model

        Parameters
        ----------
        force_init: bool
            Whether to force initialization.
        no_code: bool
            Whether to show generated code.
        """
        # --- setup check ---
        self.init(force=force_init, no_code=no_code)
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
            msg = f"{self.class_name} solved as {status} in {s}, converged after "
            msg += n_iter_str + f"using solver {sstats.solver_name}."
            logger.warning(msg)
            self.unpack(**kwargs)
            self._post_solve()
            return True
        else:
            msg = f"{self.class_name} failed as {status} after "
            msg += n_iter_str + f"using solver {sstats.solver_name}!"
            logger.warning(msg)
            return False

    def export_csv(self, path=None):
        """
        Export dispatch results to a csv file.
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
        str
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

        idxes = [var.get_idx() for var in self.vars.values()]
        var_names = [var for var in self.vars.keys()]

        if hasattr(self, 'timeslot'):
            timeslot = self.timeslot.v.copy()
            data_dict = OrderedDict([('Time', timeslot)])
        else:
            timeslot = None
            data_dict = OrderedDict([('Time', 'T1')])

        for var, idx in zip(var_names, idxes):
            header = [f'{var} {dev}' for dev in idx]
            data = self.get(src=var, idx=idx, horizon=timeslot).round(6)
            data_dict.update(OrderedDict(zip(header, data)))

        if timeslot is None:
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

    def _ppc2ams(self):
        """
        Convert PYPOWER results to AMS.
        """
        raise NotImplementedError

    def dc2ac(self, **kwargs):
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
            for type in ["constrs", "vars", "rparams", "services"]:
                if type in self.__dict__:
                    existing_keys += list(self.__dict__[type].keys())
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
        if isinstance(value, (Param, Var, Constraint, Objective)):
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
        elif isinstance(value, RParam):
            self.rparams[key] = value
        elif isinstance(value, RBaseService):
            self.services[key] = value

    def update(self, params=None, mat_make=True,):
        """
        Update the values of Parameters in the optimization model.

        This method is particularly important when some `RParams` are
        linked with system matrices.
        In such cases, setting `mat_make=True` is necessary to rebuild
        these matrices for the changes to take effect.
        This is common in scenarios involving topology changes, connection statuses,
        or load value modifications.
        If unsure, it is advisable to use `mat_make=True` as a precautionary measure.

        Parameters
        ----------
        params: Parameter, str, or list
            Parameter, Parameter name, or a list of parameter names to be updated.
            If None, all parameters will be updated.
        mat_make: bool
            True to rebuild the system matrices. Set to False to speed up the process
            if no system matrices are changed.
        """
        t0, _ = elapsed()
        re_setup = False
        # sanitize input
        sparams = []
        if params is None:
            sparams = [val for val in self.params.values()]
            mat_make = True
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
                re_setup = True
                break
        if mat_make:
            self.system.mats.make()
        if re_setup:
            logger.warning(f"Re-init {self.class_name} OModel due to non-parametric change.")
            _ = self.om.init(no_code=True)
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
                self.om.initialized = False
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
                self.om.initialized = False
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
                    self.om.initialized = False
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
                self.om.initialized = False
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
        self.om.initialized = False

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
                   vtype: Type = None,
                   model: str = None,):
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
        model : str, optional
            Model name.
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
                   type: Optional[str] = 'uq',
                   ):
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
        type : str, optional
            Constraint type, ``uq`` for uncertain, ``eq`` for equality, ``ineq`` for inequality.

        """
        item = Constraint(name=name, e_str=e_str, info=info, type=type)
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
                complex: Optional[bool] = False,
                imag: Optional[bool] = False,
                symmetric: Optional[bool] = False,
                diag: Optional[bool] = False,
                psd: Optional[bool] = False,
                nsd: Optional[bool] = False,
                hermitian: Optional[bool] = False,
                bool: Optional[bool] = False,
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
        complex : bool, optional
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
                   complex=complex, imag=imag,
                   symmetric=symmetric, diag=diag,
                   psd=psd, nsd=nsd, hermitian=hermitian,
                   boolean=bool, integer=integer,
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

    @require_igraph
    def igmake(self, directed=True):
        """
        Build an igraph object from the system.

        Parameters
        ----------
        directed: bool
            Whether the graph is directed.

        Returns
        -------
        igraph.Graph
            An igraph object.
        """
        system = self.system
        edges = np.column_stack([system.Bus.idx2uid(system.Line.bus1.v),
                                 system.Bus.idx2uid(system.Line.bus2.v)])
        g = ig.Graph(n=system.Bus.n, directed=directed, edges=edges)
        return g

    @require_igraph
    def igraph(
        self,
        input: Optional[Union[RParam, Var]] = None,
        ytimes: Optional[float] = None,
        decimal: Optional[int] = 6,
        directed: Optional[bool] = True,
        dpi: Optional[int] = 100,
        figsize: Optional[tuple] = None,
        adjust_bus: Optional[bool] = False,
        gen_color: Optional[str] = "red",
        rest_color: Optional[str] = "black",
        vertex_shape: Optional[str] = "circle",
        vertex_font: Optional[str] = None,
        no_vertex_label: Optional[bool] = False,
        vertex_label: Optional[Union[str, list]] = None,
        vertex_size: Optional[float] = None,
        vertex_label_size: Optional[float] = None,
        vertex_label_dist: Optional[float] = 1.5,
        vertex_label_angle: Optional[float] = 10.2,
        edge_arrow_size: Optional[float] = None,
        edge_arrow_width: Optional[float] = None,
        edge_width: Optional[float] = None,
        edge_align_label: Optional[bool] = True,
        edge_background: Optional[str] = None,
        edge_color: Optional[str] = None,
        edge_curved: Optional[bool] = False,
        edge_font: Optional[str] = None,
        edge_label: Optional[Union[str, list]] = None,
        layout: Optional[str] = "rt",
        autocurve: Optional[bool] = True,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        title_loc: Optional[str] = None,
        **visual_style,
    ):
        """
        Plot a system uging `g.plot()` of `igraph`, with optional input.
        For now, only support plotting of Bus and Line elements as input.

        Examples
        --------
        >>> import ams
        >>> sp = ams.load(ams.get_case('5bus/pjm5bus_uced.xlsx'))
        >>> sp.DCOPF.run()
        >>> sp.DCOPF.plot(input=sp.DCOPF.pn,
        >>>               ytimes=10,
        >>>               adjust_bus=True,
        >>>               vertex_size=10,
        >>>               vertex_label_size=15,
        >>>               vertex_label_dist=2,
        >>>               vertex_label_angle=90,
        >>>               show=False,
        >>>               edge_align_label=True,
        >>>               autocurve=True,)

        Parameters
        ----------
        input: RParam or Var, optional
            The variable or parameter to be plotted.
        ytimes: float, optional
            The scaling factor of the values.
        directed: bool, optional
            Whether the graph is directed.
        dpi: int, optional
            Dots per inch.
        figsize: tuple, optional
            Figure size.
        adjust_bus: bool, optional
            Whether to adjust the bus size.
        gen_color: str, optional
            Color of the generator bus.
        rest_color: str, optional
            Color of the rest buses.
        no_vertex_label: bool, optional
            Whether to show vertex labels.
        vertex_shape: str, optional
            Shape of the vertices.
        vertex_font: str, optional
            Font of the vertices.
        vertex_size: float, optional
            Size of the vertices.
        vertex_label_size: float, optional
            Size of the vertex labels.
        vertex_label_dist: float, optional
            Distance of the vertex labels.
        vertex_label_angle: float, optional
            Angle of the vertex labels.
        edge_arrow_size: float, optional
            Size of the edge arrows.
        edge_arrow_width: float, optional
            Width of the edge arrows.
        edge_width: float, optional
            Width of the edges.
        edge_align_label: bool, optional
            Whether to align the edge labels.
        edge_background: str, optional
            RGB colored rectangle background of the edge labels.
        layout: str, optional
            Layout of the graph, ['rt', 'kk', 'fr', 'drl', 'lgl', 'circle', 'grid_fr'].
        autocurve: bool, optional
            Whether to use autocurve.
        ax: plt.Axes, optional
            Matplotlib axes.
        visual_style: dict, optional
            Visual style, see ``igraph.plot`` for details.

        Returns
        -------
        plt.Axes
            Matplotlib axes.
        igraph.Graph
            An igraph object.
        """

        g = self.igmake(directed=directed)

        # --- visual style ---
        vstyle = {
            # layout style
            "layout": layout,
            # vertices
            "vertex_shape": vertex_shape,
            "vertex_font": vertex_font,
            "vertex_size": vertex_size,
            "vertex_label": vertex_label,
            "vertex_label_size": vertex_label_size,
            "vertex_label_dist": vertex_label_dist,
            "vertex_label_angle": vertex_label_angle,
            # edges
            "edge_arrow_size": edge_arrow_size,
            "edge_arrow_width": edge_arrow_width,
            "edge_width": edge_width,
            "edge_align_label": edge_align_label,
            "edge_background": edge_background,
            "edge_color": edge_color,
            "edge_curved": edge_curved,
            "edge_font": edge_font,
            "edge_label": edge_label,
            # others
            **visual_style,
        }
        system = self.system
        # bus name, will be overwritten if input is not None
        vstyle["vertex_name"] = system.Bus.name.v
        if vertex_label is None:
            vstyle["vertex_label"] = None if no_vertex_label else system.Bus.name.v

        # bus size
        gidx = system.PV.idx.v + system.Slack.idx.v
        gbus = system.StaticGen.get(src="bus", attr="v", idx=gidx)
        # initialize all bus size as vertex_size
        bus_size = [vertex_size] * system.Bus.n
        if adjust_bus and isinstance(vertex_size, (int, float)):
            # adjust gen bus size using Sn
            gsn = system.StaticGen.get(src="Sn", attr="v", idx=gidx)
            gbsize = vertex_size * gsn / gsn.max()
            gbus_dict = {bus: size for bus, size in zip(gbus, gbsize)}
            for key, val in gbus_dict.items():
                bus_size[system.Bus.idx2uid(key)] = val
        if isinstance(vertex_size, Iterable):
            bus_size = vertex_size
        vstyle["vertex_size"] = bus_size

        # bus colors
        gbus_uid = system.Bus.idx2uid(gbus)
        bus_uid = system.Bus.idx2uid(system.Bus.idx.v)
        g.vs["label"] = system.Bus.name.v
        g.vs["bus_type"] = ["gen" if bus_i in gbus_uid else "rest" for bus_i in bus_uid]
        color_dict = {"gen": gen_color, "rest": rest_color}
        vstyle["vertex_color"] = [color_dict[btype] for btype in g.vs["bus_type"]]

        # --- variables ---
        k = ytimes if ytimes is not None else 1
        if input is not None:
            if input.owner.class_name == "Bus":
                logger.debug(f"Plotting <{input.name}> as vertex label.")
                values = [f"${input.tex_name}$={round(k*v, decimal)}" for v in input.v]
                vstyle["vertex_label"] = values
            elif input.owner.class_name == "Line":
                logger.debug(f"Plotting <{input.name}> as edge label.")
                values = [f"${input.tex_name}$={round(k*v, decimal)}" for v in input.v]
                elabel = system.Line.name.v
                eout = [f"{label}" for label, ein in zip(values, elabel)]
                vstyle["edge_label"] = eout
            else:
                logger.error(f"Unsupported input type <{input.owner.class_name}>.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        default_name = self.class_name
        if input is not None:
            default_name += f"\n${input.tex_name}$" + f" [${input.unit}$]"
        ax.set_title(title if title else default_name, loc=title_loc)
        ig.plot(g, autocurve=autocurve, target=ax, **vstyle)
        return ax, g
