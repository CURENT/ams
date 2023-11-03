"""
Module for routine data.
"""

import logging  # NOQA
from typing import Optional, Union, Type, Iterable  # NOQA
from collections import OrderedDict  # NOQA

import numpy as np  # NOQA

from andes.core import Config  # NOQA
from andes.shared import deg2rad  # NOQA
from andes.utils.misc import elapsed  # NOQA
from ams.utils import timer  # NOQA
from ams.core.param import RParam  # NOQA
from ams.opt.omodel import OModel, Var, Constraint, Objective  # NOQA

from ams.core.symprocessor import SymProcessor  # NOQA
from ams.core.documenter import RDocumenter  # NOQA
from ams.core.service import RBaseService, ValueService  # NOQA

from ams.models.group import GroupBase  # NOQA
from ams.core.model import Model  # NOQA

logger = logging.getLogger(__name__)


class RoutineData:
    """
    Class to hold routine parameters.
    """

    def __init__(self):
        self.rparams = OrderedDict()  # list out RParam in a routine


class RoutineModel:
    """
    Class to hold descriptive routine models and data mapping.
    """

    def __init__(self, system=None, config=None):
        self.system = system
        self.config = Config(self.class_name)
        self.info = None
        self.tex_names = OrderedDict((('sys_f', 'f_{sys}'),
                                      ('sys_mva', 'S_{b,sys}'),
                                      ))
        self.syms = SymProcessor(self)  # symbolic processor
        self._syms = False  # flag if symbols has been generated

        self.services = OrderedDict()  # list out services in a routine

        self.vars = OrderedDict()  # list out Vars in a routine
        self.constrs = OrderedDict()
        self.obj = None
        self.initialized = False
        self.type = 'UndefinedType'
        self.docum = RDocumenter(self)

        # --- sync mapping ---
        self.map1 = OrderedDict()  # from ANDES
        self.map2 = OrderedDict()  # to ANDES

        # --- optimization modeling ---
        self.om = OModel(routine=self)

        if config is not None:
            self.config.load(config)
        # TODO: these default configs might to be revised
        self.config.add(OrderedDict((('sparselib', 'klu'),
                                     ('linsolve', 0),
                                     )))
        self.config.add_extra("_help",
                              sparselib="linear sparse solver name",
                              linsolve="solve symbolic factorization each step (enable when KLU segfaults)",
                              )
        self.config.add_extra("_alt",
                              sparselib=("klu", "umfpack", "spsolve", "cupy"),
                              linsolve=(0, 1),
                              )

        self.exec_time = 0.0  # recorded time to execute the routine in seconds
        # TODO: check exit_code of gurobipy or any other similiar solvers
        self.exit_code = 0  # exit code of the routine;

        self.is_ac = False  # whether the routine is converted to AC

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
            raise ValueError(f'Var <{self.class_name}.{src}> does not contain value with idx={idx_none[0]}')

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
            pq_zone = self.system.PQ.get(src='zone', attr='v', idx=idx)
            pq0 = self.system.PQ.get(src=src, attr=attr, idx=idx)
        col = [all_zone.index(pq_z) for pq_z in pq_zone]

        mdl = self.system.__dict__[model]
        if mdl.n == 0:
            raise ValueError(f'<{model}> does not have data, check input file.')
        if factor not in mdl.__dict__.keys():
            raise ValueError(f'<{model}> does not have <{factor}>.')
        sdv = mdl.__dict__[factor].v

        horizon_all = mdl.idx.v
        try:
            row = horizon_all.index(horizon)
        except ValueError:
            raise ValueError(f'<{model}> does not have horizon with idx=<{horizon}>.')
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
        horizon: int, str, or list, optional
            Horizon index.
        """
        if src not in self.__dict__.keys():
            raise ValueError(f'<{src}> does not exist in <<{self.class_name}>.')
        item = self.__dict__[src]

        if not hasattr(item, attr):
            raise ValueError(f'{attr} does not exist in {self.class_name}.{src}.')

        idx_all = item.get_idx()

        if idx_all is None:
            raise ValueError(f'<{self.class_name}> item <{src}> has no idx.')

        if isinstance(idx, (str, int)):
            idx = [idx]

        if isinstance(idx, np.ndarray):
            idx = idx.tolist()

        loc = [idx_all.index(idxe) if idxe in idx_all else None for idxe in idx]
        if None in loc:
            idx_none = [idxe for idxe in idx if idxe not in idx_all]
            raise ValueError(f'Var <{self.class_name}.{src}> does not contain value with idx={idx_none}')
        out = getattr(item, attr)[loc]

        if horizon is not None:
            if item.horizon is None:
                raise ValueError(f'horizon is not defined for {self.class_name}.{src}.')
            horizon_all = item.horizon.get_idx()
            if isinstance(horizon, int):
                horizon = [horizon]
            if isinstance(horizon, str):
                horizon = [int(horizon)]
            if isinstance(horizon, np.ndarray):
                horizon = horizon.tolist()
            if isinstance(horizon, list):
                loc_h = [horizon_all.index(idxe) if idxe in horizon_all else None for idxe in horizon]
                if None in loc_h:
                    idx_none = [idxe for idxe in horizon if idxe not in horizon_all]
                    raise ValueError(f'Var <{self.class_name}.{src}> does not contain horizon with idx={idx_none}')
                out = out[:, loc_h]
                if out.shape[1] == 1:
                    out = out[:, 0]
        return out

    def set(self, src: str, idx, attr: str = 'v', value=0.0):
        """
        Set the value of an attribute of a routine parameter.
        """
        if self.__dict__[src].owner is not None:
            # TODO: fit to `_v` type param in the future
            owner = self.__dict__[src].owner
            return owner.set(src=src, idx=idx, attr=attr, value=value)
        else:
            logger.info(f'Variable {self.name} has no owner.')
            # FIXME: add idx for non-grouped variables
            return None

    def doc(self, max_width=78, export='plain'):
        """
        Retrieve routine documentation as a string.
        """
        return self.docum.get(max_width=max_width, export=export)

    def _constr_check(self):
        """
        Chcek if constraints are in-use.
        """
        disabled = []
        for cname, c in self.constrs.items():
            if c.is_disabled:
                disabled.append(cname)
        if len(disabled) > 0:
            logger.warning(f"Disabled constraints: {disabled}")
        return True

    def _data_check(self):
        """
        Check if data is valid for a routine.
        """
        no_input = []
        owner_list = []
        for rname, rparam in self.rparams.items():
            if rparam.owner is not None:
                if rparam.owner.n == 0:
                    no_input.append(rname)
                    owner_list.append(rparam.owner.class_name)
        if len(no_input) > 0:
            logger.error(f"Following models are missing from input file: {set(owner_list)}")
            return False
        # TODO: add data validation for RParam, typical range, etc.
        return True

    def init(self, force=False, no_code=True, **kwargs):
        """
        Setup optimization model.

        Parameters
        ----------
        force: bool
            Whether to force initialization.
        no_code: bool
            Whether to show generated code.
        """
        # TODO: add input check, e.g., if GCost exists
        if not force and self.initialized:
            logger.debug(f'{self.class_name} has already been initialized.')
            return True
        if self._data_check():
            logger.debug(f'{self.class_name} data check passed.')
        else:
            logger.warning(f'{self.class_name} data check failed, setup may run into error!')
        self._constr_check()
        # FIXME: build the system matrices every init might slow down the process
        self.system.mats.make()
        results, elapsed_time = self.om.setup(no_code=no_code)
        common_msg = f"Routine <{self.class_name}> "
        if results:
            msg = f"initialized in {elapsed_time}."
            self.initialized = True
        else:
            msg = "initialization failed!"
        logger.info(common_msg + msg)
        return results

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
        pass
        return True

    def unpack(self, **kwargs):
        """
        Unpack the results.
        """
        return None

    def run(self, force_init=False, no_code=True, **kwargs):
        """
        Run the routine.

        Parameters
        ----------
        force_init: bool
            Whether to force initialization.
        no_code: bool
            Whether to show generated code.
        """
        # --- setup check ---
        self.init(force=force_init, no_code=no_code)
        # NOTE: if the model data is altered, we need to re-setup the model
        # this implementation if not efficient at large-scale
        # FIXME: find a more efficient way to update the OModel values if
        # the system data is altered
        # elif self.exec_time > 0:
        #     self.init(no_code=no_code)
        # --- solve optimization ---
        t0, _ = elapsed()
        _ = self.solve(**kwargs)
        status = self.om.mdl.status
        self.exit_code = self.syms.status[status]
        self.system.exit_code = self.exit_code
        _, s = elapsed(t0)
        self.exec_time = float(s.split(' ')[0])
        sstats = self.om.mdl.solver_stats  # solver stats
        n_iter = int(sstats.num_iters)
        n_iter_str = f"{n_iter} iterations " if n_iter > 1 else f"{n_iter} iteration "
        if self.exit_code == 0:
            msg = f"{self.class_name} solved as {status} in {s}, converged after "
            msg += n_iter_str + f"using solver {sstats.solver_name}."
            logger.warning(msg)
            self.unpack(**kwargs)
            return True
        else:
            msg = f"{self.class_name} failed after "
            msg += n_iter_str + f"using solver {sstats.solver_name}!"
            logger.warning(msg)
            return False

    def summary(self, **kwargs):
        """
        Summary interface
        """
        raise NotImplementedError

    def report(self, **kwargs):
        """
        Report interface.
        """
        raise NotImplementedError

    def __repr__(self):
        return f'{self.class_name} at {hex(id(self))}'

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
            for type in ['constrs', 'vars', 'rparams']:
                if type in self.__dict__:
                    existing_keys += list(self.__dict__[type].keys())
            if key in existing_keys:
                logger.warning(f"{self.class_name}: redefinition of member <{key}>. Likely a modeling error.")

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
        elif isinstance(value, RParam):
            value.id = len(self.rparams)
        self._check_attribute(key, value)
        self._register_attribute(key, value)

        super(RoutineModel, self).__setattr__(key, value)

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to the routine instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        """
        if isinstance(value, (Var, Constraint, Objective)):
            value.om = self.om
        if isinstance(value, Var):
            self.vars[key] = value
        elif isinstance(value, Constraint):
            self.constrs[key] = value
        elif isinstance(value, RParam):
            self.rparams[key] = value
        elif isinstance(value, RBaseService):
            self.services[key] = value

    def __delattr__(self, name):
        """
        Overload the delattr function to unregister attributes.

        Parameters
        ----------
        name: str
            name of the attribute
        """
        self._unregister_attribute(name)
        if name == 'obj':
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
            for n in name:
                if n not in self.constrs:
                    logger.warning(f"Constraint <{n}> not found.")
                    continue
                if not self.constrs[n].is_disabled:
                    logger.warning(f"Constraint <{n}> has already been enabled.")
                    continue
                self.constrs[n].is_disabled = False
                self.initialized = False
            return True

        if name in self.constrs:
            if not self.constrs[name].is_disabled:
                logger.warning(f"Constraint <{name}> has already been enabled.")
            else:
                self.constrs[name].is_disabled = False
                self.initialized = False
                logger.warning(f"Enable constraint <{name}>.")
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
            for n in name:
                if n not in self.constrs:
                    logger.warning(f"Constraint <{n}> not found.")
                    continue
                if self.constrs[n].is_disabled:
                    logger.warning(f"Constraint <{n}> has already been disabled.")
                    continue
                self.constrs[n].is_disabled = True
                self.initialized = False
            return True

        if name in self.constrs:
            if self.constrs[name].is_disabled:
                logger.warning(f"Constraint <{name}> has already been disabled.")
            else:
                self.constrs[name].is_disabled = True
                self.initialized = False
                logger.warning(f"Disable constraint <{name}>.")
            return True

        logger.warning(f"Constraint <{name}> not found.")

    def _post_add_check(self):
        """
        Post-addition check.
        """
        self.initialized = False
        self.exec_time = 0.0
        self.exit_code = 0

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
        return True

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
        item = ValueService(name=name, value=value, tex_name=tex_name, unit=unit,
                            info=info, vtype=vtype, model=model)
        # add the service as an routine attribute
        setattr(self, name, item)

        self._post_add_check()

        return True

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

        return True

    def addVars(self,
                name: str,
                model: Optional[str] = None,
                shape: Optional[Union[int, tuple]] = None,
                tex_name: Optional[str] = None,
                info: Optional[str] = None,
                src: Optional[str] = None,
                unit: Optional[str] = None,
                lb: Optional[str] = None,
                ub: Optional[str] = None,
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
        item = Var(name=name, tex_name=tex_name, info=info, src=src, unit=unit,
                   model=model, shape=shape, lb=lb, ub=ub, horizon=horizon, nonneg=nonneg,
                   nonpos=nonpos, complex=complex, imag=imag, symmetric=symmetric,
                   diag=diag, psd=psd, nsd=nsd, hermitian=hermitian, bool=bool,
                   integer=integer, pos=pos, neg=neg, )

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
                msg = f'Model indicator \'{item.model}\' of <{item.rtn.class_name}.{name}>'
                msg += ' is not a model or group. Likely a modeling error.'
                logger.warning(msg)

        self._post_add_check()

        return True

    def _initial_guess(self):
        """
        Generate initial guess for the optimization model.
        """
        raise NotImplementedError
