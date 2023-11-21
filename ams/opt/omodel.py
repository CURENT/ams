"""
Module for optimization modeling.
"""
import logging

from typing import Any, Optional, Union
from collections import OrderedDict
import re

import numpy as np

from andes.core.common import Config

from ams.utils import timer

import cvxpy as cp

logger = logging.getLogger(__name__)


class OptzBase:
    """
    Base class for optimization elements, e.g., Var and Constraint.

    Parameters
    ----------
    name : str, optional
        Name.
    info : str, optional
        Descriptive information

    Attributes
    ----------
    rtn : ams.routines.Routine
        The owner routine instance.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 ):
        self.om = None
        self.name = name
        self.info = info
        self.unit = unit
        self.is_disabled = False
        self.rtn = None
        self.optz = None  # corresponding optimization element

    def parse(self):
        """
        Parse the object.
        """
        raise NotImplementedError

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    @property
    def n(self):
        """
        Return the number of elements.
        """
        if self.owner is None:
            return len(self.v)
        else:
            return self.owner.n

    @property
    def shape(self):
        """
        Return the shape.
        """
        try:
            return self.om.__dict__[self.name].shape
        except KeyError:
            logger.warning('Shape info is not ready before initialziation.')
            return None

    @property
    def size(self):
        """
        Return the size.
        """
        if self.rtn.initialized:
            return self.om.__dict__[self.name].size
        else:
            logger.warning(f'<{self.rtn.class_name}> is not initialized yet.')
            return None


class Param(OptzBase):
    """
    Base class for parameters used in a routine.

    Parameters
    ----------
    no_parse: bool, optional
        True to skip parsing the parameter.
    nonneg: bool, optional
        True to set the parameter as non-negative.
    nonpos: bool, optional
        True to set the parameter as non-positive.
    complex: bool, optional
        True to set the parameter as complex.
    imag: bool, optional
        True to set the parameter as imaginary.
    symmetric: bool, optional
        True to set the parameter as symmetric.
    diag: bool, optional
        True to set the parameter as diagonal.
    hermitian: bool, optional
        True to set the parameter as hermitian.
    boolean: bool, optional
        True to set the parameter as boolean.
    integer: bool, optional
        True to set the parameter as integer.
    pos: bool, optional
        True to set the parameter as positive.
    neg: bool, optional
        True to set the parameter as negative.
    sparsity: list, optional
        Sparsity pattern of the parameter.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 no_parse: Optional[bool] = False,
                 nonneg: Optional[bool] = False,
                 nonpos: Optional[bool] = False,
                 complex: Optional[bool] = False,
                 imag: Optional[bool] = False,
                 symmetric: Optional[bool] = False,
                 diag: Optional[bool] = False,
                 hermitian: Optional[bool] = False,
                 boolean: Optional[bool] = False,
                 integer: Optional[bool] = False,
                 pos: Optional[bool] = False,
                 neg: Optional[bool] = False,
                 sparsity: Optional[list] = None,
                 ):
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self.no_parse = no_parse  # True to skip parsing the parameter
        self.sparsity = sparsity  # sparsity pattern

        self.config = Config(name=self.class_name)  # `config` that can be exported

        self.config.add(OrderedDict((('nonneg', nonneg),
                                     ('nonpos', nonpos),
                                     ('complex', complex),
                                     ('imag', imag),
                                     ('symmetric', symmetric),
                                     ('diag', diag),
                                     ('hermitian', hermitian),
                                     ('boolean', boolean),
                                     ('integer', integer),
                                     ('pos', pos),
                                     ('neg', neg),
                                     )))

    def parse(self):
        """
        Parse the parameter.
        """
        config = self.config.as_dict()  # NOQA
        sub_map = self.om.rtn.syms.sub_map
        shape = np.shape(self.v)
        code_param = f"tmp=param(shape={shape}, **config)"
        for pattern, replacement, in sub_map.items():
            code_param = re.sub(pattern, replacement, code_param)
        exec(code_param)
        exec("self.optz=tmp")
        try:
            exec("self.optz.value = self.v")
        except ValueError:
            msg = f"Parameter <{self.name}> has non-numeric value, "
            msg += "no_parse=True is applied."
            logger.warning(msg)
            self.no_parse = True
            return False
        return True

    def update(self):
        """
        Update the parameter value from RParam, MParam, or Service.
        """
        self.optz.value = self.v

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.name}'


class Var(OptzBase):
    """
    Base class for variables used in a routine.

    When `horizon` is provided, the variable will be expanded to a matrix,
    where rows are indexed by the source variable index and columns are
    indexed by the horizon index.

    Parameters
    ----------
    info : str, optional
        Descriptive information
    unit : str, optional
        Unit
    tex_name : str
        LaTeX-formatted variable symbol. If is None, the value of `name` will be
        used.
    name : str, optional
        Variable name. One should typically assigning the name directly because
        it will be automatically assigned by the model. The value of ``name``
        will be the symbol name to be used in expressions.
    src : str, optional
        Source variable name. If is None, the value of `name` will be used.
    model : str, optional
        Name of the owner model or group.
    lb : str, optional
        Lower bound
    ub : str, optional
        Upper bound
    ctrl : str, optional
        Controllability
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
    boolean : bool, optional
        Boolean variable
    integer : bool, optional
        Integer variable
    pos : bool, optional
        Positive variable
    neg : bool, optional
        Negative variable

    Attributes
    ----------
    a : array-like
        variable address
    v : array-like
        local-storage of the variable value
    rtn : ams.routines.Routine
        The owner routine instance.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 src: Optional[str] = None,
                 unit: Optional[str] = None,
                 model: Optional[str] = None,
                 shape: Optional[Union[tuple, int]] = None,
                 lb=None,
                 ub=None,
                 ctrl: Optional[str] = None,
                 v0: Optional[str] = None,
                 horizon=None,
                 nonneg: Optional[bool] = False,
                 nonpos: Optional[bool] = False,
                 complex: Optional[bool] = False,
                 imag: Optional[bool] = False,
                 symmetric: Optional[bool] = False,
                 diag: Optional[bool] = False,
                 psd: Optional[bool] = False,
                 nsd: Optional[bool] = False,
                 hermitian: Optional[bool] = False,
                 boolean: Optional[bool] = False,
                 integer: Optional[bool] = False,
                 pos: Optional[bool] = False,
                 neg: Optional[bool] = False,
                 ):
        self.name = name
        self.info = info
        self.unit = unit

        self.tex_name = tex_name if tex_name else name
        self.owner = None  # instance of the owner Model
        self.id = None     # variable internal index inside a model (assigned in run time)
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self.src = src
        self.is_group = False
        self.model = model  # indicate if this variable is a group variable
        self.owner = None  # instance of the owner model or group
        self.lb = lb
        self.ub = ub
        self.ctrl = ctrl
        self.v0 = v0
        self.horizon = horizon
        self._shape = shape
        self._v = None

        self.optz_lb = None
        self.optz_ub = None

        self.config = Config(name=self.class_name)  # `config` that can be exported

        self.config.add(OrderedDict((('nonneg', nonneg),
                                     ('nonpos', nonpos),
                                     ('complex', complex),
                                     ('imag', imag),
                                     ('symmetric', symmetric),
                                     ('diag', diag),
                                     ('psd', psd),
                                     ('nsd', nsd),
                                     ('hermitian', hermitian),
                                     ('boolean', boolean),
                                     ('integer', integer),
                                     ('pos', pos),
                                     ('neg', neg),
                                     )))

        self.id = None     # variable internal index inside a model (assigned in run time)

        self.a: np.ndarray = np.array([], dtype=int)

    @property
    def v(self):
        """
        Return the CVXPY variable value.
        """
        if self.name in self.om.vars:
            out = self.om.vars[self.name].value if self._v is None else self._v
        else:
            out = None
        return out

    @v.setter
    def v(self, value):
        self._v = value

    def get_idx(self):
        if self.is_group:
            return self.owner.get_idx()
        elif self.owner is None:
            logger.info(f'Variable <{self.name}> has no owner.')
            return None
        else:
            return self.owner.idx.v

    def parse(self):
        """
        Parse the variable.
        """
        sub_map = self.om.rtn.syms.sub_map
        # only used for CVXPY
        # NOTE: Config only allow lower case letters, do a conversion here
        config = {}
        for k, v in self.config.as_dict().items():
            if k == 'psd':
                config['PSD'] = v
            elif k == 'nsd':
                config['NSD'] = v
            elif k == 'bool':
                config['boolean'] = v
            else:
                config[k] = v
        # NOTE: number of rows is the size of the source variable
        if self.owner is not None:
            nr = self.owner.n
            nc = 0
            if self.horizon:
                # NOTE: numer of columns is the horizon if exists
                nc = int(self.horizon.n)
                shape = (nr, nc)
            else:
                shape = (nr,)
        elif isinstance(self._shape, int):
            shape = (self._shape,)
            nr = shape
            nc = 0
        elif isinstance(self._shape, tuple):
            shape = self._shape
            nr = shape[0]
            nc = shape[1] if len(shape) > 1 else 0
        else:
            raise ValueError(f"Invalid shape {self._shape}.")
        code_var = f"tmp=var({shape}, **config)"
        for pattern, replacement, in sub_map.items():
            code_var = re.sub(pattern, replacement, code_var)
        exec(code_var)  # build the Var object
        exec("self.optz = tmp")  # assign the to the optz attribute
        u_ctrl = self.ctrl.v if self.ctrl else np.ones(nr)
        v0 = self.v0.v if self.v0 else np.zeros(nr)
        if self.lb:
            lv = self.lb.owner.get(src=self.lb.name, idx=self.get_idx(), attr='v')
            u = self.lb.owner.get(src='u', idx=self.get_idx(), attr='v')
            # element-wise lower bound considering online status
            elv = u_ctrl * u * lv + (1 - u_ctrl) * v0
            # fit variable shape if horizon exists
            elv = np.tile(elv, (nc, 1)).T if nc > 0 else elv
            exec("self.optz_lb = tmp >= elv")
        if self.ub:
            uv = self.ub.owner.get(src=self.ub.name, idx=self.get_idx(), attr='v')
            u = self.lb.owner.get(src='u', idx=self.get_idx(), attr='v')
            # element-wise upper bound considering online status
            euv = u_ctrl * u * uv + (1 - u_ctrl) * v0
            # fit variable shape if horizon exists
            euv = np.tile(euv, (nc, 1)).T if nc > 0 else euv
            exec("self.optz_ub = tmp <= euv")
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}'


class Constraint(OptzBase):
    """
    Base class for constraints.

    This class is used as a template for defining constraints. Each
    instance of this class represents a single constraint.

    Parameters
    ----------
    name : str, optional
        A user-defined name for the constraint.
    e_str : str, optional
        A mathematical expression representing the constraint.
    info : str, optional
        Additional informational text about the constraint.
    type : str, optional
        The type of constraint, defaults to 'uq'. It might represent
        different types of mathematical relationships such as equality or
        inequality.

    Attributes
    ----------
    name : str or None
        Name of the constraint.
    e_str : str or None
        Expression string of the constraint.
    info : str or None
        Additional information about the constraint.
    type : str
        Type of the constraint.
    rtn : ams.routines.Routine
        The owner routine instance.
    is_disabled : bool
        Flag indicating if the constraint is disabled, False by default.

    Notes
    -----
    - The attribute 'type' needs to be properly handled with predefined types.
    - There is also a TODO for incorporating constraint information from the solver.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 info: Optional[str] = None,
                 type: Optional[str] = 'uq',
                 ):
        OptzBase.__init__(self, name=name, info=info)
        self.e_str = e_str
        self.type = type  # TODO: determine constraint type
        self.is_disabled = False
        # TODO: add constraint info from solver

    def parse(self, no_code=True):
        """
        Parse the constraint.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the code should be shown, True by default.
        """
        sub_map = self.om.rtn.syms.sub_map
        if self.is_disabled:
            return True
        code_constr = self.e_str
        for pattern, replacement in sub_map.items():
            try:
                code_constr = re.sub(pattern, replacement, code_constr)
            except TypeError as e:
                logger.error(f"Error in parsing constr <{self.name}>.")
                raise e
        if self.type == 'uq':
            code_constr = f'tmp = {code_constr} <= 0'
        elif self.type == 'eq':
            code_constr = f'tmp = {code_constr} == 0'
        else:
            raise ValueError(f'Constraint type {self.type} is not supported.')
        logger.debug(f"Set constrs {self.name}: {self.e_str} {'<= 0' if self.type == 'uq' else '== 0'}")
        if not no_code:
            logger.info(f"Code Constr: {code_constr}")
        exec(code_constr)
        exec("self.optz = tmp")
        return True

    def __repr__(self):
        enabled = 'ON' if self.name in self.om.constrs else 'OFF'
        out = f"[{enabled}]: {self.e_str}"
        out += " =0" if self.type == 'eq' else " <=0"
        return out

    @property
    def v(self):
        """
        Return the CVXPY constraint LHS value.
        """
        if self.name in self.om.constrs:
            return self.om.constrs[self.name]._expr.value
        else:
            return None


class Objective(OptzBase):
    """
    Base class for objective functions.

    This class serves as a template for defining objective functions. Each
    instance of this class represents a single objective function that can
    be minimized or maximized depending on the sense ('min' or 'max').

    Parameters
    ----------
    name : str, optional
        A user-defined name for the objective function.
    e_str : str, optional
        A mathematical expression representing the objective function.
    info : str, optional
        Additional informational text about the objective function.
    sense : str, optional
        The sense of the objective function. It should be either 'min'
        for minimization or 'max' for maximization. Default is 'min'.

    Attributes
    ----------
    name : str or None
        Name of the objective function.
    e_str : str or None
        Expression string of the objective function.
    info : str or None
        Additional information about the objective function.
    sense : str
        Sense of the objective function ('min' or 'max').
    v : NoneType
        The value of the objective function. It needs to be set through
        computation.
    rtn : ams.routines.Routine
        The owner routine instance.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 sense: Optional[str] = 'min'):
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self.e_str = e_str
        self.sense = sense
        self._v = None

    @property
    def v(self):
        """
        Return the CVXPY objective value.
        """
        out = self.om.obj.value
        out = self._v if out is None else out
        return out

    @v.setter
    def v(self, value):
        self._v = value

    def parse(self, no_code=True):
        """
        Parse the objective function.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the code should be shown, True by default.
        """
        sub_map = self.om.rtn.syms.sub_map
        code_obj = self.e_str
        for pattern, replacement, in sub_map.items():
            code_obj = re.sub(pattern, replacement, code_obj)
        if self.sense == 'min':
            code_obj = f'cp.Minimize({code_obj})'
        elif self.sense == 'max':
            code_obj = f'cp.Maximize({code_obj})'
        else:
            raise ValueError(f'Objective sense {self.sense} is not supported.')
        code_obj = 'self.om.obj=' + code_obj
        logger.debug(f"Set obj {self.name}: {self.sense}. {self.e_str}")
        if not no_code:
            logger.info(f"Code Obj: {code_obj}")
        exec(code_obj)
        return True

    def __repr__(self):
        name_str = f"{self.name}=" if self.name is not None else "obj="
        value_str = f"{self.v:.4f}, " if self.v is not None else ""
        return f"{name_str}{value_str}{self.e_str}"


class OModel:
    """
    Base class for optimization models.

    Parameters
    ----------
    routine: Routine
        Routine that to be modeled.

    Attributes
    ----------
    mdl: cvxpy.Problem
        Optimization model.
    params: OrderedDict
        Parameters.
    vars: OrderedDict
        Decision variables.
    constrs: OrderedDict
        Constraints.
    obj: Objective
        Objective function.
    n: int
        Number of decision variables.
    m: int
        Number of constraints.
    """

    def __init__(self, routine):
        self.rtn = routine
        self.mdl = None
        self.params = OrderedDict()
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = None
        self.n = 0  # number of decision variables
        self.m = 0  # number of constraints

    @timer
    def setup(self, no_code=True, force_generate=False):
        """
        Setup the optimziation model from symbolic description.

        Decision variables are the ``Var`` of a routine.
        For example, the power outputs ``pg`` of routine ``DCOPF``
        are decision variables.

        Disabled constraints (indicated by attr ``is_disabled``) will not
        be added to the optimization model.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the code should be shown, True by default.
        force : bool, optional
            True to force generating symbols, False by default.
        """
        rtn = self.rtn
        rtn.syms.generate_symbols(force_generate=force_generate)
        # --- add RParams and Services as parameters ---
        for key, val in rtn.params.items():
            # logger.debug(f"Parsing param {key}")
            if not val.no_parse:
                val.parse()
                setattr(self, key, val.optz)
        # --- add decision variables ---
        for key, val in rtn.vars.items():
            val.parse()
            setattr(self, key, val.optz)
        # --- add constraints ---
        for key, val in rtn.constrs.items():
            val.parse(no_code=no_code)
            setattr(self, key, val.optz)
        # --- parse objective functions ---
        if rtn.type == 'PF':
            # NOTE: power flow type has no objective function
            pass
        elif rtn.obj is not None:
            rtn.obj.parse(no_code=no_code)
            # --- finalize the optimziation formulation ---
            code_mdl = "problem(self.obj, [constr for constr in self.constrs.values()])"
            for pattern, replacement in self.rtn.syms.sub_map.items():
                code_mdl = re.sub(pattern, replacement, code_mdl)
            code_mdl = "self.mdl=" + code_mdl
            exec(code_mdl)

        # --- count ---
        n_list = [cpvar.size for cpvar in self.vars.values()]
        self.n = np.sum(n_list)  # number of decision variables
        m_list = [cpconstr.size for cpconstr in self.constrs.values()]
        self.m = np.sum(m_list)  # number of constraints

        if rtn.type != 'PF' and rtn.obj is None:
            logger.warning(f"{rtn.class_name} has no objective function.")
            return False

        return True

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    def __register_attribute(self, key, value):
        """
        Register a pair of attributes to OModel instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        """
        if isinstance(value, cp.Variable):
            self.vars[key] = value
        elif isinstance(value, cp.Constraint):
            self.constrs[key] = value
        elif isinstance(value, cp.Parameter):
            self.params[key] = value

    def __setattr__(self, __name: str, __value: Any):
        self.__register_attribute(__name, __value)
        super().__setattr__(__name, __value)

    def update_param(self, params=Optional[Union[Param, str, list]]):
        """
        Update the Parameter values.
        """
        if params is None:
            for _, val in self.params.items():
                val.update()
        elif isinstance(params, Param):
            params.update()
        elif isinstance(params, str):
            self.params[params].update()
        elif isinstance(params, list):
            for param in params:
                param.update()
        return True

    def __repr__(self) -> str:
        return f'{self.rtn.class_name}.{self.__class__.__name__} at {hex(id(self))}'
