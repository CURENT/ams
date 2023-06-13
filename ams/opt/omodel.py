"""
Module for optimization models.
"""

import logging

from typing import Optional, Union
from collections import OrderedDict
import re

import numpy as np

from andes.core.common import Config
from andes.core import BaseParam, DataParam, IdxParam, NumParam
from andes.models.group import GroupBase

from ams.core.param import RParam
from ams.core.var import Algeb

from ams.utils import timer

import cvxpy as cp

logger = logging.getLogger(__name__)


class Var(Algeb):
    """
    Class for variables used in a routine.

    Parameters
    ----------
    info : str, optional
        Descriptive information
    unit : str, optional
        Unit
    tex_name : str
        LaTeX-formatted variable symbol. If is None, the value of `name` will be
        used.
    discrete : Discrete
        Discrete component on which this variable depends. ANDES will call
        `check_var()` of the discrete component before initializing this
        variable.
    name : str, optional
        Variable name. One should typically assigning the name directly because
        it will be automatically assigned by the model. The value of ``name``
        will be the symbol name to be used in expressions.
    is_group : bool, optional
        Indicate if this variable is a group variable. If True, `group_name`
        must be provided.
    group_name : str, optional
        Name of the group. Must be provided if `is_group` is True.

    Attributes
    ----------
    a : array-like
        variable address
    v : array-like
        local-storage of the variable value
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 src: Optional[str] = None,
                 unit: Optional[str] = None,
                 owner_name: Optional[str] = None,
                 lb: Optional[str] = None,
                 ub: Optional[str] = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, info=info, unit=unit)
        self.src = name if (src is None) else src
        self.is_group = False
        self.owner_name = owner_name  # indicate if this variable is a group variable
        self.owner = None  # instance of the owner model or group
        self.lb = lb
        self.ub = ub
        self.id = None     # variable internal index inside a model (assigned in run time)

        # TODO: set a
        # address into the variable and equation arrays (dae.f/dae.g and dae.x/dae.y)
        self.a: np.ndarray = np.array([], dtype=int)

        self.v: np.ndarray = np.array([], dtype=float)  # variable value array

    def get_idx(self):
        if self.is_group:
            return self.owner.get_idx()
        else:
            return self.owner.idx.v

    @property
    def n(self):
        return self.owner.n

    def __repr__(self):
        if self.owner.n == 0:
            span = []

        elif 1 <= self.owner.n <= 20:
            span = f'a={self.a}, v={self.v}'

        else:
            span = []
            span.append(self.a[0])
            span.append(self.a[-1])
            span.append(self.a[1] - self.a[0])
            span = ':'.join([str(i) for i in span])
            span = 'a=[' + span + ']'

        return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}, {span}'

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__


class Constraint:
    """
    Base class for constraints.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 info: Optional[str] = None,
                 type: Optional[str] = 'uq',
                 ):
        self.name = name
        self.e_str = e_str
        self.info = info
        self.type = type  # TODO: determine constraint type
        # TODO: add constraint info from solver

    @property
    def class_name(self):
        return self.__class__.__name__

    def __repr__(self):
        name = self.name if self.name is not None else 'Unnamed constr'
        return f"{name}: {self.e_str}"


class Objective:
    """
    Base class for objective functions.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 info: Optional[str] = None,
                 sense: Optional[str] = 'min'):
        self.name = name
        self.e_str = e_str
        self.info = info
        self.sense = sense
        self.v = None  # objective value

    @property
    def class_name(self):
        return self.__class__.__name__

    def __repr__(self):
        if self.name is not None:
            if self.v is not None:
                return f"{self.name}: {self.e_str}, {self.name}={self.v:.4f}"
            else:
                return f"{self.name}: {self.e_str}"
        else:
            if self.v is not None:
                return f"{self.e_str}={self.v:.4f}"
            else:
                return f"Unnamed obj: {self.e_str}"


class OModel:
    r"""
    Base class for optimization models.

    Parameters
    ----------
    routine: Routine
        Routine that to be modeled.

    Attributes
    ----------
    mdl: cvxpy.Problem
        Optimization model.
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
        self.routine = routine
        self.mdl = None
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = None
        self.n = 0  # number of decision variables
        self.m = 0  # number of constraints

    @timer
    def setup(self):
        """
        Setup the optimziation model from symbolic description.

        Decision variables are the ``Var`` of a routine.
        For example, the power outputs ``pg`` of routine ``DCOPF``.
        """
        self.routine.syms.generate_symbols()
        # --- add decision variables ---
        for ovname, ovar in self.routine.vars.items():
            self.parse_var(ovar=ovar,
                           sub_map=self.routine.syms.sub_map)
            self.n += ovar.n

        # --- parse constraints ---
        for cname, constr in self.routine.constrs.items():
            self.parse_constr(constr=constr,
                              sub_map=self.routine.syms.sub_map)
            self.m += self.constrs[cname].size

        # --- parse objective functions ---
        if self.routine.obj is not None:
            self.parse_obj(obj=self.routine.obj,
                           sub_map=self.routine.syms.sub_map)
            # --- finalize the optimziation formulation ---
            code_mdl = f"problem(self.obj, [constr for constr in self.constrs.values()])"
            for pattern, replacement, in self.routine.syms.sub_map.items():
                code_mdl = re.sub(pattern, replacement, code_mdl)
            code_mdl = "self.mdl=" + code_mdl
            exec(code_mdl)
        else:
            logger.warning(f"{self.routine.class_name} has no objective function.")
        return True

    @property
    def class_name(self):
        return self.__class__.__name__

    def parse_var(self,
                  ovar: Var,
                  sub_map: OrderedDict,
                  ):
        """
        Parse the decision variables from symbolic dispatch model.

        Parameters
        ----------
        var : Var
            The routine Var
        sub_map : OrderedDict
            A dictionary of substitution map, generated by symprocessor.
        """
        # only used for CVXPY
        code_var = "tmp=var(ovar.n, boolean=(ovar.unit == 'bool'))"
        for pattern, replacement, in sub_map.items():
            code_var = re.sub(pattern, replacement, code_var)
        exec(code_var)
        exec("setattr(self, ovar.name, tmp)")
        exec("self.vars[ovar.name] = tmp")
        if ovar.lb:
            lv = ovar.lb.owner.get(src=ovar.lb.name, idx=ovar.get_idx(), attr='v')
            u = ovar.lb.owner.get(src='u', idx=ovar.get_idx(), attr='v')
            elv = u * lv
            exec("self.constrs[ovar.lb.name] = tmp >= elv")
            self.m += ovar.lb.owner.n
        if ovar.ub:
            uv = ovar.ub.owner.get(src=ovar.ub.name, idx=ovar.get_idx(), attr='v')
            u = ovar.lb.owner.get(src='u', idx=ovar.get_idx(), attr='v')
            euv = u * uv
            exec("self.constrs[ovar.ub.name] = tmp <= euv")
            self.m += ovar.ub.owner.n

    def parse_obj(self,
                  obj: Objective,
                  sub_map: OrderedDict,
                  ):
        """
        Parse the objective function from symbolic dispatch model.

        Parameters
        ----------
        obj : Objective
            The routine Objective
        sub_map : OrderedDict
            A dictionary of substitution map, generated by symprocessor.
        """
        code_obj = obj.e_str
        for pattern, replacement, in sub_map.items():
            code_obj = re.sub(pattern, replacement, code_obj)
        if obj.sense == 'min':
            code_obj = f'cp.Minimize({code_obj})'
        elif obj.sense == 'max':
            code_obj = f'cp.Maximize({code_obj})'
        else:
            raise ValueError(f'Objective sense {obj.sense} is not supported.')
        code_obj = 'self.obj=' + code_obj
        exec(code_obj)
        return True

    def parse_constr(self,
                     constr: Constraint,
                     sub_map: OrderedDict,
                     ):
        """
        Parse the constraint from symbolic dispatch model.

        Parameters
        ----------
        constr : Constraint
            The routine Constraint
        sub_map : OrderedDict
            A dictionary of substitution map, generated by symprocessor.
        """
        code_constr = constr.e_str
        for pattern, replacement in sub_map.items():
            code_constr = re.sub(pattern, replacement, code_constr)
        if constr.type == 'uq':
            code_constr = f'{code_constr} <= 0'
        elif constr.type == 'eq':
            code_constr = f'{code_constr} == 0'
        else:
            raise ValueError(f'Objective sense {self.routine.obj.sense} is not supported.')
        code_constr = f'self.constrs["{constr.name}"]=' + code_constr
        logger.debug(f"Set constrs {constr.info}: {code_constr}")
        exec(code_constr)
