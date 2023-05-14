"""
Module for optimization models.
"""

import logging

from typing import Optional, Union
from collections import OrderedDict

import numpy as np

from andes.core.common import Config
from andes.core import BaseParam, DataParam, IdxParam, NumParam
from andes.models.group import GroupBase

from ams.core.var import Algeb

logger = logging.getLogger(__name__)


class OVar:
    """
    Decision variables in optimization.
    """

    def __init__(self,
                 name: str,
                 type: Optional[type] = np.float64,
                 n: Optional[int] = 1,
                 lb: Optional[np.ndarray] = - np.inf,
                 ub: Optional[np.ndarray] = np.inf,
                 info: Optional[str] = None,
                 ):
        """
        Decision variables in optimization.

        Parameters
        ----------
        name: str
            Name of the variable.
        type: type, optional
            Type of the variable, by default np.float64
        n: int, optional
            Number of variables, by default 1
        lb: np.ndarray, optional
            Lower bound of the variable, by default - np.inf
        ub: np.ndarray, optional
            Upper bound of the variable, by default np.inf
        info: str, optional
            Information of the variable, by default None
        """
        self.name = name
        self.type = type
        self.n = n
        # TODO: add sanity check for lb and ub
        type_ndarray = isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray)
        type_float = isinstance(lb, float) and isinstance(ub, float)
        self.lb = lb
        self.ub = ub
        self.info = info
        self.v = np.empty(n)

    @property
    def class_name(self):
        return self.__class__.__name__

    def __repr__(self):
        dev_text = 'OVar' if self.n == 1 else 'OVars'
        return f'{self.name} ({self.n} {dev_text}) at {hex(id(self))}'


class OParam:
    """
    Class for parameters in a routine.

    This class is used to store the ``tex_name`` only.

    Parameters:
    -----------
    Param : Union[BaseParam, DataParam, IdxParam, NumParam]
        The parameter object associated with this OParam instance.

    Attributes:
    ------------
    Param : Union[BaseParam, DataParam, IdxParam, NumParam]
        The parameter object associated with this OParam instance.
    name : str
        The name of the parameter.
    tex_name : str
        The LaTeX representation of the parameter name, possibly including
        additional information based on the owner.
    """

    def __init__(self,
                 Param: Union[BaseParam, DataParam, IdxParam, NumParam],
                 ) -> None:
        self.Param = Param
        self.name = Param.name + Param.owner.class_name

        mname = Param.owner.class_name
        if "_" in Param.tex_name:
            tex_name_parts = Param.tex_name.split("_")
            tex_name = tex_name_parts[0] + "_{" + f"{tex_name_parts[1]}, {mname}" + "}"
        else:
            tex_name = f'{Param.tex_name}_{{{mname}}}' if Param.tex_name else self.name
        self.tex_name = tex_name


class OAlgeb:
    """ 
    Class for algebraic variable in a routine.

    This class is an extension of ``Algeb`` that revise the ``tex_name`` and keep a copy of the value
    so the value can be accessed if other routiens are called.

    In ``ams.system.System.init_algebs()``, all the ``Algeb`` from models are registered
    as an ``OAlgeb`` in the routiens.
    The ``OAlgeb`` can be used in the ``Routine`` to formulate optimization problems and store the
    solved values from the ``Algeb`` before they are overwritted by orther routines.
    """

    def __init__(self,
                 Algeb: Algeb,
                 ):
        self.Algeb = Algeb
        self.name = Algeb.name + Algeb.owner.class_name
        self.info = Algeb.info
        self.unit = Algeb.unit
        self.type = np.float64 if self.unit == 'bool' else np.int64

        mname = Algeb.owner.class_name
        if "_" in Algeb.tex_name:
            tex_name_parts = Algeb.tex_name.split("_")
            tex_name = tex_name_parts[0] + "_{" + f"{tex_name_parts[1]}, {mname}" + "}"
        else:
            tex_name = f'{Algeb.tex_name}_{{{mname}}}' if Algeb.tex_name else self.name
        self.tex_name = tex_name

        self.owner = Algeb.owner  # instance of the owner Model
        self.v = np.empty(self.owner.n)  # variable value

    def __repr__(self):
        n = self.owner.n
        dev_text = 'OAlgeb' if n == 1 else 'OAlgebs'
        return f'{self.owner.class_name}.{self.name} ({n} {dev_text}) at {hex(id(self))}'


class OAlgebs:
    """ 
    Class for algebraic variable in a routine.

    This class is an extension of ``OAlgeb`` that combines same ``Algeb`` from one group together.
    
    TODO: deprecate this class for now.
    """

    def __init__(self,
                 AName: str,
                 Group: GroupBase,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 system = None,
                 ):
        self.AName = AName
        self.Group = Group
        self.name = AName + Group.class_name
        self.info = info
        self.unit = unit
        self.type = np.float64 if self.unit == 'bool' else np.int64
        self.tex_name = f'{Algeb.tex_name}_{{{Group.class_name}}}' if tex_name else self.name
        self.system = system  # instance of the owner Model

        self.v = np.empty(self.owner.n)  # variable value

    def __repr__(self):
        n = self.owner.n
        dev_text = 'OAlgeb' if n == 1 else 'OAlgebs'
        return f'{self.owner.class_name}.{self.name} ({n} {dev_text}) at {hex(id(self))}'


class Constraint:
    """
    Base class for constraints.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 n: Optional[int] = 1,
                 expr: Optional[str] = None,
                 type: Optional[str] = 'uq',
                 info: Optional[str] = None,
                 ):
        self.name = name
        self.n = n
        self.expr = expr
        self.type = type
        self.info = info

    @property
    def class_name(self):
        return self.__class__.__name__

    def __repr__(self):
        dev_text = 'Constraint' if self.n == 1 else 'Constraints'
        return f'{self.name} ({self.n} {dev_text}) at {hex(id(self))}'


class Objective:
    """
    Base class for objective functions.
    """

    def __init__(self):
        pass

    def set(self,
            expr=None,
            sense='min'):
        """
        Set objective functions.
        """
        self.expr = expr
        self.sense = sense


class OModel:
    r"""
    Base class for optimization models.
    The optimziation problem is formulated as:

    .. math::
        \min_x \ & c^T x \\
        \mbox{such that} \ & A_{ub} x \leq b_{ub},\\
        & A_{eq} x = b_{eq},\\
        & l \leq x \leq u ,

    where :math:`x` is a vector of decision variables; :math:`c`,
    :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    # TODO: include integrality parameters.

    The defined arrays and descriptions are as follows:

    +-----------+---------------------------------------------+
    |   Array   |                 Description                 |
    +===========+=============================================+
    |     c     | Array for decision variable coefficients    |
    +-----------+---------------------------------------------+
    |    Aub    | Array for inequality coefficients           |
    +-----------+---------------------------------------------+
    |    Aeq    | Array for equality coefficients             |
    +-----------+---------------------------------------------+
    |    bub    | Array for inequality upper bounds           |
    +-----------+---------------------------------------------+
    |    beq    | Array for equality bounds                   |
    +-----------+---------------------------------------------+
    |     lb    | Array for decision variable lower bounds    |
    +-----------+---------------------------------------------+
    |     ub    | Array for decision variable upper bounds    |
    +-----------+---------------------------------------------+

    """

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    def __init__(self, routine=None):
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = Objective()
        self.routine = routine

        self.c = np.array([])
        self.Aub, self.Aeq = np.array([]), np.array([])
        self.bub, self.beq = np.array([]), np.array([])
        self.lb, self.ub = np.array([]), np.array([])

    def __repr__(self):
        n_vars = len(self.vars)
        n_constrs = len(self.constrs)
        class_name = self.__class__.__name__
        rtn_name = self.routine.__class__.__name__
        repr_msg = f"Routine {rtn_name} {class_name}: {n_vars} vars, {n_constrs} constrs."
        return repr_msg

    def init(self):
        pass

    def AddOVars(self,
                 OAlgeb: Union[OAlgeb, str],
                 lb: Optional[Union[NumParam, str]] = None,
                 ub: Optional[Union[NumParam, str]] = None,
                 ):
        """
        Add variables to optimization model from OAlgeb.
        """
        name = OAlgeb.name
        type = OAlgeb.type
        n = OAlgeb.owner.n
        lb = np.array([- np.inf] * n) if lb is None else lb.v
        ub = np.array([np.inf] * n) if ub is None else ub.v
        var = OVar(name=name, type=type, n=n, lb=lb, ub=ub, info=OAlgeb.info)
        self.vars[name] = var
        # TODO: translate var bounds into constraints
        setattr(self, name, var)
        return var

    def AddVar(self,
               name='var',
               type: Optional[type] = np.float64,
               n: Optional[int] = 1,
               lb: Optional[np.ndarray] = - np.inf,
               ub: Optional[np.ndarray] = np.inf,
               info: Optional[str] = None,
               ):
        """
        Add variable to optimization model.
        """
        var = OVar(name=name, type=type, n=n, lb=lb, ub=ub)
        self.vars[name] = var
        setattr(self, name, var)
        return var

    def AddConstrs(self,
                   name: Optional[str] = None,
                   n: Optional[int] = 1,
                   expr: Optional[str] = None,
                   type: Optional[str] = 'uq',
                   info: Optional[str] = None,
                   ):
        """
        Add constraints to optimization model, in the format
        of ``expr <= 0`` or ``expr == 0``.

        Parameters
        ----------
        name : str, optional
            Name of the constraint, by default None
        n : int, optional
            Number of constraints, by default 1
        expr : str, optional
            LHS of the constraint, by default None
        type : str, optional
            Type of the constraint, by default 'uq'
        info : str, optional
            Description of the constraint, by default None
        """
        ub = np.array([np.inf] * n)
        constr = Constraint(name=name, n=n, expr=expr, type=type, info=info)
        self.constrs[name] = constr
        setattr(self, name, constr)
        return constr

    def setObjective(self, *args, **kwargs):
        self.obj.set(*args, **kwargs)
