"""
Module for optimization models.
"""

import logging

from typing import Optional, Union
from collections import OrderedDict

import numpy as np

from andes.core.common import Config
from andes.core import NumParam
from andes.models.group import GroupBase

from ams.core.var import Algeb
from ams.opt.ovar import OVar

logger = logging.getLogger(__name__)


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

        tex_name = Algeb.tex_name
        mname = Algeb.owner.class_name
        self.tex_name = f'{Algeb.tex_name}_{{{mname}}}' if tex_name else self.name
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
    """

    def __init__(self,
                 AName: str,
                 Group: GroupBase,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 system: Optional = None,
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
    """
    Base class for optimization models.
    """

    def __init__(self):
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = Objective()

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
