"""
Module for optimization models.
"""

import logging

from typing import Optional, Union
from collections import OrderedDict

import numpy as np

from andes.core.common import Config
from andes.core import NumParam

from ams.core.var import OAlgeb
from ams.opt.ovar import OVar
from ams.opt.constraint import Constraint
from ams.opt.objective import Objective

logger = logging.getLogger(__name__)


class OModel:
    """
    Base class for optimization models.
    """

    def __init__(self):
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = None

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
        var = OVar(name=name, type=type, n=n, lb=lb, ub=ub)
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
               ):
        """
        Add variable to optimization model.
        """
        var = OVar(name=name, type=type, n=n, lb=lb, ub=ub)
        self.vars[name] = var
        setattr(self, name, var)
        return var

    def AddOConstrs(self,
                    name: Optional[str] = None,
                    n: Optional[int] = 1,
                    expr: Optional[str] = None,
                    ):
        """
        Add constraints to optimization model from OAlgeb.
        """
        ub = np.array([np.inf] * n)
        constr = Constraint(name=name, n=n, expr=expr)
        self.constrs[name] = constr
        setattr(self, name, constr)
        return constr

    def AddConstr(self, *args, **kwargs):
        """
        Add constraint to optimization model.
        """
        pass

    def setObjective(self, *args, **kwargs):
        self.objectives.add(*args, **kwargs)
