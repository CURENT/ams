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

from ams.core.param import RParam
from ams.core.var import Algeb, RAlgeb

from ams.utils import timer

import cvxpy as cp

logger = logging.getLogger(__name__)


class Var:
    """
    Decision variables in optimization.
    """

    def __init__(self,
                 name: str,
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
        self.lb = lb
        self.ub = ub
        self.info = info

        self.type = None  # TODO: var type
        # TODO: add sanity check for lb and ub
        type_ndarray = isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray)
        type_float = isinstance(lb, float) and isinstance(ub, float)

        self.v = None

    @property
    def class_name(self):
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

    @property
    def class_name(self):
        return self.__class__.__name__


class Objective:
    """
    Base class for objective functions.
    """

    def __init__(self,
                 e_str: Optional[str] = None,
                 sense: Optional[str] = 'min'):
        self.e_str = e_str
        self.sense = sense


class OModel:
    r"""
    Base class for optimization models.
    The optimziation problem is formulated as:

    .. math::
        \min_x \ & x^_{t} c_{2}^T x + c_{1} x \\
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
    |    c2     | quadratic objective coefficients            |
    +-----------+---------------------------------------------+
    |    c1     | linear objective coefficients               |
    +-----------+---------------------------------------------+
    |    Aub    | inequality coefficients                     |
    +-----------+---------------------------------------------+
    |    Aeq    | equality coefficients                       |
    +-----------+---------------------------------------------+
    |    bub    | inequality upper bounds                     |
    +-----------+---------------------------------------------+
    |    beq    | equality bounds                             |
    +-----------+---------------------------------------------+
    |    lb     | decision variable lower bounds              |
    +-----------+---------------------------------------------+
    |    ub     | decision variable upper bounds              |
    +-----------+---------------------------------------------+
    """

    def __init__(self, routine):
        self.routine = routine
        # --- colloect optimziation model ---
        self.mdl = None
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = None
        self.n = 0  # number of decision variables

        # self._array_and_counter = {
        #     'c': 'n',  # decision variables
        #     'Aub': 'n',  # inequality LHS
        #     'Aeq': 'm',  # equality LHS
        #     'bub': 'm',  # inequality RHS
        #     'beq': 'o',  # equality RHS
        #     'lb': 'n',  # decision variables lower bounds
        #     'ub': 'n',  # decision variables upper bounds
        # }

        self.c = np.array([])
        self.Aub, self.Aeq = np.array([]), np.array([])
        self.bub, self.beq = np.array([]), np.array([])
        self.lb, self.ub = np.array([]), np.array([])

    @timer
    def setup(self):
        """
        Setup the numerical optimziation formulation from symbolic disaptch model.

        Decision variables are the ``RAlgeb`` of a routine.
        For example, the power outputs ``pg`` of routine ``DCOPF``.

        """
        # --- add decision variables ---
        for rname, ralgeb in self.routine.ralgebs.items():
            var = cp.Variable(ralgeb.n, boolean=(ralgeb.unit == 'bool'))
            self.n += ralgeb.n
            setattr(self, rname, var)
            self.vars[rname] = var
            if ralgeb.lb:
                bv = ralgeb.lb.owner.get(src=ralgeb.lb.name, idx=ralgeb.get_idx(), attr='v')
                constr = var >= bv
                setattr(self, ralgeb.lb.name, constr)
                self.constrs[ralgeb.lb.name] = constr
            if ralgeb.ub:
                bv = ralgeb.ub.owner.get(src=ralgeb.ub.name, idx=ralgeb.get_idx(), attr='v')
                constr = var <= bv
                setattr(self, ralgeb.ub.name, constr)
                self.constrs[ralgeb.ub.name] = constr
        # --- add constraints ---
        for cname, constr in self.routine.constrs.items():
                pass# TODO: might need to parse the constraint into CVXPY format
                # self.constrs[cname] =  >= ralgeb.lb.v

        # --- parse objective functions ---
        # TODO: might need to parse the objective into CVXPY format
        # self.obj = cp.Minimize(self.routine.obj.e)
        # logger.debug(f'Eval obj: {eval(self.routine.obj.e_str)}')

        # --- finalize the optimziation formulation ---
        # mdl = cp.Problem(self.obj, constraints)
        return True

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__
