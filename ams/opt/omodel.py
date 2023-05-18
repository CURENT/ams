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
        self.routine = routine

        self.c = np.array([])
        self.Aub, self.Aeq = np.array([]), np.array([])
        self.bub, self.beq = np.array([]), np.array([])
        self.lb, self.ub = np.array([]), np.array([])

    def setup(self):
        """
        Setup the numerical optimziation formulation from symbolic disaptch model.
        """
        pass
        # loop self.vars
        # loop self.constrs
        # build self.objective
