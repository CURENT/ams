"""
Module for optimization variables.
"""

import logging

from typing import Optional

import numpy as np

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
