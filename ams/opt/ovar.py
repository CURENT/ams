"""
Module for optimization variables.
"""

import logging

from typing import Optional

from ams.core.var import RAlgeb
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
                 ):
        self.type = type
        self.n = n
        # TODO: add sanity check for lb and ub
        type_ndarray = isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray)
        type_float = isinstance(lb, float) and isinstance(ub, float)
        self.lb = lb
        self.ub = ub
        self.v = np.empty(n)
