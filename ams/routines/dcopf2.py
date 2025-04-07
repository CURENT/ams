"""
DCOPF routines.
"""
import logging

import numpy as np
from ams.core.param import RParam
from ams.core.service import NumOp, NumOpDual

from ams.routines.dcopf import DCOPF

from ams.opt import Constraint, Objective, ExpressionCalc, Expression


logger = logging.getLogger(__name__)


class DCOPF2(DCOPF):
    """
    DC optimal power flow (DCOPF) using PTDF.

    The nodal price is calculated as ``pi`` in ``pic``.
    """

    def __init__(self, system, config):
        DCOPF.__init__(self, system, config)
        self.info = 'DCOPF using PTDF'
        self.type = 'DCED'

        # --- rewrite power balance ---
        # TODO: online status?
        # TODO: check if we consider load and shunt online status in DCOPF
        pb = 'sum(pd) + sum(gsh) - sum(pg)'