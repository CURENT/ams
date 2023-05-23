"""
Power flow routines.
"""
import logging
from collections import OrderedDict

import numpy as np

from andes.shared import deg2rad
from andes.utils.misc import elapsed

from ams.solver.pypower.runopf import runopf

from ams.io.pypower import system2ppc
from ams.core.param import RParam
from ams.core.var import RAlgeb

from ams.routines.pflow import PFlowData, PFlowBase
from ams.opt.omodel import Constraint, Objective

logger = logging.getLogger(__name__)


class ACOPFData(PFlowData):
    """
    AC Power Flow routine.
    """

    def __init__(self):
        PFlowData.__init__(self)

    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        res = runopf(ppc, **kwargs)
        return ppc

    def run(self, **kwargs):
        """
        Routine the routine.
        """
        if not self.is_setup:
            logger.info(f"Setup model for {self.class_name}")
            self.setup()
        t0, _ = elapsed()
        res = self.solve(**kwargs)
        # TODO: check exit_code
        _, s = elapsed(t0)
        self.exec_time = float(s.split(' ')[0])
        self.unpack(res)
        info = f"{self.class_name} completed in {s} with exit code {self.exit_code}."
        logger.info(info)
        return self.exit_code


class ACOPFBase(PFlowBase):
    """
    Base class for AC Power Flow model.
    """

    def __init__(self, system, config):
        PFlowBase.__init__(self, system, config)
        self.info = 'AC Optimal Power Flow'

        # --- constraints ---
        self.pb = Constraint(name='pb',
                             info='power balance',
                             e_str='sum(pd) - sum(pg)',
                             type='eq',
                             )
        # TODO: ACOPF formulation


class ACOPF(ACOPFData, ACOPFBase):
    """
    AC Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        ACOPFData.__init__(self)
        ACOPFBase.__init__(self, system, config)
