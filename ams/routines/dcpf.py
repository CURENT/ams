"""
Power flow routines.
"""
import logging
from collections import OrderedDict

import numpy as np

from andes.shared import deg2rad

from ams.routines.routine import RoutineData, Routine
from ams.solver.pypower.runpf import runpf, rundcpf

from ams.io.pypower import system2ppc
from ams.core.param import RParam
from ams.core.var import RAlgeb

logger = logging.getLogger(__name__)


class DCPFlowData(RoutineData):
    """
    Data class for power flow routines.
    """

    def __init__(self):
        RoutineData.__init__(self)
        # --- load ---
        self.pd = RParam(info='active power load in system base',
                         name='pd',
                         src='p0',
                         tex_name=r'p_{d}',
                         unit='p.u.',
                         owner_name='PQ',
                         )
        # --- load ---
        self.pd = RParam(info='active power load in system base',
                         name='pd',
                         src='p0',
                         tex_name=r'p_{d}',
                         unit='p.u.',
                         owner_name='PQ',
                         )
        # --- load ---
        self.qd = RParam(info='active power load in system base',
                         name='pd',
                         src='p0',
                         tex_name=r'p_{d}',
                         unit='p.u.',
                         owner_name='PQ',
                         )


class DCPFlowBase(Routine):
    """
    Base class for Power Flow model.

    Overload the ``solve``, ``unpack``, ``run``, and ``__repr__`` methods.
    """

    def __init__(self, system, config):
        Routine.__init__(self, system, config)

    def __repr__(self) -> str:
        info = f"Routine {self.class_name}: Is Setup: {self.is_setup}; Exit Code: {self.exit_code}"
        return info


class DCPF(DCPFlowBase):
    """
    DC Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        DCPFlowBase.__init__(self, system, config)

    def run(self, **kwargs):
        """
        Run power flow.
        """
        pass

    def summary(self, **kwargs):
        """
        Print power flow summary.
        """
        pass

    def report(self, **kwargs):
        """
        Print power flow report.
        """
        pass
