"""
Power flow routines.
"""
import logging
from collections import OrderedDict

import numpy as np

from andes.shared import deg2rad

from ams.routines.routine import RoutineData, Routine
from ams.opt.omodel import Constraint, Objective
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
        # --- line ---
        self.x = RParam(info="line reactance",
                        name='x',
                        tex_name='x',
                        src='x',
                        unit='p.u.',
                        owner_name='Line',
                        )
        self.tap = RParam(info="transformer branch tap ratio",
                          name='tap',
                          src='tap',
                          tex_name='t_{ap}',
                          unit='float',
                          owner_name='Line',
                          )
        self.phi = RParam(info="transformer branch phase shift in rad",
                          name='phi',
                          src='phi',
                          tex_name=r'\phi',
                          unit='radian',
                          owner_name='Line',
                          )

        # --- load ---
        self.pd = RParam(info='active power load in system base',
                         name='pd',
                         src='p0',
                         tex_name=r'p_{d}',
                         unit='p.u.',
                         owner_name='PQ',
                         )
        self.qd = RParam(info='active power load in system base',
                         name='qd',
                         src='q0',
                         tex_name=r'q_{d}',
                         unit='p.u.',
                         owner_name='PQ',
                         )


class DCPFlowBase(Routine):
    """
    Base class for Power Flow model.

    Overload the ``solve``, ``unpack``, and ``run`` methods.
    """

    def __init__(self, system, config):
        Routine.__init__(self, system, config)
        self.info = 'DC Power Flow'

        # --- bus ---
        self.aBus = RAlgeb(info='bus voltage angle',
                           unit='rad',
                           name='aBus',
                           tex_name=r'a_{Bus}',
                           owner_name='Bus',
                           )
        # --- gen ---
        self.pg = RAlgeb(info='actual active power generation',
                         unit='p.u.',
                         name='pg',
                         tex_name=r'p_{g}',
                         owner_name='StaticGen',
                         )
        # --- constraints ---
        self.pb = Constraint(name='pb',
                             info='power balance',
                             e_str='sum(pd) - sum(pg)',
                             type='eq',
                             )

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


class DCPF(DCPFlowData, DCPFlowBase):
    """
    DC Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        DCPFlowData.__init__(self)
        DCPFlowBase.__init__(self, system, config)
