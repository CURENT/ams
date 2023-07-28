"""
Power flow routines.
"""
import logging
from collections import OrderedDict

import numpy as np

from andes.shared import deg2rad

from ams.solver.pypower.runpf import runpf, rundcpf

from ams.io.pypower import system2ppc
from ams.core.param import RParam
from ams.solver.pypower.runpf import runpf

from ams.routines.dcpf import DCPFlowData, DCPFlowBase
from ams.opt.omodel import Var, Constraint, Objective

logger = logging.getLogger(__name__)


class PFlowData(DCPFlowData):
    """
    AC Power Flow routine.
    """

    def __init__(self):
        DCPFlowData.__init__(self)
        self.qd = RParam(info='reactive power load in system base',
                         name='qd',
                         src='q0',
                         tex_name=r'q_{d}',
                         unit='p.u.',
                         model='PQ',
                         )


class PFlowModel(DCPFlowBase):
    """
    AC Power Flow model.
    """

    def __init__(self, system, config):
        DCPFlowBase.__init__(self, system, config)
        self.info = 'AC Power Flow'
        self.type = 'PF'

        # --- bus ---
        self.aBus = Var(info='bus voltage angle',
                        unit='rad',
                        name='aBus',
                        src='a',
                        tex_name=r'a_{Bus}',
                        model='Bus',
                        )
        self.vBus = Var(info='bus voltage magnitude',
                        unit='p.u.',
                        name='vBus',
                        src='v',
                        tex_name=r'v_{Bus}',
                        model='Bus',
                        )
        # --- gen ---
        self.pg = Var(info='active power generation',
                      unit='p.u.',
                      name='pg',
                      src='p',
                      tex_name=r'p_{g}',
                      model='StaticGen',
                      )
        self.qg = Var(info='reactive power generation',
                      unit='p.u.',
                      name='qg',
                      src='q',
                      tex_name=r'q_{g}',
                      model='StaticGen',
                      )
        # --- constraints ---
        self.pb = Constraint(name='pb',
                             info='power balance',
                             e_str='sum(pd) - sum(pg)',
                             type='eq',
                             )
        # TODO: AC power flow formulation

    def solve(self, **kwargs):
        """
        Solve the AC Power Flow with PYPOWER.
        """
        ppc = system2ppc(self.system)
        res, success = runpf(ppc, **kwargs)
        return res, success


class PFlow(PFlowData, PFlowModel):
    """
    AC Power Flow routine.

    Notes
    -----
    1. AC pwoer flow is solved with PYPOWER ``runpf`` function.
    2. AC power flow formulation in AMS style is NOT DONE YET,
       but this does not affect the results
       because the data are passed to PYPOWER for solving.
    """

    def __init__(self, system=None, config=None):
        PFlowData.__init__(self)
        PFlowModel.__init__(self, system, config)
