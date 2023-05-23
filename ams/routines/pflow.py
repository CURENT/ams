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
from ams.core.var import RAlgeb

from ams.routines.dcpf import DCPFlowData, DCPFlowBase
from ams.opt.omodel import Constraint, Objective

logger = logging.getLogger(__name__)


class PFlowData(DCPFlowData):
    """
    AC Power Flow routine.
    """

    def __init__(self):
        DCPFlowData.__init__(self)

    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        res, success = runpf(ppc, **kwargs)
        return ppc, success


class PFlowBase(DCPFlowBase):
    """
    Base class for AC Power Flow model.
    """

    def __init__(self, system, config):
        DCPFlowBase.__init__(self, system, config)
        self.info = 'AC Power Flow'

        # --- bus ---
        self.aBus = RAlgeb(info='bus voltage angle',
                           unit='rad',
                           name='aBus',
                           src='a',
                           tex_name=r'a_{Bus}',
                           owner_name='Bus',
                           )
        self.vBus = RAlgeb(info='bus voltage magnitude',
                           unit='p.u.',
                           name='vBus',
                           src='v',
                           tex_name=r'v_{Bus}',
                           owner_name='Bus',
                           )
        # --- gen ---
        self.pg = RAlgeb(info='active power generation',
                         unit='p.u.',
                         name='pg',
                         src='p',
                         tex_name=r'p_{g}',
                         owner_name='StaticGen',
                         )
        self.qg = RAlgeb(info='reactive power generation',
                         unit='p.u.',
                         name='qg',
                         src='q',
                         tex_name=r'q_{g}',
                         owner_name='StaticGen',
                         )
        # --- constraints ---
        self.pb = Constraint(name='pb',
                             info='power balance',
                             e_str='sum(pd) - sum(pg)',
                             type='eq',
                             )
        # TODO: AC power flow formulation


class PFlow(PFlowData, PFlowBase):
    """
    AC Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        PFlowData.__init__(self)
        PFlowBase.__init__(self, system, config)
