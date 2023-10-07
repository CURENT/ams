"""
Power flow routines.
"""
import logging  # NOQA

from ams.pypower.routines.pflow.solve import runpf  # NOQA

from ams.io.pypower import system2ppc  # NOQA
from ams.core.param import RParam  # NOQA

from ams.routines.dcpf import DCPFlowData, DCPFlowBase  # NOQA
from ams.opt.omodel import Var, Constraint  # NOQA

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
                             e_str='sum(pl) - sum(pg)',
                             type='eq',
                             )
        # TODO: AC power flow formulation

    def solve(self, **kwargs):
        """
        Solve the AC Power Flow with PYPOWER.
        """
        ppc = system2ppc(self.system)
        res, success, info = runpf(ppc, **kwargs)
        return res, success, info


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
