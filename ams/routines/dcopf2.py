"""
DCOPF routines.
"""
import logging

import numpy as np
from ams.core.param import RParam
from ams.core.service import NumOp

from ams.routines.dcopf import DCOPF
from ams.opt import ExpressionCalc

from ams.shared import sps


logger = logging.getLogger(__name__)


class DCOPF2(DCOPF):
    """
    DC optimal power flow (DCOPF) using PTDF formulation.

    For large cases, it is recommended to build the PTDF first, especially when incremental
    build is necessary.

    Notes
    -----
    - This routine requires PTDF matrix.
    - Nodal price ``pi`` is calculated with three parts.
    - Bus angle ``aBus`` is calculated after solving the problem.

    References
    ----------
    1. R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, “MATPOWER: Steady-State
       Operations, Planning, and Analysis Tools for Power Systems Research and Education,” IEEE
       Trans. Power Syst., vol. 26, no. 1, pp. 12-19, Feb. 2011
    2. Y. Chen et al., "Security-Constrained Unit Commitment for Electricity Market: Modeling,
       Solution Methods, and Future Challenges," in IEEE Transactions on Power Systems, vol. 38, no. 5,
       pp. 4668-4681, Sept. 2023
    """

    def __init__(self, system, config):
        DCOPF.__init__(self, system, config)
        self.info = 'DCOPF using PTDF'
        self.type = 'DCED'

        # NOTE: in this way, we still follow the implementation that devices
        # connectivity status is considered in connection matrix
        self.ued = NumOp(u=self.Cl,
                         name='ued', tex_name=r'u_{e,d}',
                         info='Effective load connection status',
                         fun=np.sum, args=dict(axis=0),
                         no_parse=True)
        self.uesh = NumOp(u=self.Csh,
                          name='uesh', tex_name=r'u_{e,sh}',
                          info='Effective shunt connection status',
                          fun=np.sum, args=dict(axis=0),
                          no_parse=True)

        self.PTDF = RParam(info='PTDF',
                           name='PTDF', tex_name=r'P_{TDF}',
                           model='mats', src='PTDF',
                           no_parse=True, sparse=True)

        # --- rewrite Expression plf: line flow---
        self.plf.e_str = 'PTDF @ (Cg@pg - Cl@pd - Csh@gsh - Pbusinj)'

        # --- rewrite nodal price ---
        self.Cft = RParam(info='Line connection matrix',
                          name='Cft', tex_name=r'C_{ft}',
                          model='mats', src='Cft',
                          no_parse=True, sparse=True,)
        self.pilb = ExpressionCalc(info='Congestion price, dual of <plflb>',
                                   name='pilb',
                                   model='Line', src=None,
                                   e_str='plflb.dual_variables[0]')
        self.piub = ExpressionCalc(info='Congestion price, dual of <plfub>',
                                   name='piub',
                                   model='Line', src=None,
                                   e_str='plfub.dual_variables[0]')
        self.pib = ExpressionCalc(info='Energy price, dual of <pb>',
                                  name='pib',
                                  model='Bus', src=None,
                                  e_str='pb.dual_variables[0]')
        pi = 'pb.dual_variables[0] + Cft@(plfub.dual_variables[0] - plflb.dual_variables[0])'
        self.pi.e_str = pi

    def _post_solve(self):
        """Calculate aBus"""
        super()._post_solve()
        sys = self.system
        Pbus = sys.mats.Cg._v @ self.pg.v
        Pbus -= sys.mats.Cl._v @ self.pd.v
        Pbus -= sys.mats.Csh._v @ self.gsh.v
        Pbus -= self.Pbusinj.v
        aBus = sps.linalg.spsolve(sys.mats.Bbus._v, Pbus)
        slack0_uid = sys.Bus.idx2uid(sys.Slack.bus.v[0])
        self.aBus.v = aBus - aBus[slack0_uid]
        return super()._post_solve()

    def init(self, **kwargs):
        if self.system.mats.PTDF._v is None:
            logger.warning('PTDF is not available, build it now')
            self.system.mats.build_ptdf()
        return super().init(**kwargs)
