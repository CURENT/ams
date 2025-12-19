"""
RTED routines.
"""
import logging

import numpy as np
from ams.core.param import RParam
from ams.core.service import NumOp

from ams.routines.dcopf2 import DCOPF2
from ams.routines.rted import RTED
from ams.opt import ExpressionCalc

from ams.shared import sps


logger = logging.getLogger(__name__)


class RTED2(RTED, DCOPF2):
    """
    DC-based real-time economic dispatch (RTED) using PTDF.

    For large cases, it is recommended to build the PTDF first, especially when incremental
    build is necessary.

    RTED2 extends DCOPF2 with:

    - Vars for SFR reserve: ``pru`` and ``prd``
    - Param for linear SFR cost: ``cru`` and ``crd``
    - Param for SFR requirement: ``du`` and ``dd``
    - Param for ramping: start point ``pg0`` and ramping limit ``R10``
    - Param ``pg0``, which can be retrieved from dynamic simulation results.

    The function ``dc2ac`` sets the ``vBus`` value from solved ACOPF.
    Without this conversion, dynamic simulation might fail due to the gap between
    DC-based dispatch results and AC-based dynamic initialization.

    Notes
    -----
    - Formulations has been adjusted with interval ``config.t``, 5/60 [Hour] by default.
    - The tie-line flow related constraints are ommited in this formulation.
    - Power generation is balanced for the entire system.
    - SFR is balanced for each area.
    """

    def __init__(self, system, config):
        RTED.__init__(self, system, config)

        self.info = 'Real-time economic dispatch using PTDF'
        self.type = 'DCED'

        # NOTE: in this way, we still follow the implementation that devices
        # connectivity status is considered in connection matrix
        self.PTDF = RParam(info='PTDF',
                           name='PTDF', tex_name=r'P_{TDF}',
                           model='mats', src='PTDF',
                           no_parse=True, sparse=True)
        self.PTDFt = NumOp(u=self.PTDF,
                           name='PTDFt', tex_name=r'P_{TDF}^T',
                           info='PTDF transpose',
                           fun=np.transpose, no_parse=True)

        # --- rewrite Constraint pb: power balance ---
        self.pb.e_str = 'sum(pg) - sum(pd)'

        # --- rewrite Expression plf: line flow---
        self.plf.e_str = 'PTDF @ (Cg@pg - Cl@pd - Csh@gsh - Pbusinj)'

        # --- rewrite nodal price ---
        self.pie = ExpressionCalc(info='Energy price',
                                  name='pie', unit='$/p.u.',
                                  e_str='-pb.dual_variables[0]')
        pic = '-PTDFt@(plfub.dual_variables[0] - plflb.dual_variables[0])'
        self.pic = ExpressionCalc(info='Congestion price',
                                  name='pic', unit='$/p.u.',
                                  e_str=pic,
                                  model='Bus', src=None)

        # NOTE: another implementation of self.pi.e_str can be:
        # self.pi.e_str = self.pie.e_str + self.pic.e_str
        # but it is less intuitive to read, as this implementation is not likely
        # to be used again in other routines.
        pi = '-pb.dual_variables[0] - PTDFt@(plfub.dual_variables[0] - plflb.dual_variables[0])'
        self.pi.e_str = pi
        self.pi.info = 'locational marginal price (LMP)'

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
