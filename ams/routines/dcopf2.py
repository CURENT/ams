"""
DCOPF routines using PTDF formulation.
"""

import logging

import numpy as np
from ams.core.param import RParam
from ams.core.service import NumOp

from ams.routines.dcopf import DCOPF
from ams.opt import ExpressionCalc

from ams.shared import sps

logger = logging.getLogger(__name__)


class PTDFMixin:
    """
    Mixin class for PTDF-based formulations.

    This mixin provides PTDF parameters and methods for routines that need
    to use PTDF formulation instead of B-theta formulation.

    The PTDF (Power Transfer Distribution Factor) formulation is more efficient
    for large-scale systems as it eliminates the need to solve for bus angles
    explicitly in the optimization problem.
    """

    def __init__(self, system, config):
        super().__init__(system, config)
        # NOTE: in this way, we still follow the implementation that devices
        # connectivity status is considered in connection matrix
        self.PTDF = RParam(info="PTDF",
                           name="PTDF",
                           tex_name=r"P_{TDF}",
                           model="mats",
                           src="PTDF",
                           no_parse=True,
                           sparse=True,)
        self.PTDFt = NumOp(u=self.PTDF,
                           name="PTDFt",
                           tex_name=r"P_{TDF}^T",
                           info="PTDF transpose",
                           fun=np.transpose,
                           no_parse=True,
                           sparse=True,)

        # --- rewrite Constraint pb: power balance ---
        # PTDF formulation uses system-wide balance instead of nodal balance
        self.pb.e_str = "sum(pg) - sum(pd)"

        # --- rewrite Expression plf: line flow---
        # Use PTDF matrix instead of Bf@aBus
        self.plf.e_str = "PTDF @ (Cg@pg - Cl@pd - Csh@gsh - Pbusinj)"

        # --- rewrite nodal price ---
        # Energy price component (from power balance dual)
        self.pie = ExpressionCalc(info="Energy price",
                                  name="pie",
                                  unit="$/p.u.",
                                  e_str="-pb.dual_variables[0]",)

        # Congestion price component (from line flow constraint duals)
        pic = "-PTDFt@(plfub.dual_variables[0] - plflb.dual_variables[0])"
        self.pic = ExpressionCalc(info="Congestion price",
                                  name="pic",
                                  unit="$/p.u.",
                                  e_str=pic,
                                  model="Bus",)

        # Total LMP = energy price + congestion price
        # NOTE: another implementation could be:
        # self.pi.e_str = self.pie.e_str + self.pic.e_str
        # but the current implementation is more explicit
        pi = "-pb.dual_variables[0] - PTDFt@(plfub.dual_variables[0] - plflb.dual_variables[0])"
        self.pi.e_str = pi
        self.pi.info = "locational marginal price (LMP)"

    def _post_solve(self):
        # Calculate bus angles after solving
        sys = self.system
        Pbus = sys.mats.Cg._v @ self.pg.v
        Pbus -= sys.mats.Cl._v @ self.pd.v
        Pbus -= sys.mats.Csh._v @ self.gsh.v
        Pbus -= self.Pbusinj.v
        aBus = sps.linalg.spsolve(sys.mats.Bbus._v, Pbus)
        slack0_uid = sys.Bus.idx2uid(sys.Slack.bus.v[0])
        self.aBus.v = aBus - aBus[slack0_uid]
        return True


class DCOPF2(PTDFMixin, DCOPF):
    """
    DC optimal power flow (DCOPF) using PTDF formulation.

    For large cases, it is recommended to build the PTDF first, especially when incremental
    build is necessary.

    Notes
    -----
    - This routine requires PTDF matrix.
    - LMP ``pi`` is calculated with two parts, energy price and congestion price.
    - Bus angle ``aBus`` is calculated after solving the problem.
    - In export results, ``pi`` and ``pic`` are kept for each bus, while ``pie``
      can be restored manually by ``pie = pi - pic`` if needed.

    Warning
    -------
    In this implementation, the dual variables for constraints have opposite signs compared
    to the mathematical formulation: 1. The dual of `pb` returns a negative value, so energy
    price is computed as `-pb.dual_variables[0]`. 2. Similarly, a minus sign is applied to
    the duals of `plfub` and `plflb` when calculating congestion price. The reason for this
    sign difference is not yet fully understood.

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
        super().__init__(system, config)
