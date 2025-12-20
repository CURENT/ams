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

    def _setup_ptdf_params(self):
        """
        Setup PTDF parameters.

        Creates the PTDF matrix parameter and its transpose for use in
        power flow and LMP calculations.
        """
        # NOTE: in this way, we still follow the implementation that devices
        # connectivity status is considered in connection matrix
        self.PTDF = RParam(
            info="PTDF",
            name="PTDF",
            tex_name=r"P_{TDF}",
            model="mats",
            src="PTDF",
            no_parse=True,
            sparse=True,
        )
        self.PTDFt = NumOp(
            u=self.PTDF,
            name="PTDFt",
            tex_name=r"P_{TDF}^T",
            info="PTDF transpose",
            fun=np.transpose,
            no_parse=True,
            sparse=True,
        )

    def _setup_ptdf_expressions(self):
        """
        Setup PTDF-based expressions.

        Overwrites the default B-theta formulation with PTDF-based formulation for:
        - Power balance constraint (system-wide instead of nodal)
        - Line flow expression (using PTDF instead of Bf@aBus)
        - Nodal price calculation (energy price + congestion price)
        """
        # --- rewrite Constraint pb: power balance ---
        # PTDF formulation uses system-wide balance instead of nodal balance
        self.pb.e_str = "sum(pg) - sum(pd)"

        # --- rewrite Expression plf: line flow---
        # Use PTDF matrix instead of Bf@aBus
        self.plf.e_str = "PTDF @ (Cg@pg - Cl@pd - Csh@gsh - Pbusinj)"

        # --- rewrite nodal price ---
        # Energy price component (from power balance dual)
        self.pie = ExpressionCalc(
            info="Energy price",
            name="pie",
            unit="$/p.u.",
            e_str="-pb.dual_variables[0]",
        )

        # Congestion price component (from line flow constraint duals)
        pic = "-PTDFt@(plfub.dual_variables[0] - plflb.dual_variables[0])"
        self.pic = ExpressionCalc(
            info="Congestion price",
            name="pic",
            unit="$/p.u.",
            e_str=pic,
            model="Bus",
            src=None,
        )

        # Total LMP = energy price + congestion price
        # NOTE: another implementation could be:
        # self.pi.e_str = self.pie.e_str + self.pic.e_str
        # but the current implementation is more explicit
        pi = "-pb.dual_variables[0] - PTDFt@(plfub.dual_variables[0] - plflb.dual_variables[0])"
        self.pi.e_str = pi
        self.pi.info = "locational marginal price (LMP)"

    def _ptdf_post_solve(self):
        """
        Calculate bus angles after solving using PTDF formulation.

        Since PTDF formulation doesn't include bus angles in the optimization,
        we need to back-calculate them from the power injections.
        """
        sys = self.system
        # Calculate net power injection at each bus
        Pbus = sys.mats.Cg._v @ self.pg.v
        Pbus -= sys.mats.Cl._v @ self.pd.v
        Pbus -= sys.mats.Csh._v @ self.gsh.v
        Pbus -= self.Pbusinj.v

        aBus = sps.linalg.spsolve(sys.mats.Bbus._v, Pbus)

        slack0_uid = sys.Bus.idx2uid(sys.Slack.bus.v[0])
        self.aBus.v = aBus - aBus[slack0_uid]

    def _check_ptdf(self):
        """
        Check if PTDF matrix is available and build if necessary.

        The PTDF matrix should be pre-built for large systems to avoid
        computational overhead during routine initialization.
        """
        if self.system.mats.PTDF._v is None:
            logger.warning("PTDF is not available, build it now")
            self.system.mats.build_ptdf()


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
        DCOPF.__init__(self, system, config)
        self.type = "DCED"

        self._setup_ptdf_params()
        self._setup_ptdf_expressions()

    def _post_solve(self):
        """Calculate aBus and perform other post-solve operations."""
        super()._post_solve()
        self._ptdf_post_solve()
        return True

    def init(self, **kwargs):
        """Initialize the routine, ensuring PTDF is available."""
        self._check_ptdf()
        return super().init(**kwargs)
