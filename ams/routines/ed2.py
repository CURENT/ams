"""
Economic dispatch routines using PTDF formulation.
"""
import logging

from ams.routines.dcopf2 import PTDFBase
from ams.routines.ed import ED, ESD1MPBase, DGMPBase

from ams.shared import sps

logger = logging.getLogger(__name__)


class PTDFMPBase(PTDFBase):
    """
    Extend :ref:`PTDFBase` for multi-period scheduling.
    """
    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        # rewrite pb to ensure power balance for each period
        self.pb.e_str = "sum(pg, axis=0) - sum(pds, axis=0)"

        # --- rewrite Expression plf: line flow---
        self.plf.e_str = "PTDF @ (Cg@pg - Cl@pds - Csh@gsh@tlv - Pbusinj@tlv)"

    def _post_solve(self):
        if self.aBus.horizon is not None:
            # Calculate bus angles after solving
            sys = self.system
            Pbus = sys.mats.Cg._v @ self.pg.v
            Pbus -= sys.mats.Cl._v @ self.pds.v
            Pbus -= sys.mats.Csh._v @ self.gsh.v @ self.tlv.v
            Pbus -= self.Pbusinj.v @ self.tlv.v
            aBus = sps.linalg.spsolve(sys.mats.Bbus._v, Pbus)
            slack0_uid = sys.Bus.idx2uid(sys.Slack.bus.v[0])
            self.aBus.v = aBus - aBus[slack0_uid]
        return super()._post_solve()


class ED2(PTDFMPBase, ED):
    """
    DC-based multi-period economic dispatch (ED) using PTDF.
    Dispatch interval ``config.t`` ($T_{cfg}$) is introduced, 1 [Hour] by default.
    ED extends DCOPF as follows:

    - Vars ``pg``, ``pru``, ``prd`` are extended to 2D
    - 2D Vars ``rgu`` and ``rgd`` are introduced
    - Param ``ug`` is sourced from ``EDTSlot.ug`` as generator commitment

    Notes
    -----
    - Formulations have been adjusted with interval ``config.t``
    - The tie-line flow is not implemented in this model.
    - ``EDTSlot.ug`` is used instead of ``StaticGen.u`` for generator commitment.
    - Following reserves are balanced for each "Area": RegUp reserve ``rbu``,
      RegDn reserve ``rbd``, and Spinning reserve ``rsr``.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)


class ED2DG(DGMPBase, ED2):
    """
    ED with distributed generation :ref:`DG` using PTDF.

    Note that ED2DG only includes DG output power. If ESD1 is included,
    ED2ES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)


class ED2ES(ESD1MPBase, ED2):
    """
    ED with energy storage :ref:`ESD1` using PTDF.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)
