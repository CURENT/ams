"""
Economic dispatch routines using PTDF formulation.
"""
import logging

from ams.routines.dcopf2 import PTDFMixin
from ams.routines.rted import DGBase
from ams.routines.ed import ED, ESD1MPBase

from ams.shared import sps

logger = logging.getLogger(__name__)


class PTDFMixinMP(PTDFMixin):
    """
    PTDFMixin for multi-period routines.
    """
    def __init__(self):
        super().__init__()

        # rewrite pb to ensure power balance for each period
        self.pb.e_str = "sum(pg, axis=0) - sum(pds, axis=0)"

        # --- rewrite Expression plf: line flow---
        self.plf.e_str = "PTDF @ (Cg@pg - Cl@pds - Csh@gsh@tlv - Pbusinj@tlv)"

    def _post_solve(self):
        # Calculate bus angles after solving
        sys = self.system
        Pbus = sys.mats.Cg._v @ self.pg.v
        Pbus -= sys.mats.Cl._v @ self.pds.v
        Pbus -= sys.mats.Csh._v @ self.gsh.v @ self.tlv.v
        Pbus -= self.Pbusinj.v @ self.tlv.v
        aBus = sps.linalg.spsolve(sys.mats.Bbus._v, Pbus)
        slack0_uid = sys.Bus.idx2uid(sys.Slack.bus.v[0])
        self.aBus.v = aBus - aBus[slack0_uid]
        return True


class ED2(ED, PTDFMixinMP):
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

    def __init__(self, system, config):
        super().__init__(system, config)
        PTDFMixinMP.__init__(self)

    def _post_solve(self):
        PTDFMixinMP._post_solve(self)
        return super()._post_solve()


class ED2DG(ED2, DGBase):
    """
    ED with distributed generation :ref:`DG` using PTDF.

    Note that EDDG only includes DG output power. If ESD1 is included,
    EDES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config):
        super().__init__(system, config)
        DGBase.__init__(self)

        self.pgdg.horizon = self.timeslot


class ED2ES(ED2, ESD1MPBase):
    """
    ED with energy storage :ref:`ESD1` using PTDF.
    """

    def __init__(self, system, config):
        super().__init__(system, config)
        ESD1MPBase.__init__(self)
