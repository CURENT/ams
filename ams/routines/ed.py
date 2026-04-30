"""
Economic dispatch routines.
"""
import logging
import numpy as np

import cvxpy as cp

from ams.core.param import RParam
from ams.core.service import (NumOpDual, NumHstack,
                              RampSub, NumOp, LoadScale)

from ams.routines.rted import RTED, DGBase, ESD1Base

from ams.opt import Var, Constraint

logger = logging.getLogger(__name__)


# --- ED e_fn callables (Phase 4.4) ---


def _ed_pmaxe(r):
    return (cp.multiply(cp.multiply(r.nctrle, r.pg0), r.tlv)
            + cp.multiply(cp.multiply(r.ctrle, r.tlv), r.pmax))


def _ed_pmine(r):
    return (cp.multiply(cp.multiply(r.nctrle, r.pg0), r.tlv)
            + cp.multiply(cp.multiply(r.ctrle, r.tlv), r.pmin))


def _ed_pglb(r):
    return -r.pg + r.pmine <= 0


def _ed_pgub(r):
    return r.pg - r.pmaxe <= 0


def _ed_prsb(r):
    return cp.multiply(r.ugt, r.pmax @ r.tlv - r.pg) - r.prs == 0


def _ed_rsr(r):
    return -r.gs @ r.prs + r.dsr <= 0


def _ed_plflb(r):
    return -r.Bf @ r.aBus - r.Pfinj @ r.tlv - cp.multiply(r.ul, r.rate_a) @ r.tlv <= 0


def _ed_plfub(r):
    return r.Bf @ r.aBus + r.Pfinj @ r.tlv - cp.multiply(r.ul, r.rate_a) @ r.tlv <= 0


def _ed_alflb(r):
    return -r.CftT @ r.aBus + r.amin @ r.tlv <= 0


def _ed_alfub(r):
    return r.CftT @ r.aBus - r.amax @ r.tlv <= 0


def _ed_plf(r):
    return r.Bf @ r.aBus + r.Pfinj @ r.tlv


def _ed_pb(r):
    return r.Bbus @ r.aBus + r.Pbusinj @ r.tlv + r.Cl @ r.pds + r.Csh @ r.gsh @ r.tlv - r.Cg @ r.pg == 0


def _ed_rbu(r):
    return r.gs @ cp.multiply(r.ugt, r.pru) - cp.multiply(r.dud, r.tlv) == 0


def _ed_rbd(r):
    return r.gs @ cp.multiply(r.ugt, r.prd) - cp.multiply(r.ddd, r.tlv) == 0


def _ed_rru(r):
    return r.pg + r.pru - cp.multiply(cp.multiply(r.ugt, r.pmax), r.tlv) <= 0


def _ed_rrd(r):
    return -r.pg + r.prd + cp.multiply(cp.multiply(r.ugt, r.pmin), r.tlv) <= 0


def _ed_rgu(r):
    return r.pg @ r.Mr - r.t * r.RR30 <= 0


def _ed_rgd(r):
    return -r.pg @ r.Mr - r.t * r.RR30 <= 0


def _ed_rgu0(r):
    return cp.multiply(r.ugt[:, 0], r.pg[:, 0] - r.pg0[:, 0] - r.R30) <= 0


def _ed_rgd0(r):
    return cp.multiply(r.ugt[:, 0], -r.pg[:, 0] + r.pg0[:, 0] - r.R30) <= 0


def _ed_obj(r):
    return (cp.sum(r.t ** 2 * r.c2 @ r.pg ** 2)
            + r.t * cp.sum(r.c1 @ r.pg + r.csr @ r.prs)
            + cp.sum(cp.multiply(r.ugt, cp.multiply(r.c0, r.tlv))))


class SRBase:
    """
    Base class for spinning reserve.
    """

    def __init__(self, system, config, **kwargs) -> None:
        super().__init__(system, config, **kwargs)
        self.dsrp = RParam(info='spinning reserve requirement in percentage',
                           name='dsr', tex_name=r'd_{sr}',
                           model='SR', src='demand',
                           unit='%',)
        self.csr = RParam(info='cost for spinning reserve',
                          name='csr', tex_name=r'c_{sr}',
                          model='SRCost', src='csr',
                          unit=r'$/(p.u.*h)',
                          indexer='gen', imodel='StaticGen',)

        self.prs = Var(name='prs', tex_name=r'p_{r,s}',
                       info='spinning reserve', unit='p.u.',
                       model='StaticGen', nonneg=True,)

        self.dsrpz = NumOpDual(u=self.pdz, u2=self.dsrp, fun=np.multiply,
                               name='dsrpz', tex_name=r'd_{s,r, p, z}',
                               info='zonal spinning reserve requirement in percentage',)
        self.dsr = NumOpDual(u=self.dsrpz, u2=self.sd, fun=np.multiply,
                             rfun=np.transpose,
                             name='dsr', tex_name=r'd_{s,r,z}',
                             info='zonal spinning reserve requirement',)

        # NOTE: define e_str in the scheduling model
        self.prsb = Constraint(info='spinning reserve balance',
                               name='prsb', is_eq=True,)
        self.rsr = Constraint(info='spinning reserve requirement',
                              name='rsr', is_eq=False,)


class MPBase:
    """
    Base class for multi-period scheduling.
    """

    def __init__(self, system, config, **kwargs) -> None:
        super().__init__(system, config, **kwargs)
        # NOTE: Setting `ED.scale.owner` to `Horizon` will cause an error when calling `ED.scale.v`.
        # This is because `Horizon` is a group that only contains the model `TimeSlot`.
        # The `get` method of `Horizon` calls `andes.models.group.GroupBase.get` and results in an error.
        self.sd = RParam(info='zonal load factor for ED',
                         name='sd', tex_name=r's_{d}',
                         src='sd', model='EDTSlot')

        # NOTE: update timeslot.model in dispatch model if necessary
        self.timeslot = RParam(info='Time slot for multi-period ED',
                               name='timeslot', tex_name=r't_{s,idx}',
                               src='idx', model='EDTSlot',
                               no_parse=True)

        self.tlv = NumOp(u=self.timeslot, fun=np.ones_like,
                         args=dict(dtype=float),
                         expand_dims=0,
                         name='tlv', tex_name=r'1_{tl}',
                         info='time length vector',
                         no_parse=True)

        self.pds = LoadScale(u=self.pd0, sd=self.sd,
                             name='pds', tex_name=r'p_{d,s}',
                             info='Scaled load',)

        self.R30 = RParam(info='30-min ramp rate',
                          name='R30', tex_name=r'R_{30}',
                          src='R30', unit='p.u./h',
                          model='StaticGen', no_parse=True,)
        self.Mr = RampSub(u=self.pg, name='Mr', tex_name=r'M_{r}',
                          info='Subtraction matrix for ramping',
                          no_parse=True, sparse=True,)
        self.RR30 = NumHstack(u=self.R30, ref=self.Mr,
                              name='RR30', tex_name=r'R_{30,R}',
                              info='Repeated ramp rate', no_parse=True,)

        items_to_expand = ['ctrl', 'c0', 'pmax', 'pmin', 'pg0', 'rate_a',
                           'Pfinj', 'Pbusinj', 'gsh', 'ul']
        for item in items_to_expand:
            self.__dict__[item].expand_dims = 1

        # NOTE: extend pg to 2D matrix: row for gen and col for timeslot
        self.pg.horizon = self.timeslot
        self.pg.info = '2D Gen power'
        self.aBus.horizon = self.timeslot
        self.aBus.info = '2D Bus angle'
        self.vBus.horizon = self.timeslot
        self.vBus.info = '2D Bus voltage'
        self.pi.horizon = self.timeslot
        self.mu1.horizon = self.timeslot
        self.mu2.horizon = self.timeslot


class ED(SRBase, MPBase, RTED):
    """
    DC-based multi-period economic dispatch (ED).
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

        self.config.t = 1  # scheduling interval in hour
        self.config.add_extra("_help",
                              t="time interval in hours",
                              )

        self.ug.info = 'unit commitment decisions'
        self.ug.model = 'EDTSlot'
        self.ug.src = 'ug'
        # self.ug.tex_name = r'u_{g}',

        self.dud.expand_dims = 1
        self.ddd.expand_dims = 1
        self.amin.expand_dims = 1
        self.amax.expand_dims = 1

        # --- Data Section ---
        self.ugt = NumOp(u=self.ug, fun=np.transpose,
                         name='ugt', tex_name=r'u_{g}',
                         info='input ug transpose',
                         no_parse=True)

        # --- Model Section (Phase 4.4: e_fn form) ---
        # --- gen ---
        self.ctrle.u2 = self.ugt
        self.nctrle.u2 = self.ugt
        self.pmaxe.e_fn = _ed_pmaxe
        self.pmaxe.horizon = self.timeslot
        self.pmine.e_fn = _ed_pmine
        self.pmine.horizon = self.timeslot
        self.pglb.e_fn = _ed_pglb
        self.pgub.e_fn = _ed_pgub

        self.pru.horizon = self.timeslot
        self.pru.info = '2D RegUp power'
        self.prd.horizon = self.timeslot
        self.prd.info = '2D RegDn power'

        self.prs.horizon = self.timeslot
        self.prsb.e_fn = _ed_prsb
        self.rsr.e_fn = _ed_rsr

        # --- line ---
        self.plf.horizon = self.timeslot
        self.plf.info = '2D Line flow'
        self.plflb.e_fn = _ed_plflb
        self.plfub.e_fn = _ed_plfub
        self.alflb.e_fn = _ed_alflb
        self.alfub.e_fn = _ed_alfub

        self.plf.e_fn = _ed_plf

        # --- power balance ---
        self.pb.e_fn = _ed_pb

        # --- ramping ---
        self.rbu.e_fn = _ed_rbu
        self.rbd.e_fn = _ed_rbd

        self.rru.e_fn = _ed_rru
        self.rrd.e_fn = _ed_rrd

        self.rgu.e_fn = _ed_rgu
        self.rgd.e_fn = _ed_rgd

        self.rgu0 = Constraint(name='rgu0',
                               info='Initial gen ramping up',
                               e_fn=_ed_rgu0,)
        self.rgd0 = Constraint(name='rgd0',
                               info='Initial gen ramping down',
                               e_fn=_ed_rgd0,)

        # --- objective ---
        self.obj.e_fn = _ed_obj

    def dc2ac(self, kloss=1.0, **kwargs):
        """
        AC conversion ``dc2ac`` is not implemented yet for
        multi-period scheduling.
        """
        raise NotImplementedError("dc2ac is not implemented for multi-period scheduling")

    def unpack(self, res, **kwargs):
        """
        Multi-period scheduling will not unpack results from
        solver into devices.

        # TODO: unpack first period results, and allow input
        # to specify which period to unpack.
        """
        return None


class DGMPBase(DGBase):
    """
    Extend :ref:`DGBase` for multi-period scheduling.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        # NOTE: extend vars to 2D
        self.pgdg.horizon = self.timeslot


class EDDG(DGMPBase, ED):
    """
    ED with distributed generation :ref:`DG`.

    Note that EDDG only includes DG output power. If ESD1 is included,
    EDES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)


class ESD1MPBase(ESD1Base):
    """
    Extend :ref:`ESD1Base` for multi-period scheduling.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        self.Mre = RampSub(u=self.SOC, name='Mre', tex_name=r'M_{r,ES}',
                           info='Subtraction matrix for SOC',
                           no_parse=True, sparse=True,)
        self.EnR = NumHstack(u=self.En, ref=self.Mre,
                             name='EnR', tex_name=r'E_{n,R}',
                             info='Repeated En as 2D matrix, (ng, ng-1)')
        self.EtaCR = NumHstack(u=self.EtaC, ref=self.Mre,
                               name='EtaCR', tex_name=r'\eta_{c,R}',
                               info='Repeated Etac as 2D matrix, (ng, ng-1)')
        self.REtaDR = NumHstack(u=self.REtaD, ref=self.Mre,
                                name='REtaDR', tex_name=r'R_{\eta_d,R}',
                                info='Repeated REtaD as 2D matrix, (ng, ng-1)')
        SOCb = 'mul(EnR, SOC @ Mre) - t dot mul(EtaCR, pce[:, 1:])'
        SOCb += ' + t dot mul(REtaDR, pde[:, 1:])'
        self.SOCb.e_str = SOCb

        SOCb0 = 'mul(En, SOC[:, 0] - SOCinit) - t dot mul(EtaC, pce[:, 0])'
        SOCb0 += ' + t dot mul(REtaD, pde[:, 0])'
        self.SOCb0 = Constraint(name='SOCb0', is_eq=True,
                                info='ESD1 SOC initial balance',
                                e_str=SOCb0,)

        self.SOCr.e_str = 'SOCend - SOC[:, -1]'

        # NOTE: extend vars to 2D
        self.pgdg.horizon = self.timeslot
        self.SOC.horizon = self.timeslot
        self.pce.horizon = self.timeslot
        self.pde.horizon = self.timeslot
        self.ucd.horizon = self.timeslot
        self.udd.horizon = self.timeslot
        self.zce.horizon = self.timeslot
        self.zde.horizon = self.timeslot


class EDES(ESD1MPBase, ED):
    """
    ED with energy storage :ref:`ESD1`.
    The bilinear term in the formulation is linearized with big-M method.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)
