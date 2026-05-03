"""
Economic dispatch routines.
"""
import logging
import numpy as np

from ams.core.param import RParam
from ams.core.service import (NumOpDual, NumHstack,
                              RampSub, NumOp, LoadScale)

from ams.routines.rted import RTED, DGBase, ESD1Base

from ams.opt import Var, Constraint

logger = logging.getLogger(__name__)


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
        self.dsr = NumOpDual(u=self.dsrpz, u2=self.sdT, fun=np.multiply,
                             rfun=np.transpose,
                             name='dsr', tex_name=r'd_{s,r,z}',
                             info='zonal spinning reserve requirement',)

        # NOTE: define e_str in the scheduling model
        self.prsb = Constraint(info='spinning reserve balance',
                               name='prsb',)
        self.rsr = Constraint(info='spinning reserve requirement',
                              name='rsr',)


class MPBase:
    """
    Base class for multi-period scheduling.
    """

    def __init__(self, system, config, **kwargs) -> None:
        super().__init__(system, config, **kwargs)

        # NOTE: update timeslot.model in dispatch model if necessary
        self.timeslot = RParam(info='Time slot for multi-period ED',
                               name='timeslot', tex_name=r't_{s,idx}',
                               src='idx', model='EDTSlot',
                               no_parse=True)

        # 2D area-load-scaling param sourced from EDSlotLoad
        # (one row per (area, slot)). Routines override `model` to
        # `UCSlotLoad` for UC.
        self.sd = RParam(info='area load scaling factor',
                         name='sd', tex_name=r's_{d}',
                         src='sd', model='EDSlotLoad',
                         indexer='area', imodel='Area',
                         horizon=self.timeslot, hindexer='slot')

        # Transposed (nslot, narea) view of sd, used by LoadScale and
        # the SRBase/NSRBase reserve-requirement chain whose downstream
        # NumOps were authored against the pre-v1.3.0 (nslot, narea)
        # shape. Lets those untouched.
        self.sdT = NumOp(u=self.sd, fun=np.transpose,
                         name='sdT', tex_name=r's_{d}^{T}',
                         info='sd transposed to (nslot, narea)',
                         no_parse=True)

        self.tlv = NumOp(u=self.timeslot, fun=np.ones_like,
                         args=dict(dtype=float),
                         expand_dims=0,
                         name='tlv', tex_name=r'1_{tl}',
                         info='time length vector',
                         no_parse=True)

        self.pds = LoadScale(u=self.pd0, sd=self.sdT,
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
    - Param ``ug`` is sourced from ``EDSlotGen.ug`` as generator commitment

    Notes
    -----
    - Formulations have been adjusted with interval ``config.t``
    - The tie-line flow is not implemented in this model.
    - ``EDSlotGen.ug`` is used instead of ``StaticGen.u`` for generator commitment.
    - Following reserves are balanced for each "Area": RegUp reserve ``rbu``,
      RegDn reserve ``rbd``, and Spinning reserve ``rsr``.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        self.config.t = 1  # scheduling interval in hour
        self.config.add_extra("_help",
                              t="time interval in hours",
                              )

        # Re-source ug from EDSlotGen as a 2D (ngen, nslot) param.
        # The inherited self.ug from DCPF was a 1D StaticGen.u; we
        # mutate in place (rather than re-binding) so the routine's
        # rparams registry doesn't see a double-registration. With
        # the new RParam.horizon shape, the previous NumOp(transpose)
        # ugt workaround disappears.
        self.ug.info = 'unit commitment decisions'
        self.ug.model = 'EDSlotGen'
        self.ug.src = 'ug'
        self.ug.indexer = 'gen'
        self.ug.imodel = 'StaticGen'
        self.ug.horizon = self.timeslot
        self.ug.hindexer = 'slot'

        self.dud.expand_dims = 1
        self.ddd.expand_dims = 1
        self.amin.expand_dims = 1
        self.amax.expand_dims = 1

        # --- Model Section ---
        # --- gen ---
        self.ctrle.u2 = self.ug
        self.nctrle.u2 = self.ug
        pmaxe = 'cp.multiply(cp.multiply(nctrle, pg0), tlv) + cp.multiply(cp.multiply(ctrle, tlv), pmax)'
        self.pmaxe.e_str = pmaxe
        self.pmaxe.horizon = self.timeslot
        pmine = 'cp.multiply(cp.multiply(nctrle, pg0), tlv) + cp.multiply(cp.multiply(ctrle, tlv), pmin)'
        self.pmine.e_str = pmine
        self.pmine.horizon = self.timeslot
        self.pglb.e_str = '-pg + pmine <= 0'
        self.pgub.e_str = 'pg - pmaxe <= 0'

        self.pru.horizon = self.timeslot
        self.pru.info = '2D RegUp power'
        self.prd.horizon = self.timeslot
        self.prd.info = '2D RegDn power'

        self.prs.horizon = self.timeslot
        self.prsb.e_str = 'cp.multiply(ug, pmax@tlv - pg) - prs == 0'
        self.rsr.e_str = '-gs@prs + dsr <= 0'

        # --- line ---
        self.plf.horizon = self.timeslot
        self.plf.info = '2D Line flow'
        self.plflb.e_str = '-Bf@aBus - Pfinj@tlv - cp.multiply(ul, rate_a)@tlv <= 0'
        self.plfub.e_str = 'Bf@aBus + Pfinj@tlv - cp.multiply(ul, rate_a)@tlv <= 0'
        self.alflb.e_str = '-CftT@aBus + amin@tlv <= 0'
        self.alfub.e_str = 'CftT@aBus - amax@tlv <= 0'

        self.plf.e_str = 'Bf@aBus + Pfinj@tlv'

        # --- power balance ---
        self.pb.e_str = 'Bbus@aBus + Pbusinj@tlv + Cl@pds + Csh@gsh@tlv - Cg@pg == 0'

        # --- ramping ---
        self.rbu.e_str = 'gs@cp.multiply(ug, pru) - cp.multiply(dud, tlv) == 0'
        self.rbd.e_str = 'gs@cp.multiply(ug, prd) - cp.multiply(ddd, tlv) == 0'

        self.rru.e_str = 'pg + pru - cp.multiply(cp.multiply(ug, pmax), tlv) <= 0'
        self.rrd.e_str = '-pg + prd + cp.multiply(cp.multiply(ug, pmin), tlv) <= 0'

        self.rgu.e_str = 'pg @ Mr - t * RR30 <= 0'
        self.rgd.e_str = '-pg @ Mr - t * RR30 <= 0'

        self.rgu0 = Constraint(name='rgu0',
                               info='Initial gen ramping up',
                               e_str='cp.multiply(ug[:, 0], pg[:, 0] - pg0[:, 0] - R30) <= 0',
                               )
        self.rgd0 = Constraint(name='rgd0',
                               info='Initial gen ramping down',
                               e_str='cp.multiply(ug[:, 0], -pg[:, 0] + pg0[:, 0] - R30) <= 0',
                               )

        # --- objective ---
        cost = 'cp.sum(t**2 * c2 @ pg**2)'
        cost += '+ t * cp.sum(c1 @ pg + csr @ prs)'
        cost += '+ cp.sum(cp.multiply(ug, cp.multiply(c0, tlv)))'
        self.obj.e_str = cost

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
        SOCb = 'cp.multiply(EnR, SOC @ Mre) - t * cp.multiply(EtaCR, pce[:, 1:])'
        SOCb += ' + t * cp.multiply(REtaDR, pde[:, 1:]) == 0'
        self.SOCb.e_str = SOCb

        SOCb0 = 'cp.multiply(En, SOC[:, 0] - SOCinit) - t * cp.multiply(EtaC, pce[:, 0])'
        SOCb0 += ' + t * cp.multiply(REtaD, pde[:, 0]) == 0'
        self.SOCb0 = Constraint(name='SOCb0',
                                info='ESD1 SOC initial balance',
                                e_str=SOCb0,)

        self.SOCr.e_str = 'SOCend - SOC[:, -1] <= 0'

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
