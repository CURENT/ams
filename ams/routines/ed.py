"""
Economic dispatch routines.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import (NumOpDual, NumHstack,
                              RampSub, NumOp, LoadScale)

from ams.routines.rted import RTED, DGBase, ESD1Base

from ams.opt.omodel import Var, Constraint

logger = logging.getLogger(__name__)


class SRBase:
    """
    Base class for spinning reserve.
    """

    def __init__(self) -> None:
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

        # NOTE: define e_str in dispatch model
        self.prsb = Constraint(info='spinning reserve balance',
                               name='prsb', type='eq',)
        self.rsr = Constraint(info='spinning reserve requirement',
                              name='rsr', type='uq',)


class MPBase:
    """
    Base class for multi-period dispatch.
    """

    def __init__(self) -> None:
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

        self.pds = LoadScale(u=self.pd, sd=self.sd,
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

        self.ctrl.expand_dims = 1
        self.c0.expand_dims = 1
        self.pmax.expand_dims = 1
        self.pmin.expand_dims = 1
        self.pg0.expand_dims = 1
        self.rate_a.expand_dims = 1
        self.Pfinj.expand_dims = 1
        self.Pbusinj.expand_dims = 1
        self.gsh.expand_dims = 1

        # NOTE: extend pg to 2D matrix: row for gen and col for timeslot
        self.pg.horizon = self.timeslot
        self.pg.info = '2D Gen power'
        self.aBus.horizon = self.timeslot
        self.aBus.info = '2D Bus angle'


class ED(RTED):
    """
    DC-based multi-period economic dispatch (ED).
    Dispath interval ``config.t`` (:math:`T_{cfg}`) is introduced,
    1 [Hour] by default.

    ED extends DCOPF as follows:

    - Vars ``pg``, ``pru``, ``prd`` are extended to 2D
    - 2D Vars ``rgu`` and ``rgd`` are introduced
    - Param ``ug`` is sourced from ``EDTSlot.ug`` as commitment decisions

    Notes
    -----
    1. Formulations has been adjusted with interval ``config.t``

    2. The tie-line flow is not implemented in this model.
    """

    def __init__(self, system, config):
        RTED.__init__(self, system, config)
        MPBase.__init__(self)
        SRBase.__init__(self)

        self.config.add(OrderedDict((('t', 1),
                                     )))
        self.config.add_extra("_help",
                              t="time interval in hours",
                              )

        self.info = 'Economic dispatch'
        self.type = 'DCED'

        self.ug.info = 'unit commitment decisions'
        self.ug.model = 'EDTSlot'
        self.ug.src = 'ug'
        # self.ug.tex_name = r'u_{g}',

        self.dud.expand_dims = 1
        self.ddd.expand_dims = 1

        # --- Data Section ---
        self.ugt = NumOp(u=self.ug, fun=np.transpose,
                         name='ugt', tex_name=r'u_{g}',
                         info='input ug transpose',
                         no_parse=True)

        # --- Model Section ---
        # --- gen ---
        self.ctrle.u2 = self.ugt
        self.nctrle.u2 = self.ugt
        pglb = '-pg + mul(mul(nctrle, pg0), tlv) '
        pglb += '+ mul(mul(ctrle, tlv), pmin)'
        self.pglb.e_str = pglb
        pgub = 'pg - mul(mul(nctrle, pg0), tlv) '
        pgub += '- mul(mul(ctrle, tlv), pmax)'
        self.pgub.e_str = pgub

        self.pru.horizon = self.timeslot
        self.pru.info = '2D RegUp power'
        self.prd.horizon = self.timeslot
        self.prd.info = '2D RegDn power'

        self.prs.horizon = self.timeslot
        self.prsb.e_str = 'mul(ugt, pmax@tlv - pg) - prs'
        self.rsr.e_str = '-gs@prs + dsr'

        # --- line ---
        self.plf.horizon = self.timeslot
        self.plf.info = '2D Line flow'
        self.plflb.e_str = '-Bf@aBus - Pfinj@tlv - rate_a@tlv'
        self.plfub.e_str = 'Bf@aBus + Pfinj@tlv - rate_a@tlv'
        self.alflb.e_str = '-CftT@aBus - amax@tlv'
        self.alfub.e_str = 'CftT@aBus - amax@tlv'

        # --- power balance ---
        self.pb.e_str = 'Bbus@aBus + Pbusinj@tlv + Cl@pds + Csh@gsh@tlv - Cg@pg'

        # --- ramping ---
        self.rbu.e_str = 'gs@mul(ugt, pru) - mul(dud, tlv)'
        self.rbd.e_str = 'gs@mul(ugt, prd) - mul(ddd, tlv)'

        self.rru.e_str = 'mul(ugt, pg + pru) - mul(pmax, tlv)'
        self.rrd.e_str = 'mul(ugt, -pg + prd) + mul(pmin, tlv)'

        self.rgu.e_str = 'pg @ Mr - t dot RR30'
        self.rgd.e_str = '-pg @ Mr - t dot RR30'

        self.rgu0 = Constraint(name='rgu0',
                               info='Initial gen ramping up',
                               e_str='pg[:, 0] - pg0[:, 0] - R30',
                               type='uq',)
        self.rgd0 = Constraint(name='rgd0',
                               info='Initial gen ramping down',
                               e_str='- pg[:, 0] + pg0[:, 0] - R30',
                               type='uq',)

        # --- objective ---
        cost = 'sum(t**2 dot c2 @ pg**2)'
        cost += '+ t dot sum(c1 @ pg + csr @ prs)'
        cost += '+ sum(mul(ugt, mul(c0, tlv)))'
        self.obj.e_str = cost

    def _post_solve(self):
        """
        Overwrite ``_post_solve``.
        """
        # --- post-solving calculations ---
        # line flow: Bf@aBus + Pfinj
        mats = self.system.mats
        self.plf.optz.value = mats.Bf._v@self.aBus.v + self.Pfinj.v@self.tlv.v
        return True

    def dc2ac(self, **kwargs):
        """
        AC conversion ``dc2ac`` is not implemented yet for
        multi-period dispatch.
        """
        return NotImplementedError

    def unpack(self, **kwargs):
        """
        Multi-period dispatch will not unpack results from
        solver into devices.

        # TODO: unpack first period results, and allow input
        # to specify which period to unpack.
        """
        return None


class EDDG(ED, DGBase):
    """
    ED with distributed generation :ref:`DG`.

    Note that EDDG only inlcudes DG output power. If ESD1 is included,
    EDES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config):
        ED.__init__(self, system, config)
        DGBase.__init__(self)

        self.config.t = 1  # dispatch interval in hour

        self.info = 'Economic dispatch with distributed generation'
        self.type = 'DCED'

        # NOTE: extend vars to 2D
        self.pgdg.horizon = self.timeslot


class ESD1MPBase(ESD1Base):
    """
    Extended base class for energy storage in multi-period dispatch.
    """

    def __init__(self):
        ESD1Base.__init__(self)

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
        SOCb = 'mul(EnR, SOC @ Mre) - t dot mul(EtaCR, zce[:, 1:])'
        SOCb += ' + t dot mul(REtaDR, zde[:, 1:])'
        self.SOCb.e_str = SOCb

        SOCb0 = 'mul(En, SOC[:, 0] - SOCinit) - t dot mul(EtaC, zce[:, 0])'
        SOCb0 += ' + t dot mul(REtaD, zde[:, 0])'
        self.SOCb0 = Constraint(name='SOCb0', type='eq',
                                info='ESD1 SOC initial balance',
                                e_str=SOCb0,)

        self.SOCr = Constraint(name='SOCr', type='eq',
                               info='SOC requirement',
                               e_str='SOC[:, -1] - SOCinit',)


class EDES(ED, ESD1MPBase):
    """
    ED with energy storage :ref:`ESD1`.
    The bilinear term in the formulation is linearized with big-M method.
    """

    def __init__(self, system, config):
        ED.__init__(self, system, config)
        ESD1MPBase.__init__(self)

        self.config.t = 1  # dispatch interval in hour

        self.info = 'Economic dispatch with energy storage'
        self.type = 'DCED'

        # NOTE: extend vars to 2D
        self.pgdg.horizon = self.timeslot
        self.SOC.horizon = self.timeslot
        self.pce.horizon = self.timeslot
        self.pde.horizon = self.timeslot
        self.uce.horizon = self.timeslot
        self.ude.horizon = self.timeslot
        self.zce.horizon = self.timeslot
        self.zde.horizon = self.timeslot
