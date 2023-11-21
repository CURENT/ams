"""
Economic dispatch routines.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import (ZonalSum, NumOpDual, NumHstack,
                              RampSub, NumOp, LoadScale)

from ams.routines.rted import RTED, ESD1Base

from ams.opt.omodel import Constraint

logger = logging.getLogger(__name__)


class ED(RTED):
    """
    DC-based multi-period economic dispatch (ED).
    Dispath interval ``config.t`` (:math:`T_{cfg}`) is introduced,
    1 [Hour] by default.

    ED extends DCOPF as follows:

    1. Var ``pg`` is extended to 2D

    2. 2D Vars ``rgu`` and ``rgd`` are introduced

    Notes
    -----
    1. Formulations has been adjusted with interval ``config.t``

    2. The tie-line flow is not implemented in this model.
    """

    def __init__(self, system, config):
        RTED.__init__(self, system, config)

        self.config.add(OrderedDict((('t', 1),
                                     )))
        self.config.add_extra("_help",
                              t="time interval in hours",
                              )

        self.info = 'Economic dispatch'
        self.type = 'DCED'

        self.ug.expand_dims = 1
        self.c0.expand_dims = 1
        self.pmax.expand_dims = 1
        self.pmin.expand_dims = 1
        self.rate_a.expand_dims = 1
        self.dud.expand_dims = 1
        self.ddd.expand_dims = 1

        # --- params ---
        # NOTE: Setting `ED.scale.owner` to `Horizon` will cause an error when calling `ED.scale.v`.
        # This is because `Horizon` is a group that only contains the model `TimeSlot`.
        # The `get` method of `Horizon` calls `andes.models.group.GroupBase.get` and results in an error.
        self.sd = RParam(info='zonal load factor for ED',
                         name='sd', tex_name=r's_{d}',
                         src='sd', model='EDTSlot')
        self.timeslot = RParam(info='Time slot for multi-period ED',
                               name='timeslot', tex_name=r't_{s,idx}',
                               src='idx', model='EDTSlot',
                               no_parse=True)
        self.R30 = RParam(info='30-min ramp rate (system base)',
                          name='R30', tex_name=r'R_{30}',
                          src='R30', unit='p.u./min',
                          model='StaticGen')
        self.dsrp = RParam(info='spinning reserve requirement in percentage',
                           name='dsr', tex_name=r'd_{sr}',
                           model='SR', src='demand',
                           unit='%',)
        self.csr = RParam(info='cost for spinning reserve',
                          name='csr', tex_name=r'c_{sr}',
                          model='SRCost', src='csr',
                          unit=r'$/(p.u.*h)',
                          indexer='gen', imodel='StaticGen',)
        self.Cl = RParam(info='connection matrix for Load and Bus',
                         name='Cl', tex_name=r'C_{l}',
                         model='mats', src='Cl',
                         const=True,)
        self.zl = RParam(info='zone of load',
                         name='zl', tex_name=r'z_{l}',
                         model='StaticLoad', src='zone',
                         no_parse=True)

        # --- vars ---
        # NOTE: extend pg to 2D matrix, where row is gen and col is timeslot
        self.pg.horizon = self.timeslot
        self.pg.info = '2D power generation (system base)'

        self.pn.horizon = self.timeslot
        self.pn.info = '2D Bus power injection (system base)'

        self.pru.horizon = self.timeslot
        self.pru.info = '2D RegUp power (system base)'

        self.prd.horizon = self.timeslot
        self.prd.info = '2D RegDn power (system base)'

        # --- constraints ---
        # --- power balance ---
        self.pds = NumOpDual(u=self.sd, u2=self.pdz,
                             fun=np.multiply, rfun=np.transpose,
                             name='pds', tex_name=r'p_{d,s,t}',
                             unit='p.u.', info='Scaled total load as row vector')
        # NOTE: Spg @ pg returns a row vector
        self.pb.e_str = '- gs @ pg + pds'  # power balance

        # spinning reserve
        self.tlv = NumOp(u=self.timeslot, fun=np.ones_like,
                         args=dict(dtype=float),
                         expand_dims=0,
                         info='time length vector',
                         no_parse=True)
        self.dsrpz = NumOpDual(u=self.pdz, u2=self.dsrp, fun=np.multiply,
                               name='dsrpz', tex_name=r'd_{sr, p, z}',
                               info='zonal spinning reserve requirement in percentage',)
        self.dsr = NumOpDual(u=self.dsrpz, u2=self.sd, fun=np.multiply,
                             rfun=np.transpose,
                             name='dsr', tex_name=r'd_{sr}',
                             info='zonal spinning reserve requirement',)

        self.sr = Constraint(name='sr', info='spinning reserve', type='uq',
                             e_str='-gs@mul(mul(pmax, tlv) - pg, mul(ug, tlv)) + dsr')

        # --- bus power injection ---
        self.Cli = NumOp(u=self.Cl, fun=np.linalg.pinv,
                         name='Cli', tex_name=r'C_{l}^{-1}',
                         info='inverse of Cl',
                         const=True)
        self.Rpd = LoadScale(u=self.zl, sd=self.sd, Cl=self.Cl,
                             name='Rpd', tex_name=r'p_{d,R}',
                             info='Scaled nodal load',)
        self.pinj.e_str = 'Cg @ (pn - Rpd) - pg'  # power injection

        # --- line limits ---
        self.lub.e_str = 'PTDF @ (pn - Rpd) - mul(rate_a, tlv)'
        self.llb.e_str = '-PTDF @ (pn - Rpd) - mul(rate_a, tlv)'

        # --- ramping ---
        self.Mr = RampSub(u=self.pg, name='Mr', tex_name=r'M_{r}',
                          info='Subtraction matrix for ramping, (ng, ng-1)',)
        self.RR30 = NumHstack(u=self.R30, ref=self.Mr,
                              name='RR30', tex_name=r'R_{30,R}',
                              info='Repeated ramp rate as 2D matrix, (ng, ng-1)',)

        self.rbu.e_str = 'gs @ mul(mul(ug, tlv), pru) - mul(dud, tlv)'
        self.rbd.e_str = 'gs @ mul(mul(ug, tlv), prd) - mul(ddd, tlv)'

        self.rru.e_str = 'mul(mul(ug, tlv), pg + pru) - mul(pmax, tlv)'
        self.rrd.e_str = 'mul(mul(ug, tlv), -pg + prd) - mul(pmin, tlv)'

        self.rgu.e_str = 'pg @ Mr - t dot RR30'
        self.rgd.e_str = '-pg @ Mr - t dot RR30'

        self.rgu0 = Constraint(name='rgu0',
                               info='Initial gen ramping up',
                               e_str='pg[:, 0] - pg0 - R30',
                               type='uq',)
        self.rgd0 = Constraint(name='rgd0',
                               info='Initial gen ramping down',
                               e_str='- pg[:, 0] + pg0 - R30',
                               type='uq',)

        # --- objective ---
        cost = 'sum(c2 @ (t dot pg)**2 + c1 @ (t dot pg))'
        cost += '+ sum(mul(mul(ug, c0), tlv))'  # constant cost
        cost += ' + sum(csr @ (mul(mul(ug, pmax), tlv) - pg))'  # spinning reserve cost
        self.obj.e_str = cost

    def unpack(self, **kwargs):
        """
        ED will not unpack results from solver into devices
        because the resutls are multi-time-period.
        """
        return None


# TODO: add data check
# if has model ``TimeSlot``, mandatory
# if has model ``Region``, optional
# if ``Region``, if ``Bus`` has param ``zone``, optional, if none, auto fill


class ED2(ED, ESD1Base):
    """
    ED with energy storage :ref:`ESD1`.
    The bilinear term in the formulation is linearized with big-M method.
    """

    def __init__(self, system, config):
        ED.__init__(self, system, config)
        ESD1Base.__init__(self)

        self.config.t = 1  # dispatch interval in hour

        self.info = 'Economic dispatch with energy storage'
        self.type = 'DCED'

        # NOTE: extend vars to 2D
        self.SOC.horizon = self.timeslot
        self.pce.horizon = self.timeslot
        self.pde.horizon = self.timeslot
        self.uce.horizon = self.timeslot
        self.ude.horizon = self.timeslot
        self.zce.horizon = self.timeslot
        self.zde.horizon = self.timeslot

        self.Mre = RampSub(u=self.SOC, name='Mre', tex_name=r'M_{r,E}',
                           info='Subtraction matrix for SOC',
                           no_parse=True)
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
        self.SOCb0 = Constraint(name='SOCb', type='eq',
                                info='ESD1 SOC initial balance',
                                e_str=SOCb0,)

        self.SOCr = Constraint(name='SOCr', type='eq',
                               info='SOC requirement',
                               e_str='SOC[:, -1] - SOCinit',)
