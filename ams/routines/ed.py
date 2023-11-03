"""
Real-time economic dispatch.
"""
import logging  # NOQA
from collections import OrderedDict  # NOQA
import numpy as np  # NOQA

from ams.core.param import RParam  # NOQA
from ams.core.service import (ZonalSum, NumOpDual, NumHstack,
                              RampSub, NumOp, LoadScale)  # NOQA

from ams.routines.rted import RTEDData, ESD1Base  # NOQA
from ams.routines.dcopf import DCOPFModel  # NOQA

from ams.core.service import VarSelect  # NOQA
from ams.opt.omodel import Var, Constraint  # NOQA

logger = logging.getLogger(__name__)


class EDData(RTEDData):
    """
    Economic dispatch data.
    """

    def __init__(self):
        RTEDData.__init__(self)
        # NOTE: Setting `ED.scale.owner` to `Horizon` will cause an error when calling `ED.scale.v`.
        # This is because `Horizon` is a group that only contains the model `TimeSlot`.
        # The `get` method of `Horizon` calls `andes.models.group.GroupBase.get` and results in an error.
        self.sd = RParam(info='zonal load factor for ED',
                         name='sd', tex_name=r's_{d}',
                         src='sd', model='EDTSlot')
        self.timeslot = RParam(info='Time slot for multi-period ED',
                               name='timeslot', tex_name=r't_{s,idx}',
                               src='idx', model='EDTSlot')
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
                         model='mats', src='Cl',)
        self.zl = RParam(info='zone of load',
                         name='zl', tex_name=r'z_{l}',
                         model='StaticLoad', src='zone',)


class EDModel(DCOPFModel):
    """
    Economic dispatch model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)

        self.config.add(OrderedDict((('t', 1),
                                     )))
        self.config.add_extra("_help",
                              t="time interval in hours",
                              )

        self.info = 'Economic dispatch'
        self.type = 'DCED'

        # --- vars ---
        # NOTE: extend pg to 2D matrix, where row is gen and col is timeslot
        self.pg.horizon = self.timeslot
        self.pg.info = '2D power generation (system base)'

        self.pn.horizon = self.timeslot
        self.pn.info = '2D Bus power injection (system base)'

        # --- constraints ---
        # --- power balance ---
        self.ds = ZonalSum(u=self.zb, zone='Region',
                           name='ds', tex_name=r'S_{d}',
                           info='Sum pl vector in shape of zone',)
        self.pdz = NumOpDual(u=self.ds, u2=self.pl,
                             fun=np.multiply,
                             rfun=np.sum, rargs=dict(axis=1),
                             expand_dims=0,
                             name='pdz', tex_name=r'p_{d,z}',
                             unit='p.u.', info='zonal load')
        self.pds = NumOpDual(u=self.sd, u2=self.pdz,
                             fun=np.multiply, rfun=np.transpose,
                             name='pds', tex_name=r'p_{d,s,t}',
                             unit='p.u.', info='Scaled total load as row vector')
        self.gs = ZonalSum(u=self.zg, zone='Region',
                           name='gs', tex_name=r'S_{g}',
                           info='Sum Gen vars vector in shape of zone')
        # NOTE: Spg @ pg returns a row vector
        self.pb.e_str = '- gs @ pg + pds'  # power balance

        # spinning reserve
        self.Rpmax = NumHstack(u=self.pmax, ref=self.timeslot,
                               name='Rpmax', tex_name=r'p_{max, R}',
                               info='Repetated pmax',)
        self.Rug = NumHstack(u=self.ug, ref=self.timeslot,
                             name='Rug', tex_name=r'u_{g, R}',
                             info='Repetated ug',)
        self.dsrpz = NumOpDual(u=self.pdz, u2=self.dsrp, fun=np.multiply,
                               name='dsrpz', tex_name=r'd_{sr, p, z}',
                               info='zonal spinning reserve requirement in percentage',)
        self.dsr = NumOpDual(u=self.dsrpz, u2=self.sd, fun=np.multiply,
                             rfun=np.transpose,
                             name='dsr', tex_name=r'd_{sr}',
                             info='zonal spinning reserve requirement',)
        self.sr = Constraint(name='sr', info='spinning reserve', type='uq',
                             e_str='-gs@multiply(Rpmax - pg, Rug) + dsr')

        # --- bus power injection ---
        self.Cli = NumOp(u=self.Cl, fun=np.linalg.pinv,
                         name='Cli', tex_name=r'C_{l}^{-1}',
                         info='inverse of Cl',)
        self.Rpd = LoadScale(u=self.zl, sd=self.sd, Cl=self.Cl,
                             name='Rpd', tex_name=r'p_{d,R}',
                             info='Scaled nodal load',)
        self.pinj.e_str = 'Cg @ (pn - Rpd) - pg'  # power injection

        # --- line limits ---
        self.RRA = NumHstack(u=self.rate_a, ref=self.timeslot,
                             name='RRA', tex_name=r'R_{ATEA,R}',
                             info='Repeated rate_a',)
        self.lub.e_str = 'PTDF @ (pn - Rpd) - RRA'
        self.llb.e_str = '-PTDF @ (pn - Rpd) - RRA'

        # --- ramping ---
        self.Mr = RampSub(u=self.pg, name='Mr', tex_name=r'M_{r}',
                          info='Subtraction matrix for ramping, (ng, ng-1)',)
        self.RR30 = NumHstack(u=self.R30, ref=self.Mr,
                              name='RR30', tex_name=r'R_{30,R}',
                              info='Repeated ramp rate as 2D matrix, (ng, ng-1)',)
        self.rgu = Constraint(name='rgu', info='Gen ramping up',
                              e_str='pg @ Mr - t dot RR30',
                              type='uq',)
        self.rgd = Constraint(name='rgd',
                              info='Gen ramping down',
                              e_str='-pg @ Mr - t dot RR30',
                              type='uq',)
        self.rgu0 = Constraint(name='rgu0',
                               info='Initial gen ramping up',
                               e_str='pg[:, 0] - pg0 - R30',
                               type='uq',)
        self.rgd0 = Constraint(name='rgd0',
                               info='Initial gen ramping down',
                               e_str='- pg[:, 0] + pg0 - R30',
                               type='uq',)

        # --- objective ---
        # NOTE: no need to fix objective function
        gcost = 'sum(c2 @ (t dot pg)**2 + c1 @ (t dot pg) + ug * c0)'
        rcost = ' + sum(csr * ug * (Rpmax - pg))'
        self.obj.e_str = gcost + rcost

    def unpack(self, **kwargs):
        """
        ED will not unpack results from solver into devices
        because the resutls are multi-time-period.
        """
        return None


class ED(EDData, EDModel):
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
        EDData.__init__(self)
        EDModel.__init__(self, system, config)

# TODO: add data check
# if has model ``TimeSlot``, mandatory
# if has model ``Region``, optional
# if ``Region``, if ``Bus`` has param ``zone``, optional, if none, auto fill


class ED2(EDData, EDModel, ESD1Base):
    """
    ED with energy storage :ref:`ESD1`.
    The bilinear term in the formulation is linearized with big-M method.
    """

    def __init__(self, system, config):
        EDData.__init__(self)
        EDModel.__init__(self, system, config)
        ESD1Base.__init__(self)

        self.config.t = 1  # dispatch interval in hour

        self.info = 'Economic dispatch with energy storage'
        self.type = 'DCED'

        self.SOC.horizon = self.timeslot
        self.pge.horizon = self.timeslot
        self.ued.horizon = self.timeslot
        self.zue.horizon = self.timeslot
