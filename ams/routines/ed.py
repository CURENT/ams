"""
Real-time economic dispatch.
"""
import logging  # NOQA
import numpy as np  # NOQA

from ams.core.param import RParam  # NOQA
from ams.core.service import (ZonalSum, NumOpDual, NumHstack,
                              RampSub, NumOp, LoadScale)  # NOQA

from ams.routines.rted import RTEDData  # NOQA
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
        self.config.t = 1  # dispatch interval in hour

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

    ED extends DCOPF as follows:

    1. Power generation ``pg`` is extended to 2D matrix using argument
    ``horizon`` to represent the power generation of each generator in
    each time period (horizon).

    2. The rows correspond to generators and the columns correspond to time
    periods (horizons).

    3. Ramping limits ``rgu`` and ``rgd`` are introduced as 2D matrices to
    represent the upward and downward ramping limits for each generator.

    Notes
    -----
    1. Formulations has been adjusted with interval ``config.t``, 1 [Hour] by default.

    2. The tie-line flow has not been implemented in formulations
    """

    def __init__(self, system, config):
        EDData.__init__(self)
        EDModel.__init__(self, system, config)

# TODO: add data check
# if has model ``TimeSlot``, mandatory
# if has model ``Region``, optional
# if ``Region``, if ``Bus`` has param ``zone``, optional, if none, auto fill


class ED2Data(EDData):
    """
    Data for economic dispatch, with ESD1.
    """

    def __init__(self):
        EDData.__init__(self)
        self.En = RParam(info='Rated energy capacity',
                         name='En', src='En',
                         tex_name='E_n', unit='MWh',
                         model='ESD1',)
        self.SOCmin = RParam(info='Minimum required value for SOC in limiter',
                             name='SOCmin', src='SOCmin',
                             tex_name='SOC_{min}', unit='%',
                             model='ESD1',)
        self.SOCmax = RParam(info='Maximum allowed value for SOC in limiter',
                             name='SOCmax', src='SOCmax',
                             tex_name='SOC_{max}', unit='%',
                             model='ESD1',)
        self.SOCinit = RParam(info='Initial state of charge',
                              name='SOCinit', src='SOCinit',
                              tex_name=r'SOC_{init}', unit='%',
                              model='ESD1',)
        self.EtaC = RParam(info='Efficiency during charging',
                           name='EtaC', src='EtaC',
                           tex_name='Eta_C', unit='%',
                           model='ESD1',)
        self.EtaD = RParam(info='Efficiency during discharging',
                           name='EtaD', src='EtaD',
                           tex_name='Eta_D', unit='%',
                           model='ESD1',)
        self.genE = RParam(info='gen of ESD1',
                           name='genE', tex_name=r'g_{ESD1}',
                           model='ESD1', src='gen',)


class ED2Model(EDModel):
    """
    ED model with ESD1.
    """

    def __init__(self, system, config):
        EDModel.__init__(self, system, config)
        self.config.t = 1  # dispatch interval in hour

        self.info = 'Economic dispatch with energy storage'
        self.type = 'DCED'

        # --- ESD1 vars ---
        self.SOC = Var(info='ESD1 SOC in 2D',
                       name='SOC', tex_name=r'SOC', unit='%',
                       model='ESD1', pos=True,
                       horizon=self.timeslot,)

        self.ce = VarSelect(u=self.pg, indexer='genE',
                            name='ce', tex_name=r'C_{ESD1}',
                            info='Select ESD1 pg from StaticGen',)
        self.pec = Var(info='ESD1 charging power (system base)',
                       unit='p.u.', name='pec', tex_name=r'p_{c,ESD1}',
                       model='ESD1',
                       horizon=self.timeslot,)
        self.uc = Var(info='ESD1 charging decision',
                      name='uc', tex_name=r'u_{c}',
                      model='ESD1', boolean=True,
                      horizon=self.timeslot,)
        self.zc = Var(info='Aux var for ESD1 charging',
                      name='zc', tex_name=r'z_{c}',
                      model='ESD1', pos=True,
                      horizon=self.timeslot,)

        # --- constraints ---
        self.cpge = Constraint(name='cpge', type='eq',
                               info='Select ESD1 power from StaticGen',
                               e_str='multiply(ce, pg) - zc',)


class ED2(ED2Data, ED2Model):
    """
    DC-based multi-period economic dispatch (ED) with ESD1.
    """

    def __init__(self, system, config):
        ED2Data.__init__(self)
        ED2Model.__init__(self, system, config)
