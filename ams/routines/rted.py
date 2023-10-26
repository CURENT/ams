"""
Real-time economic dispatch.
"""
import logging  # NOQA
from collections import OrderedDict  # NOQA
import numpy as np  # NOQA

from ams.core.param import RParam  # NOQA
from ams.core.service import ZonalSum, VarSelect, NumOp, NumOpDual  # NOQA
from ams.routines.dcopf import DCOPFData, DCOPFModel  # NOQA

from ams.opt.omodel import Var, Constraint  # NOQA

logger = logging.getLogger(__name__)


class RTEDData(DCOPFData):
    """
    Data for real-time economic dispatch.
    """

    def __init__(self):
        DCOPFData.__init__(self)

        # 1. reserve
        # 1.1. reserve cost
        self.cru = RParam(info='RegUp reserve coefficient',
                          name='cru', tex_name=r'c_{r,u}',
                          model='SFRCost', src='cru',
                          unit=r'$/(p.u.)',)
        self.crd = RParam(info='RegDown reserve coefficient',
                          name='crd', tex_name=r'c_{r,d}',
                          model='SFRCost', src='crd',
                          unit=r'$/(p.u.)',)
        # 1.2. reserve requirement
        self.du = RParam(info='RegUp reserve requirement in percentage',
                         name='du', tex_name=r'd_{u}',
                         model='SFR', src='du',
                         unit='%',)
        self.dd = RParam(info='RegDown reserve requirement in percentage',
                         name='dd', tex_name=r'd_{d}',
                         model='SFR', src='dd',
                         unit='%',)
        self.zb = RParam(info='Bus zone',
                         name='zb', tex_name='z_{one,bus}',
                         model='Bus', src='zone', )
        self.zg = RParam(info='generator zone data',
                         name='zg', tex_name='z_{one,g}',
                         model='StaticGen', src='zone',)
        # 2. generator
        self.R10 = RParam(info='10-min ramp rate (system base)',
                          name='R10', tex_name=r'R_{10}',
                          model='StaticGen', src='R10',
                          unit='p.u./h',)


class RTEDModel(DCOPFModel):
    """
    RTED model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)
        self.config.t = 5/60  # time interval in hours
        self.map1 = OrderedDict([
            ('StaticGen', {
                'pg0': 'p',
            }),
        ])
        # NOTE: define map2
        # DC-based RTED assume bus voltage to be 1
        # here we mock the ACOPF bus voltage results to fit the interface
        self.map2 = OrderedDict([
            ('Bus', {
                'vBus': 'v0',
            }),
            ('StaticGen', {
                'pg': 'p0',
            }),
        ])
        self.info = 'Real-time economic dispatch'
        self.type = 'DCED'

        # --- service ---
        self.gs = ZonalSum(u=self.zg, zone='Region',
                           name='gs', tex_name=r'S_{g}',
                           info='Sum Gen vars vector in shape of zone')

        # --- vars ---
        self.pru = Var(info='RegUp reserve (system base)',
                       unit='p.u.', name='pru', tex_name=r'p_{r,u}',
                       model='StaticGen', nonneg=True,)
        self.prd = Var(info='RegDn reserve (system base)',
                       unit='p.u.', name='prd', tex_name=r'p_{r,d}',
                       model='StaticGen', nonneg=True,)
        # --- constraints ---
        self.ds = ZonalSum(u=self.zb, zone='Region',
                           name='ds', tex_name=r'S_{d}',
                           info='Sum pd vector in shape of zone',)
        self.pdz = NumOpDual(u=self.ds, u2=self.pl,
                             fun=np.multiply,
                             rfun=np.sum, rargs=dict(axis=1),
                             expand_dims=0,
                             name='pdz', tex_name=r'p_{d,z}',
                             unit='p.u.', info='zonal load')
        self.dud = NumOpDual(u=self.pdz, u2=self.du, fun=np.multiply,
                             rfun=np.reshape, rargs=dict(newshape=(-1,)),
                             name='dud', tex_name=r'd_{u, d}',
                             info='zonal RegUp reserve requirement',)
        self.ddd = NumOpDual(u=self.pdz, u2=self.dd, fun=np.multiply,
                             rfun=np.reshape, rargs=dict(newshape=(-1,)),
                             name='ddd', tex_name=r'd_{d, d}',
                             info='zonal RegDn reserve requirement',)
        self.rbu = Constraint(name='rbu', type='eq',
                              info='RegUp reserve balance',
                              e_str='gs @ multiply(ug, pru) - dud',)
        self.rbd = Constraint(name='rbd', type='eq',
                              info='RegDn reserve balance',
                              e_str='gs @ multiply(ug, prd) - ddd',)
        self.rru = Constraint(name='rru', type='uq',
                              info='RegUp reserve ramp',
                              e_str='multiply(ug, pg + pru) - pmax',)
        self.rrd = Constraint(name='rrd', type='uq',
                              info='RegDn reserve ramp',
                              e_str='multiply(ug, -pg + prd) - pmin',)
        self.rgu = Constraint(name='rgu', type='uq',
                              info='ramp up limit of generator output',
                              e_str='multiply(ug, pg-pg0-R10)',)
        self.rgd = Constraint(name='rgd', type='uq',
                              info='ramp down limit of generator output',
                              e_str='multiply(ug, -pg+pg0-R10)',)
        # --- objective ---
        self.obj.info = 'total generation and reserve cost'
        # NOTE: the product of dt and pg is processed using ``dot``, because dt is a numnber
        self.obj.e_str = 'sum(c2 @ (t dot pg)**2) ' + \
                         '+ sum(c1 @ (t dot pg)) + ug * c0 ' + \
                         '+ sum(cru * pru + crd * prd)'


class RTED(RTEDData, RTEDModel):
    """
    DC-based real-time economic dispatch (RTED).
    RTED extends DCOPF with:

    1. Param ``pg0``, which can be retrieved from dynamic simulation results.

    2. RTED has mapping dicts to interface with ANDES.

    3. RTED routine adds a function ``dc2ac`` to do the AC conversion using ACOPF

    4. Vars for zonal SFR reserve: ``pru`` and ``prd``;

    5. Param for linear cost of zonal SFR reserve ``cru`` and ``crd``;

    6. Param for SFR requirement ``du`` and ``dd``;

    7. Param for generator ramping: start point ``pg0`` and ramping limit ``R10``;

    The function ``dc2ac`` sets the ``vBus`` value from solved ACOPF.
    Without this conversion, dynamic simulation might fail due to the gap between
    DC-based dispatch results and AC-based dynamic initialization.

    Notes
    -----
    1. Formulations has been adjusted with interval ``config.t``, 5/60 [Hour] by default.

    2. The tie-line flow has not been implemented in formulations.
    """

    def __init__(self, system, config):
        RTEDData.__init__(self)
        RTEDModel.__init__(self, system, config)

    def dc2ac(self, **kwargs):
        """
        Convert the RTED results with ACOPF.

        Overload ``dc2ac`` method.
        """
        if self.exec_time == 0 or self.exit_code != 0:
            logger.warning('RTED is not executed successfully, quit conversion.')
            return False
        # set pru and prd into pmin and pmax
        pr_idx = self.pru.get_idx()
        pmin0 = self.system.StaticGen.get(src='pmin', attr='v', idx=pr_idx)
        pmax0 = self.system.StaticGen.get(src='pmax', attr='v', idx=pr_idx)
        p00 = self.system.StaticGen.get(src='p0', attr='v', idx=pr_idx)

        # solve ACOPF
        ACOPF = self.system.ACOPF
        pmin = pmin0 + self.prd.v
        pmax = pmax0 - self.pru.v
        self.system.StaticGen.set(src='pmin', attr='v', idx=pr_idx, value=pmin)
        self.system.StaticGen.set(src='pmax', attr='v', idx=pr_idx, value=pmax)
        self.system.StaticGen.set(src='p0', attr='v', idx=pr_idx, value=self.pg.v)
        ACOPF.run()
        self.pg.v = ACOPF.pg.v

        # NOTE: mock results to fit interface with ANDES
        self.vBus = ACOPF.vBus

        # reset pmin, pmax, p0
        self.system.StaticGen.set(src='pmin', attr='v', idx=pr_idx, value=pmin0)
        self.system.StaticGen.set(src='pmax', attr='v', idx=pr_idx, value=pmax0)
        self.system.StaticGen.set(src='p0', attr='v', idx=pr_idx, value=p00)
        self.system.recent = self

        self.is_ac = True
        logger.warning('RTED is converted to AC.')
        return True

    def run(self, **kwargs):
        """
        Overload ``run()`` method.

        Notes
        -----
        1. remove ``vBus`` if has been converted with ``dc2ac``
        """
        if self.is_ac:
            delattr(self, 'vBus')
            self.is_ac = False
        return super().run(**kwargs)


class ESD1Base:
    """
    Base class for ESD1 used in DCED.
    """

    def __init__(self):
        # --- params ---
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
                           tex_name=r'\eta_c', unit='%',
                           model='ESD1',)
        self.EtaD = RParam(info='Efficiency during discharging',
                           name='EtaD', src='EtaD',
                           tex_name=r'\eta_d', unit='%',
                           model='ESD1',)
        self.genE = RParam(info='gen of ESD1',
                           name='genE', tex_name=r'g_{ESD1}',
                           model='ESD1', src='gen',)

        # --- service ---
        self.REtaD = NumOp(name='REtaD', tex_name=r'\frac{1}{\eta_d}',
                           u=self.EtaD, fun=np.reciprocal,)
        self.REn = NumOp(name='REn', tex_name=r'\frac{1}{E_n}',
                         u=self.En, fun=np.reciprocal,)
        self.Mb = NumOp(info='10 times of max of pmax as big M',
                        name='Mb', tex_name=r'M_{big}',
                        u=self.pmax, fun=np.max,
                        rfun=np.dot, rargs=dict(b=10),
                        array_out=False,)

        # --- vars ---
        self.SOC = Var(info='ESD1 SOC', unit='%',
                       name='SOC', tex_name=r'SOC',
                       model='ESD1', pos=True,)
        self.ce = VarSelect(u=self.pg, indexer='genE',
                            name='ce', tex_name=r'C_{ESD1}',
                            info='Select ESD1 pg from StaticGen',)
        self.pec = Var(info='ESD1 charging power (system base)',
                       unit='p.u.', name='pec', tex_name=r'p_{c,ESD1}',
                       model='ESD1',)
        self.uc = Var(info='ESD1 charging decision',
                      name='uc', tex_name=r'u_{c}',
                      model='ESD1', boolean=True,)
        self.zc = Var(info='Aux var for ESD1 charging',
                      name='zc', tex_name=r'z_{c}',
                      model='ESD1', pos=True,)

        # --- constraints ---
        self.cpge = Constraint(name='cpge', type='eq',
                               info='Select ESD1 power from StaticGen',
                               e_str='multiply(ce, pg) - zc',)

        self.SOClb = Constraint(name='SOClb', type='uq',
                                info='ESD1 SOC lower bound',
                                e_str='-SOC + SOCmin',)
        self.SOCub = Constraint(name='SOCub', type='uq',
                                info='ESD1 SOC upper bound',
                                e_str='SOC - SOCmax',)

        self.zclb = Constraint(name='zclb', type='uq', info='zc lower bound',
                               e_str='- zc + pec',)
        self.zcub = Constraint(name='zcub', type='uq', info='zc upper bound',
                               e_str='zc - pec - Mb dot (1-uc)',)
        self.zcub2 = Constraint(name='zcub2', type='uq', info='zc upper bound',
                                e_str='zc - Mb dot uc',)

        SOCb = 'SOC - SOCinit - t dot REn * EtaC * zc'
        SOCb += '- t dot REn * REtaD * (pec - zc)'
        self.SOCb = Constraint(name='SOCb', type='eq',
                               info='ESD1 SOC balance', e_str=SOCb,)


class RTED2(RTEDData, RTEDModel, ESD1Base):
    """
    RTED with energy storage :ref:`ESD1`.
    The bilinear term in the formulation is linearized with big-M method.
    """

    def __init__(self, system, config):
        RTEDData.__init__(self)
        RTEDModel.__init__(self, system, config)
        ESD1Base.__init__(self)
        self.info = 'Real-time economic dispatch with energy storage'
        self.type = 'DCED'
