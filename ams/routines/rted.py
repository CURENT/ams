"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import ZonalSum, VarSelect, NumOp
from ams.routines.dcopf import DCOPFData, DCOPFModel

from ams.opt.omodel import Var, Constraint, Objective

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
                          name='cru', src='cru',
                          tex_name=r'c_{r,u}', unit=r'$/(p.u.)',
                          model='SFRCost')
        self.crd = RParam(info='RegDown reserve coefficient',
                          name='crd',
                          src='crd',
                          tex_name=r'c_{r,d}',
                          unit=r'$/(p.u.)',
                          model='SFRCost',)
        # 1.2. reserve requirement
        self.du = RParam(info='RegUp reserve requirement (system base)',
                         name='du', src='du',
                         tex_name=r'd_{u}', unit='p.u.',
                         model='SFR',)
        self.dd = RParam(info='RegDown reserve requirement (system base)',
                         name='dd', src='dd',
                         tex_name=r'd_{d}', unit='p.u.',
                         model='SFR',)
        self.zg = RParam(info='generator zone data',
                         name='zg', src='zone',
                         tex_name='z_{one,g}',
                         model='StaticGen',)
        # 2. generator
        self.pg0 = RParam(info='generator active power start point (system base)',
                          name='pg0', src='p0',
                          tex_name=r'p_{g0}', unit='p.u.',
                          model='StaticGen',)
        self.R10 = RParam(info='10-min ramp rate (system base)',
                          name='R10', src='R10',
                          tex_name=r'R_{10}', unit='p.u./min',
                          model='StaticGen',)


class RTEDModel(DCOPFModel):
    """
    RTED model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)
        self.config.dth = 5/60  # time interval in hours
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

        self.info = 'Economic dispatch'
        self.type = 'DCED'
        # --- service ---
        self.gs = ZonalSum(u=self.zg, zone='Region',
                           name='gs', tex_name=r'\sum_{g}')
        self.gs.info = 'Sum Gen vars vector in shape of zone'

        self.R10h = NumOp(info='10-min ramp rate in hours',
                          name='R10h', tex_name=r'R_{10,h}',
                          u=self.R10, fun=lambda x: x / 60,)

        # --- vars ---
        self.pru = Var(info='RegUp reserve (system base)',
                       unit='p.u.', name='pru', tex_name=r'p_{r,u}',
                       model='StaticGen', nonneg=True,)
        self.prd = Var(info='RegDn reserve (system base)',
                       unit='p.u.', name='prd', tex_name=r'p_{r,d}',
                       model='StaticGen', nonneg=True,)
        # --- constraints ---
        self.rbu = Constraint(name='rbu', type='eq',
                              info='RegUp reserve balance',
                              e_str='gs @ pru - du',)
        self.rbd = Constraint(name='rbd', type='eq',
                              info='RegDn reserve balance',
                              e_str='gs @ prd - dd',)
        self.rru = Constraint(name='rru', type='uq',
                              info='RegUp reserve ramp',
                              e_str='pg + pru - pmax',)
        self.rrd = Constraint(name='rrd', type='uq',
                              info='RegDn reserve ramp',
                              e_str='-pg + prd - pmin',)
        self.rgu = Constraint(name='rgu', type='uq',
                              info='ramp up limit of generator output',
                              e_str='pg - pg0 - R10h',)
        self.rgd = Constraint(name='rgd', type='uq',
                              info='ramp down limit of generator output',
                              e_str='-pg + pg0 - R10h',)
        # --- objective ---
        self.obj.info = 'total generation and reserve cost'
        # NOTE: the product of dt and pg is processed using ``dot``, because dt is a numnber
        self.obj.e_str = 'sum(c2 @ (dth dot pg)**2) ' + \
                         '+ sum(c1 @ (dth dot pg)) + ug * c0 ' + \
                         '+ sum(cru * pru + crd * prd)'


class RTED(RTEDData, RTEDModel):
    """
    DC-based real-time economic dispatch (RTED).

    RTED extends DCOPF with:

    1. Parameter ``pg0``, which can be retrieved from dynamic simulation results.

    2. RTED has mapping dicts to interface with ANDES.

    3. RTED routine adds a function ``dc2ac`` to do the AC conversion using ACOPF

    4. zonal SFR reserve: decision variables ``pru`` and ``prd``; linear cost ``cru`` and ``crd``; requirement ``du`` and ``dd``

    5. generator ramping: start point ``pg0``; ramping limit ``R10``

    The function ``dc2ac`` sets the ``vBus`` value from solved ACOPF.
    Without this conversion, dynamic simulation might fail due to the gap between
    DC-based dispatch results and AC-based dynamic initialization.

    Notes
    -----
    1. objective function has been adjusted for RTED interval ``config.dth``, 5/60 [Hour] by default.
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


class RTED2Data(RTEDData):
    """
    Data for real-time economic dispatch, with ESD1.
    """

    def __init__(self):
        RTEDData.__init__(self)
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
        self.ge1 = RParam(info='gen of ESD1',
                          name='grg1', tex_name=r'g_{ESD1}',
                          model='ESD1', src='gen',)


class RTED2Model(RTEDModel):
    """
    RTED model with ESD1.
    """

    def __init__(self, system, config):
        RTEDModel.__init__(self, system, config)
        # --- service ---
        self.REtaD = NumOp(info='1/EtaD',
                           name='REtaD', tex_name=r'1/{Eta_D}',
                           u=self.EtaD, fun=np.reciprocal,)
        self.REn = NumOp(info='1/En',
                         name='REn', tex_name=r'1/{E_n}',
                         u=self.En, fun=np.reciprocal,)
        self.Mb = NumOp(info='10 times of max of pmax as big M',
                        name='Mb', tex_name=r'M_{big}',
                        u=self.pmax, fun=np.max,
                        rfun=np.dot, rargs=dict(b=10),)

        # --- vars ---
        self.SOC = Var(info='ESD1 SOC', unit='%',
                       name='SOC', tex_name=r'SOC',
                       model='ESD1', pos=True,)
        self.e1s = VarSelect(u=self.pg, indexer='ge1',
                             name='e1s', tex_name=r'S_{ESD1}',
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
        self.peb = Constraint(name='pges', type='eq',
                              info='Select ESD1 power from StaticGen',
                              e_str='e1s@pg - zc',)

        self.SOClb = Constraint(name='SOClb', type='uq',
                                info='ESD1 SOC lower bound',
                                e_str='-SOC + SOCmin',)
        self.SOCub = Constraint(name='SOCub', type='uq',
                                info='ESD1 SOC upper bound',
                                e_str='SOC - SOCmax',)

        self.zclb = Constraint(name='zclb', type='uq', info='zc lower bound',
                               e_str='- zc + pec',)
        self.zcub = Constraint(name='zcub', type='uq', info='zc upper bound',
                               e_str='zc - pec - Mb@(1-uc)',)
        self.zcub2 = Constraint(name='zcub2', type='uq', info='zc upper bound',
                                e_str='zc - Mb@uc',)

        SOCb = 'SOC - SOCinit - dth dot REn*EtaC*zc - dth dot REn*REtaD*(pec - zc)'
        self.SOCb = Constraint(name='SOCb', type='eq',
                               info='ESD1 SOC balance', e_str=SOCb,)


class RTED2(RTED2Data, RTED2Model):
    """
    RTED with energy storage :ref:`ESD1`.

    The bilinear term in the formulation is linearized with big-M method.
    """

    def __init__(self, system, config):
        RTED2Data.__init__(self)
        RTED2Model.__init__(self, system, config)
