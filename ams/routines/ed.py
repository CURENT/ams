"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.routines.dcopf import DCOPFData, DCOPFModel

from ams.opt.omodel import Var, Constraint, Objective

logger = logging.getLogger(__name__)


class EDData(DCOPFData):
    """
    Economic dispatch data.
    """
    
    def __init__(self):
        DCOPFData.__init__(self)
        # 1. reserve
        # 1.1. reserve cost
        self.cru = RParam(info='RegUp reserve coefficient',
                          name='cru',
                          src='cru',
                          tex_name=r'c_{r,u}',
                          unit=r'$/(p.u.)',
                          owner_name='SFRCost',
                          )
        self.crd = RParam(info='RegDown reserve coefficient',
                          name='crd',
                          src='crd',
                          tex_name=r'c_{r,d}',
                          unit=r'$/(p.u.)',
                          owner_name='SFRCost',
                          )
        # 1.2. reserve requirement
        self.du = RParam(info='RegUp reserve requirement (system base)',
                         name='du',
                         src='du',
                         tex_name=r'd_{u}',
                         unit='p.u.',
                         owner_name='SFR',
                         )
        self.dd = RParam(info='RegDown reserve requirement (system base)',
                         name='dd',
                         src='dd',
                         tex_name=r'd_{d}',
                         unit='p.u.',
                         owner_name='SFR',
                         )
        self.prs = RParam(info='sum matrix of reserve',
                          name='prs',
                          src='prs',
                          tex_name=r'\sum',
                          owner_name='SFR',
                          )
        # 2. generator
        self.pg0 = RParam(info='generator active power start point (system base)',
                          name='pg0',
                          src='pg0',
                          tex_name=r'p_{g0}',
                          unit='p.u.',
                          owner_name='StaticGen',
                          )
        self.R10 = RParam(info='10-min ramp rate (system base)',
                          name='R10',
                          src='R10',
                          tex_name=r'R_{10}',
                          unit='p.u./min',
                          owner_name='StaticGen',
                          )


class EDModel(DCOPFModel):
    """
    Economic dispatch model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)
        self.info = 'Economic dispatch'
        self.type = 'DCED'
        # --- vars ---
        self.pru = Var(info='RegUp reserve (system base)',
                       unit='p.u.',
                       name='pru',
                       tex_name=r'p_{r,u}',
                       owner_name='StaticGen',
                       nonneg=True,
                       )
        self.prd = Var(info='RegDn reserve (system base)',
                       unit='p.u.',
                       name='prd',
                       tex_name=r'p_{r,d}',
                       owner_name='StaticGen',
                       nonneg=True,
                       )
        # --- constraints ---
        self.rbu = Constraint(name='rbu',
                              info='RegUp reserve balance',
                              e_str='prs @ pru - du',
                              type='eq',
                              )
        self.rbd = Constraint(name='rbd',
                              info='RegDn reserve balance',
                              e_str='prs @ prd - dd',
                              type='eq',
                              )
        self.rru = Constraint(name='rru',
                              info='RegUp reserve ramp',
                              e_str='pg + pru - pmax',
                              type='uq',
                              )
        self.rrd = Constraint(name='rrd',
                              info='RegDn reserve ramp',
                              e_str='-pg + prd - pmin',
                              type='uq',
                              )
        self.rgu = Constraint(name='rgu',
                              info='ramp up limit of generator output',
                              e_str='pg - pg0 - R10',
                              type='uq',
                              )
        self.rgd = Constraint(name='rgd',
                              info='ramp down limit of generator output',
                              e_str='-pg + pg0 - R10',
                              type='uq',
                              )
        # --- objective ---
        self.obj = Objective(name='tc',
                             info='total generation and reserve cost',
                             e_str='sum(c2 * pg**2 + c1 * pg + c0 + cru * pru + crd * prd)',
                             sense='min',
                             )


class ED(EDData, EDModel):
    """
    DC-based economic dispatch (ED).

    ED extends DCOPF with:

    1. zonal SFR reserve: decision reserve ``pru`` and ``prd``; linear cost ``cru`` and ``crd``; requirement ``du`` and ``dd``

    2. generator ramping: start point ``pg0``; ramping limit ``R10``

    Notes
    -----
    1. ED is a DC-based model, which assumes bus voltage to be 1
    2. ED adds a data check on ``pg0``, if all zeros, correct as the value of ``p0``
    """

    def __init__(self, system, config):
        EDData.__init__(self)
        EDModel.__init__(self, system, config)

    def _data_check(self):
        """
        Check data.

        Overload ``_data_check`` method in ``DCOPF``.
        """
        if np.all(self.pg0.v == 0):
            logger.warning('RTED pg0 are all zeros, correct to p0.')
            idx = self.pg.get_idx()
            self.pg0._v = self.system.StaticGen.get(src='p0', attr='v', idx=idx)
        super()._data_check()
        return True


class RTEDModel(EDModel):
    """
    RTED model.
    """

    def __init__(self, system, config):
        EDModel.__init__(self, system, config)
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


class RTED(EDData, RTEDModel):
    """
    DC-based real-time economic dispatch (RTED).

    RTED differs from ED in that:
    1. Parameter ``pg0`` can be retrieved from dynamic simulation results.
    2. RTED has mapping dicts to interface with ANDES.
    3. RTED routine adds a function ``dc2ac`` to do the conversion using ACOPF
    
    Without ``dc2ac`` conversion, dynamic simulation might fail due to the gap between
    DC-based dispatch results and AC-based dynamic initialization.
    The function will add a Var ``aBus`` copied from solved ACOPF to fit the ANDES interface.
    """

    def __init__(self, system, config):
        EDData.__init__(self)
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
        # remove vBus if has been converted to AC using ``dc2ac``
        if self.is_ac:
            delattr(self, 'vBus')
            del self.vars['vBus']
            self.is_ac = False
        super().run(**kwargs)
