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


class RTEDData(DCOPFData):
    """
    RTED parameters and variables.
    """

    def __init__(self):
        DCOPFData.__init__(self)
        # 1. reserve
        # 1.1. reserve cost
        self.cru = RParam(info='RegUp reserve coefficient',
                          name='cru',
                          src='cru',
                          tex_name=r'c_{r,u}',
                          unit=r'$/(p.u.^2)',
                          owner_name='RCost',
                          )
        self.crd = RParam(info='RegDown reserve coefficient',
                          name='crd',
                          src='crd',
                          tex_name=r'c_{r,d}',
                          unit=r'$/(p.u.^2)',
                          owner_name='RCost',
                          )
        # 1.2. reserve requirement
        self.du = RParam(info='RegUp reserve requirement',
                         name='du',
                         src='du',
                         tex_name=r'd_{u}',
                         unit='p.u.',
                         owner_name='AGCR',
                         )
        self.dd = RParam(info='RegDown reserve requirement',
                         name='dd',
                         src='dd',
                         tex_name=r'd_{d}',
                         unit='p.u.',
                         owner_name='AGCR',
                         )
        self.prs = RParam(info='sum matrix of reserve',
                          name='prs',
                          src='prs',
                          tex_name=r'\sum',
                          owner_name='AGCR',
                          )
        # 1.3 reserve ramp rate
        # FIXME: seems not used
        self.Ragc = RParam(info='AGC ramp rate',
                           name='Ragc',
                           src='Ragc',
                           tex_name=r'R_{agc}',
                           unit='p.u./min',
                           owner_name='StaticGen',
                           )
        # 2. generator
        self.pg0 = RParam(info='generator active power start point',
                          name='pg0',
                          src='pg0',
                          tex_name=r'p_{g0}',
                          unit='p.u.',
                          owner_name='StaticGen',
                          )
        self.R10 = RParam(info='10-min ramp rate',
                          name='R10',
                          src='R10',
                          tex_name=r'R_{10}',
                          unit='p.u./min',
                          owner_name='StaticGen',
                          )


class RTEDModel(DCOPFModel):
    """
    RTED model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)
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
        # --- vars ---
        self.pru = Var(info='RegUp reserve',
                       unit='p.u.',
                       name='pru',
                       tex_name=r'p_{r,u}',
                       owner_name='StaticGen',
                       nonneg=True,
                       )
        self.prd = Var(info='RegDn reserve',
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


class RTED(RTEDData, RTEDModel):
    """
    DCOPF dispatch routine.
    """

    def __init__(self, system, config):
        RTEDData.__init__(self)
        RTEDModel.__init__(self, system, config)
    
    def _data_check(self):
        """
        Check data.

        Overload ``_data_check`` method in ``DCOPF``.
        """
        if np.all(self.pg0.v == 0):
            logger.warning('RTED pg0 are all zeros, correct to p0.')
            idx = self.pg.get_idx()
            self.pg0.is_set = True
            self.pg0._v = self.system.StaticGen.get(src='p0', attr='v', idx=idx)
        super()._data_check()
        return True

    def smooth(self, **kwargs):
        """
        Smooth the RTED results with ACOPF.

        Overload ``smooth`` method in ``DCOPF``.
        """
        if self.exec_time == 0 or self.exit_code != 0:
            logger.warning('RTED is not executed successfully, no smoothing.')
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

        self.is_smooth = True
        logger.warning('RTED is smoothed with ACOPF.')
        return True
