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


class UCData(DCOPFData):
    """
    UC data.
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
        self.du = RParam(info='RegUp reserve requirement',
                         name='du',
                         src='du',
                         tex_name=r'd_{u}',
                         unit='p.u.',
                         owner_name='SFR',
                         )
        self.dd = RParam(info='RegDown reserve requirement',
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


class UCModel(DCOPFModel):
    """
    UC model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)
        self.info = 'unit commitment'
        self.type = 'DCUC'
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


class UC(UCData, UCModel):
    """
    DC-based unit commitment (UC).
    """

    def __init__(self, system, config):
        UCData.__init__(self)
        UCModel.__init__(self, system, config)
