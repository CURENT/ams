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
        self.csu = RParam(info='startup cost',
                          name='csu',
                          src='csu',
                          tex_name=r'c_{su}',
                          unit='$',
                          owner_name='GCost',
                          )
        self.csd = RParam(info='shutdown cost',
                          name='csd',
                          src='csd',
                          tex_name=r'c_{sd}',
                          unit='$',
                          owner_name='GCost',
                          )
        self.R30 = RParam(info='30-min ramp rate (system base)',
                          name='R30',
                          src='R30',
                          tex_name=r'R_{30}',
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
        self.ug = Var(info='gen connection status',
                       name='ug',
                       tex_name=r'u_{g}',
                       owner_name='StaticGen',
                       bool=True,
                       src='u',
                       )
        # --- constraints ---
        # self.rgu = Constraint(name='rgu',
        #                       info='ramp up limit of generator output',
        #                       e_str='pg - pg0 - R10',
        #                       type='uq',
        #                       )
        # self.rgd = Constraint(name='rgd',
        #                       info='ramp down limit of generator output',
        #                       e_str='-pg + pg0 - R10',
        #                       type='uq',
        #                       )
        # --- objective ---
        self.obj = Objective(name='tc',
                             info='total generation and reserve cost',
                             e_str='sum(pg**2 * ug + c1 * pg * ug+ c0 * ug + csu * ug + csd * (1 - ug))',
                             sense='min',
                             )


class UC(UCData, UCModel):
    """
    DC-based unit commitment (UC).
    """

    def __init__(self, system, config):
        UCData.__init__(self)
        UCModel.__init__(self, system, config)
