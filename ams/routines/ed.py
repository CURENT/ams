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
        self.nt = RParam(info='Number of rolling intervals',
                         name='nt',
                         src='nt',
                         tex_name=r'n_{t}',
                         owner_name='RolHorizon',
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
        # --- constraints ---
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
                             e_str='sum(c2 * pg**2 + c1 * pg + c0)',
                             sense='min',
                             )


class ED(EDData, EDModel):
    """
    DC-based economic dispatch (ED).

    ED extends DCOPF with:

    1. zonal SFR reserve: decision variables ``pru`` and ``prd``; linear cost ``cru`` and ``crd``; requirement ``du`` and ``dd``

    2. generator ramping: start point ``pg0``; ramping limit ``R10``

    Notes
    -----
    1. ED is a DC-based model, which assumes bus voltage to be 1
    2. ED adds a data check on ``pg0``, if all zeros, correct as the value of ``p0``
    """

    def __init__(self, system, config):
        EDData.__init__(self)
        EDModel.__init__(self, system, config)
