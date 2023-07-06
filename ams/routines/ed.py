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
                         model='Horizon',
                         )

        self.R10 = RParam(info='10-min ramp rate (system base)',
                          name='R10',
                          src='R10',
                          tex_name=r'R_{10}',
                          unit='p.u./min',
                          model='StaticGen',
                          )


class EDModel(DCOPFModel):
    """
    Economic dispatch model.
    """

    def __init__(self, system, config):
        super().__init__(system, config)
        # DEBUG: clear constraints
        self.pb = None
        self.lub = None
        self.llb = None
        self.constrs.clear()
        # DEBUG: clear objective
        self.obj = None

        self.info = 'Economic dispatch'
        self.type = 'DCED'
        # --- vars ---
        self.pg.horizon = self.nt
        self.pg.lb = None
        self.pg.ub = None

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
        # # --- objective ---
        # self.obj = Objective(name='tc',
        #                      info='total generation and reserve cost',
        #                      e_str='sum(c2 * pg**2 + c1 * pg + c0)',
        #                      sense='min',
        #                      )


class ED(EDData, EDModel):
    """
    DC-based economic dispatch (ED).

    ED extends DCOPF with:

    1. rolling horizon: ``nt`` intervals

    2. generator ramping limits ``rgu`` and ``rgd``

    Notes
    -----
    1. ED is a DC-based model, which assumes bus voltage to be 1
    """

    def __init__(self, system, config):
        EDData.__init__(self)
        EDModel.__init__(self, system, config)
