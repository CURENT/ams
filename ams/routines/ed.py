"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import VarReduce, NumOperation, NumOneslike

from ams.routines.dcopf import DCOPFData, DCOPFModel

from ams.opt.omodel import Var, Constraint, Objective

logger = logging.getLogger(__name__)


class EDData(DCOPFData):
    """
    Economic dispatch data.
    """

    def __init__(self):
        DCOPFData.__init__(self)
        self.pd = RParam(info='active power load',
                         name='pd',
                         tex_name=r'p_{d}',
                         unit='p.u.',
                         src='p0',
                         model='PQ')
        self.scale = RParam(info='scaling factor for load',
                            name='scale',
                            src='scale',
                            tex_name=r's_{load}',
                            model='Horizon')

        self.R10 = RParam(info='10-min ramp rate (system base)',
                          name='R10',
                          src='R10',
                          tex_name=r'R_{10}',
                          unit='p.u./min',
                          model='StaticGen')


class EDModel(DCOPFModel):
    """
    Economic dispatch model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)
        # DEBUG: clear constraints and objective
        for name in ['lub', 'llb']:
            delattr(self, name)

        self.info = 'Economic dispatch'
        self.type = 'DCED'
        # --- vars ---
        # self.pg.horizon = self.nt

        # --- service ---
        # self.pdt = NumOperation(u=self.pd,
        #                         fun=np.sum,
        #                         name='pdt',
        #                         tex_name='p_{d, total}',
        #                         unit='p.u.',
        #                         info='Total load',
        #                         )
        # self.pdr = NumOneslike(u=self.pdt,
        #                        ref=self.pg,
        #                        name='pdr',
        #                        tex_name='p_{d, repeat}',
        #                        unit='p.u.',
        #                        info='Repeated total load',
        #                        )
        # self.pgs = VarReduce(u=self.pg,
        #                      fun=np.ones,
        #                      name='pgs',
        #                      tex_name='\sum_{p,g}',)
        # self.pgs.info = 'Sum matrix to reduce pg axis0 to 1'
        # # --- constraints ---
        # self.pb.e_str = 'pgs @ pg'
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
