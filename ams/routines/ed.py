"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import VarReduce, NumOperation, NumMultiply

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
                            tex_name=r's_{pd}',
                            model='Horizon')
        self.horizon = RParam(info='horizon idx',
                              name='horizon',
                              src='idx',
                              tex_name=r'h_{idx}',
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
        # NOTE: extend pg to 2D matrix, where row is gen and col is horizon
        self.pg.horizon = self.horizon
        self.pg.info = '2D power generation (system base, row for gen, col for horizon)'

        # --- service ---
        self.spd = NumOperation(u=self.pd,
                                fun=np.sum,
                                name='spd',
                                tex_name=r'\sum_{p,d}',
                                unit='p.u.',
                                info='Total load in shape(1)',
                                )
        self.sr = NumOperation(u=self.scale,
                               fun=np.expand_dims,
                               name='sr',
                               tex_name='s_{pd, r}',
                               unit='p.u.',
                               info='Scale in shape (1, nh)',
                               axis=0,
                               )
        self.pdr = NumMultiply(u=self.spd,
                               u2=self.sr,
                               name='pdr',
                               tex_name='p_{d, r}',
                               unit='p.u.',
                               info='Scaled total load in shape (1, nh)',
                               )
        self.pgs = VarReduce(u=self.pg,
                             fun=np.ones,
                             name='pgs',
                             tex_name='\sum_{p,g}',)
        self.pgs.info = 'Sum matrix to reduce pg axis0 to 1'
        # NOTE: pgs @ pg will return size (1, nh), where nh is the number of
        #       horizon intervals
        # # --- constraints ---
        self.pb.e_str = 'pdr - pgs @ pg'
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

    1. rolling horizon using model ``Horizon``

    2. generator ramping limits ``rgu`` and ``rgd``
    """

    def __init__(self, system, config):
        EDData.__init__(self)
        EDModel.__init__(self, system, config)
