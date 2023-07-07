"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import (VarReduction, NumOperation, NumMultiply,
                              NumHstack, VarSub)

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

        self.R30 = RParam(info='30-min ramp rate (system base)',
                          name='R30',
                          src='R30',
                          tex_name=r'R_{30}',
                          unit='p.u./min',
                          model='StaticGen')


class EDModel(DCOPFModel):
    """
    Economic dispatch model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)

        self.info = 'Economic dispatch'
        self.type = 'DCED'

        # --- vars ---
        # NOTE: extend pg to 2D matrix, where row is gen and col is horizon
        self.pg.horizon = self.horizon
        self.pg.info = '2D power generation (system base, row for gen, col for horizon)'

        # --- service ---
        self.Spd = NumOperation(u=self.pd,
                                fun=np.sum,
                                name='Spd',
                                tex_name=r'\sum_{pd}',
                                unit='p.u.',
                                info='Total load in shape (1, )',
                                )
        self.Sr = NumOperation(u=self.scale,
                               fun=np.expand_dims,
                               name='Sr',
                               tex_name=r'S_{pd}',
                               unit='p.u.',
                               info='Scale in shape (1, nh)',
                               axis=0,
                               )
        self.Rpd = NumMultiply(u=self.Spd,
                               u2=self.Sr,
                               name='Rpd',
                               tex_name=r'R_{pd}',
                               unit='p.u.',
                               info='Scaled total load in shape (1, nh)',
                               )
        self.Spg = VarReduction(u=self.pg,
                                fun=np.ones,
                                name='spg',
                                tex_name=r'\sum_{pg}',)
        self.Spg.info = 'Sum matrix to reduce pg axis0 to 1'

        # --- constraints ---
        # power balance
        # NOTE: pgs @ pg will return size (1, nh), where nh is the number of
        #       horizon intervals
        self.pb.e_str = 'Rpd - Spg @ pg'  # power balance

        # line limits
        self.cdup0 = NumOperation(u=self.scale,
                                  fun=np.ones_like,
                                  name='cdup0',
                                  tex_name=r'c_{dup,0}',
                                  info='Column duplication vector',
                                  )
        self.cdup = NumOperation(u=self.cdup0,
                                 fun=np.expand_dims,
                                 name='cdup',
                                 tex_name=r'c_{dup}',
                                 info='Column duplication vector in shape (1, nh)',
                                 axis=0,
                                 )
        self.RAr = NumOperation(u=self.rate_a,
                                fun=np.expand_dims,
                                name='RAr',
                                tex_name=r'R_{ATEA,r}',
                                unit='p.u.',
                                info='rate_a in shape (nh, 1)',
                                axis=1,
                                )
        self.Rpd1 = NumOperation(u=self.pd1,
                                 fun=np.expand_dims,
                                 name='Rpd1',
                                 tex_name=r'p_{d1,r}',
                                 unit='p.u.',
                                 info='pd1 in shape (nh, 1)',
                                 axis=1,
                                 )
        self.Rpd2 = NumOperation(u=self.pd2,
                                 fun=np.expand_dims,
                                 name='Rpd2',
                                 tex_name=r'p_{d2,r}',
                                 unit='p.u.',
                                 info='pd2 in shape (nh, 1)',
                                 axis=1,
                                 )
        self.lub.e_str = 'PTDF1@pg - PTDF1*Rpd1@cdup - PTDF2@Rpd2@cdup - RAr@cdup'
        self.llb.e_str = '-PTDF1@pg + PTDF1*Rpd1@cdup + PTDF2@Rpd2@cdup - RAr@cdup'

        # ramping limits
        self.Mr = VarSub(u=self.pg,
                         horizon=self.horizon,
                         name='Mr',
                         tex_name=r'M_{r}',
                         info='Subtraction matrix for ramping',
                         )
        self.RR30 = NumHstack(u=self.R30,
                              ref=self.Mr,
                              name='RR30',
                              tex_name=r'R_{R30}',
                              info='Repeated ramp rate',
                              )
        self.rgu = Constraint(name='rgu',
                              info='generator ramping up',
                              e_str='pg @ Mr - RR30',
                              type='uq',
                              )
        self.rgd = Constraint(name='rgd',
                              info='generator ramping down',
                              e_str='-pg @ Mr - RR30',
                              type='uq',
                              )


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
