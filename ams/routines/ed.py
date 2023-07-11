"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import (ZonalSum, VarReduction, NumOperation,
                              NumExpandDim, NumMultiply, NumHstack,
                              VarSub)

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
        # NOTE: Setting `ED.scale.owner` to `Horizon` will cause an error when calling `ED.scale.v`.
        # This is because `Horizon` is a group that only contains the model `TimeSlot`.
        # The `get` method of `Horizon` calls `andes.models.group.GroupBase.get` and results in an error.
        self.scale = RParam(info='scaling factor for load',
                            name='scale',
                            src='scale',
                            tex_name=r's_{pd}',
                            model='TimeSlot')
        self.zl = RParam(info='load zone',
                         name='zl',
                         src='zone',
                         tex_name='z_{one,l}',
                         model='PQ',
                         )
        self.timeslot = RParam(info='Time slot for multi-period dispatch',
                               name='timeslot',
                               src='idx',
                               tex_name=r't_{s,idx}',
                               model='TimeSlot')

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

        for ele in ['lub', 'llb']:
            delattr(self, ele)

        # --- vars ---
        # NOTE: extend pg to 2D matrix, where row is gen and col is timeslot
        self.pg.horizon = self.timeslot
        self.pg.info = '2D power generation (system base, row for gen, col for horizon)'

        # --- service ---
        self.lsm = ZonalSum(u=self.zl,
                            zone='Region',
                            name='gsm',
                            tex_name=r'\sum_{g}')
        self.lsm.info = 'Sum load vector in shape of zone'
        self.Spdz = NumMultiply(u=self.lsm,
                                u2=self.pd,
                                name='Spdz',
                                tex_name=r'\sum_{pd,z}',
                                unit='p.u.',
                                rfun=np.sum,
                                rargs=dict(axis=1),
                                info='Zonal total load',
                                )
        self.Spd = NumMultiply(u=self.scale,
                               u2=self.Spdz,
                               name='Spd',
                               tex_name=r'S_{pd,t}',
                               unit='p.u.',
                               rfun=np.sum,
                               rargs=dict(axis=1),
                               expand_dims=0,
                               info='Scaled total load as row vector',
                               )
        self.Spg = VarReduction(u=self.pg,
                                fun=np.ones,
                                name='Spg',
                                tex_name=r'\sum_{pg}',)
        self.Spg.info = 'Sum pg as row vector'

        # --- constraints ---
        # power balance
        # NOTE: Spg @ pg returns a row vector
        self.pb.e_str = 'Spd - Spg @ pg'  # power balance

        # line limits
        # self.cdup0 = NumOperation(u=self.scale,
        #                           fun=np.ones_like,
        #                           name='cdup0',
        #                           tex_name=r'c_{dup}',
        #                           info='Column duplication array',
        #                           )
        # self.cdup = NumExpandDim(u=self.cdup0,
        #                          name='cdup',
        #                          tex_name=r'c_{dup,r}',
        #                          info='Column duplication array as row vector',
        #                          axis=0,
        #                          )
        # self.RAr = NumExpandDim(u=self.rate_a,
        #                         name='RAr',
        #                         tex_name=r'R_{ATEA,c}',
        #                         info='rate_a as column vector',
        #                         axis=1,
        #                         )
        # self.Rpd1 = NumExpandDim(u=self.pd1,
        #                          name='Rpd1',
        #                          tex_name=r'p_{d1,c}',
        #                          info='pd1 as column vector',
        #                          axis=1,
        #                          )
        # self.Rpd2 = NumExpandDim(u=self.pd2,
        #                          name='Rpd2',
        #                          tex_name=r'p_{d2,c}',
        #                          info='pd2 as column vector',
        #                          axis=1,
        #                          )
        # self.lub.e_str = 'PTDF1@pg - PTDF1*Rpd1@cdup - PTDF2@Rpd2@cdup - RAr@cdup'
        # self.llb.e_str = '-PTDF1@pg + PTDF1*Rpd1@cdup + PTDF2@Rpd2@cdup - RAr@cdup'

        # ramping limits
        # self.Mr = VarSub(u=self.pg,
        #                  horizon=self.timeslot,
        #                  name='Mr',
        #                  tex_name=r'M_{r}',
        #                  info='Subtraction matrix for ramping',
        #                  )
        # self.RR30 = NumHstack(u=self.R30,
        #                       ref=self.Mr,
        #                       name='RR30',
        #                       tex_name=r'R_{30,R}',
        #                       info='Repeated ramp rate',
        #                       )
        # self.rgu = Constraint(name='rgu',
        #                       info='generator ramping up',
        #                       e_str='pg @ Mr - RR30',
        #                       type='uq',
        #                       )
        # self.rgd = Constraint(name='rgd',
        #                       info='generator ramping down',
        #                       e_str='-pg @ Mr - RR30',
        #                       type='uq',
        #                       )

        # --- objective ---
        # NOTE: no need to fix objective function


class ED(EDData, EDModel):
    """
    DC-based multi-period economic dispatch (ED).

    ED extends DCOPF as follows:

    Power generation ``pg`` is extended to 2D matrix using argument
    ``horizon`` to represent the power generation of each generator in
    each time period (horizon).
    The rows correspond to generators and the columns correspond to time
    periods (horizons).

    Ramping limits ``rgu`` and ``rgd`` are introduced as 2D matrices to
    represent the upward and downward ramping limits for each generator.
    """

    def __init__(self, system, config):
        EDData.__init__(self)
        EDModel.__init__(self, system, config)
