"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import (ZonalSum, VarReduction, NumOperation,
                              NumExpandDim, NumMultiply, NumAdd,
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
        # NOTE: Setting `ED.scale.owner` to `Horizon` will cause an error when calling `ED.scale.v`.
        # This is because `Horizon` is a group that only contains the model `TimeSlot`.
        # The `get` method of `Horizon` calls `andes.models.group.GroupBase.get` and results in an error.
        self.scale = RParam(info='zonal scaling factor for load',
                            name='scale',
                            src='scale',
                            tex_name=r's_{pd}',
                            model='TimeSlot')
        self.ts = RParam(info='time slot',
                         name='ts',
                         src='idx',
                         tex_name=r't_{s,idx}',
                         model='TimeSlot')
        self.zl = RParam(info='load zone',
                         name='zl',
                         src='zone',
                         tex_name='z_{one,l}',
                         model='PQ')
        self.zl1 = RParam(info='gen-bus load zone',
                          name='zl1',
                          tex_name=r'z_{one,l1}',
                          unit='p.u.')
        self.zl2 = RParam(info='non-gen-bus load zone',
                          name='zl2',
                          tex_name=r'z_{one,l2}',
                          unit='p.u.')
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

        # --- vars ---
        # NOTE: extend pg to 2D matrix, where row is gen and col is timeslot
        self.pg.horizon = self.timeslot
        self.pg.info = '2D power generation (system base, row for gen, col for horizon)'

        # --- service ---
        self.l1s = ZonalSum(u=self.zl1,
                            zone='Region',
                            name='l1s',
                            tex_name=r'\sum_{l,1}',
                            info='Sum pd1 vector in shape of zone',)
        self.l2s = ZonalSum(u=self.zl2,
                            zone='Region',
                            name='l2s',
                            tex_name=r'\sum_{l,2}',
                            info='Sum pd2 vector in shape of zone',)
        self.Spd1 = NumMultiply(u=self.l1s,
                                u2=self.pd1,
                                name='Spd1',
                                tex_name=r'S_{pd1,t}',
                                unit='p.u.',
                                rfun=np.sum,
                                rargs=dict(axis=1),
                                expand_dims=0,
                                info='Sum total load1 as row vector')
        self.Spd2 = NumMultiply(u=self.l2s,
                                u2=self.pd2,
                                name='Spd2',
                                tex_name=r'S_{pd2,t}',
                                unit='p.u.',
                                rfun=np.sum,
                                rargs=dict(axis=1),
                                expand_dims=0,
                                info='Sum total load2 as row vector')
        self.Spd = NumAdd(u=self.Spd1,
                          u2=self.Spd2,
                          name='Spd',
                          tex_name=r'S_{pd,t}',
                          unit='p.u.',
                          info='Sum total load as row vector')
        self.Spds = NumMultiply(u=self.scale,
                                u2=self.Spd,
                                name='Spds',
                                tex_name=r'S_{pd,t,s}',
                                unit='p.u.',
                                rfun=np.sum,
                                rargs=dict(axis=1),
                                expand_dims=0,
                                info='Scaled total load as row vector')
        self.Spg = VarReduction(u=self.pg,
                                fun=np.ones,
                                name='Spg',
                                tex_name=r'\sum_{pg}')
        self.Spg.info = 'Sum pg as row vector'

        # --- constraints ---
        # power balance
        # NOTE: Spg @ pg returns a row vector
        self.pb.e_str = 'Spds - Spg @ pg'  # power balance

        # line limits
        self.cdup = NumOperation(u=self.ts,
                                 fun=np.ones_like,
                                 name='cdup',
                                 tex_name=r'c_{dup}',
                                 expand_dims=0,
                                 info='row vector of 1s, to duplicate columns',
                                 dtype=int)
        self.RAr = NumExpandDim(u=self.rate_a,
                                name='RAr',
                                tex_name=r'R_{ATEA,c}',
                                info='rate_a as column vector',
                                axis=1)
        self.Spd1t = NumMultiply(u=self.pd1,
                                 u2=self.l1s,
                                 name='Spd1t',
                                 tex_name=r'S_{pd1}^{T}',
                                 unit='p.u.',
                                 rfun=np.transpose,
                                 info='Zonal total load1')
        self.Spd2t = NumMultiply(u=self.pd2,
                                 u2=self.l2s,
                                 name='Spd2t',
                                 tex_name=r'S_{pd2}^{T}',
                                 unit='p.u.',
                                 rfun=np.transpose,
                                 info='Zonal total load2')
        self.sT = NumOperation(u=self.scale,
                               fun=np.transpose,
                               name='sT',
                               tex_name=r's_{cale}^{T}',
                               unit='p.u.',
                               info='Zonal scale transpose',)
        # NOTE: PTDF1@pg returns a 2D matrix, row for line and col for horizon
        self.lub.e_str = 'PTDF1@pg - PTDF1@Spd1t@sT - PTDF2@Spd2t@sT - RAr@cdup'
        self.llb.e_str = '-PTDF1@pg + PTDF1@Spd1t@sT + PTDF2@Spd2t@sT - RAr@cdup'

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

# TODO: add data check
# if has model ``TimeSlot``, mandatory
# if has model ``Region``, optional
# if ``Region``, if ``Bus`` has param ``zone``, optional, if none, auto fill
