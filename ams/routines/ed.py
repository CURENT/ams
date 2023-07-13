"""
Real-time economic dispatch.
"""
import logging  # NOQA
from collections import OrderedDict  # NOQA
import numpy as np  # NOQA

from ams.core.param import RParam
from ams.core.service import (ZonalSum, VarReduction, NumOp,
                              NumOpDual, NumExpandDim, NumHstack,
                              RampSub)

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
        self.scale = RParam(info='zonal load factor for ED',
                            name='scale',
                            src='scale',
                            tex_name=r's_{pd}',
                            model='EDTSlot')
        self.ts = RParam(info='time slot',
                         name='ts',
                         src='idx',
                         tex_name=r't_{s,idx}',
                         model='EDTSlot')
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
        self.timeslot = RParam(info='Time slot for multi-period ED',
                               name='timeslot',
                               src='idx',
                               tex_name=r't_{s,idx}',
                               model='EDTSlot')

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
        self.pd1z = NumOpDual(u=self.l1s,
                              u2=self.pd1,
                              fun=np.multiply,
                              name='Spd1',
                              tex_name=r'p_{d1,z}',
                              unit='p.u.',
                              info='zonal load1')
        self.pd2z = NumOpDual(u=self.l2s,
                              u2=self.pd2,
                              fun=np.multiply,
                              name='Spd2',
                              tex_name=r'p_{d2,z}',
                              unit='p.u.',
                              info='zonal load2')
        self.pdz = NumOp(u=[self.pd1z, self.pd2z],
                         fun=np.hstack,
                         rfun=np.sum,
                         rargs=dict(axis=1),
                         expand_dims=0,
                         name='pdz',
                         tex_name=r'p_{d,z}',
                         unit='p.u.',
                         info='zonal load as row vector')
        self.pds = NumOpDual(u=self.scale,
                             u2=self.pdz,
                             fun=np.multiply,
                             rfun=np.sum,
                             rargs=dict(axis=1),
                             expand_dims=0,
                             name='pds',
                             tex_name=r'p_{d,s,t}',
                             unit='p.u.',
                             info='Scaled total load as row vector')
        self.Spg = VarReduction(u=self.pg,
                                fun=np.ones,
                                name='Spg',
                                tex_name=r'\sum_{pg}',
                                info='Sum pg as row vector')

        # --- constraints ---
        # power balance
        # NOTE: Spg @ pg returns a row vector
        self.pb.e_str = 'pds - Spg @ pg'  # power balance
        # line limits
        self.cdup = NumOp(u=self.ts,
                          fun=np.ones_like,
                          args=dict(dtype=int),
                          name='cdup',
                          tex_name=r'c_{dup}',
                          expand_dims=0,
                          info='row vector of 1s to duplicate column vector')
        self.RAr = NumExpandDim(u=self.rate_a,
                                axis=1,
                                name='RAr',
                                tex_name=r'R_{ATEA,c}',
                                info='rate_a as column vector',)
        self.pd1s = NumOpDual(u=self.scale,
                              u2=self.pd1z,
                              fun=np.matmul,
                              rfun=np.transpose,
                              name='pd1s',
                              tex_name=r'p_{d1,s,t}',
                              unit='p.u.',
                              info='Scaled load1')
        self.pd2s = NumOpDual(u=self.scale,
                              u2=self.pd2z,
                              fun=np.matmul,
                              rfun=np.transpose,
                              name='pd2s',
                              tex_name=r'p_{d2,s,t}',
                              unit='p.u.',
                              info='Scaled load2')
        # NOTE: PTDF1@pg returns a 2D matrix, row for line and col for horizon
        self.lub.e_str = 'PTDF1@pg - PTDF1@pd1s - PTDF2@pd2s - RAr@cdup'
        self.llb.e_str = '-PTDF1@pg + PTDF1@pd1s + PTDF2@pd2s - RAr@cdup'

        # ramping limits
        # FIXME: do we need to consider ramping limits for T0?
        # which means from p0 (initial power) to pg(:, 0)?
        self.Mr = RampSub(u=self.pg,
                          name='Mr',
                          tex_name=r'M_{r}',
                          info='Subtraction matrix for ramping, (ng, ng-1)',
                          )
        self.RR30 = NumHstack(u=self.R30,
                              ref=self.Mr,
                              name='RR30',
                              tex_name=r'R_{30,R}',
                              info='Repeated ramp rate as 2D matrix, (ng, ng-1)',
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
