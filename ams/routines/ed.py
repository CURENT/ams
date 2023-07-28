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
        self.pg0 = RParam(info='active power start point (system base)',
                          name='pg0', tex_name=r'p_{g0}',
                          unit='p.u.', src='pg0',
                          model='StaticGen')
        # NOTE: Setting `ED.scale.owner` to `Horizon` will cause an error when calling `ED.scale.v`.
        # This is because `Horizon` is a group that only contains the model `TimeSlot`.
        # The `get` method of `Horizon` calls `andes.models.group.GroupBase.get` and results in an error.
        self.scale = RParam(info='zonal load factor for ED',
                            name='scale', tex_name=r's_{pd}',
                            src='scale', model='EDTSlot')
        self.ts = RParam(info='time slot',
                         name='ts', tex_name=r't_{s,idx}',
                         src='idx', model='EDTSlot')
        self.zb = RParam(info='Bus zone',
                         name='zb', tex_name='z_{one,bus}',
                         src='zone', model='Bus')
        self.zl = RParam(info='load zone',
                         name='zl', tex_name='z_{one,l}',
                         src='zone', model='PQ')
        self.timeslot = RParam(info='Time slot for multi-period ED',
                               name='timeslot', tex_name=r't_{s,idx}',
                               src='idx', model='EDTSlot')
        self.R30 = RParam(info='30-min ramp rate (system base)',
                          name='R30', tex_name=r'R_{30}',
                          src='R30', unit='p.u./min',
                          model='StaticGen')


class EDModel(DCOPFModel):
    """
    Economic dispatch model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)
        # delattr(self, 'lub')  # TODO: debug
        # delattr(self, 'llb')  # TODO: debug

        self.info = 'Economic dispatch'
        self.type = 'DCED'

        # --- vars ---
        # NOTE: extend pg to 2D matrix, where row is gen and col is timeslot
        self.pg.horizon = self.timeslot
        self.pg.info = '2D power generation (system base, row for gen, col for horizon)'

        self.pn.horizon = self.timeslot
        self.pn.info = '2D Bus power injection (system base, row for bus, col for horizon)'

        # --- constraints ---
        # --- power balance ---
        self.ls = ZonalSum(u=self.zb, zone='Region',
                           name='ls', tex_name=r'\sum_{l}',
                           info='Sum pd vector in shape of zone',)
        self.pdz = NumOpDual(u=self.ls, u2=self.pd,
                             fun=np.multiply,
                             rfun=np.sum, rargs=dict(axis=1),
                             expand_dims=0,
                             name='pdz', tex_name=r'p_{d,z}',
                             unit='p.u.', info='zonal load')
        self.pds = NumOpDual(u=self.scale, u2=self.pdz,
                             fun=np.multiply, rfun=np.sum,
                             rargs=dict(axis=1), expand_dims=0,
                             name='pds', tex_name=r'p_{d,s,t}',
                             unit='p.u.', info='Scaled total load as row vector')
        self.Spg = VarReduction(u=self.pg, fun=np.ones,
                                name='Spg', tex_name=r'\sum_{pg}',
                                info='Sum pg as row vector')
        # NOTE: Spg @ pg returns a row vector
        self.pb.e_str = 'pds - Spg @ pg'  # power balance

        # --- bus power injection ---
        self.pdR = NumHstack(u=self.pd, ref=self.timeslot,
                             name='pdR', tex_name=r'p_{d,R}',
                             info='Repeated power demand as 2D matrix',)
        self.pinj.e_str = 'Cg @ (pn - pdR) - pg'  # power injection

        # --- line limits ---
        self.cdup = NumOp(u=self.ts,
                          fun=np.ones_like, args=dict(dtype=int),
                          expand_dims=0,
                          name='cdup', tex_name=r'c_{dup}',
                          info='row vector of 1s to duplicate column vector')
        self.RAr = NumExpandDim(u=self.rate_a, axis=1,
                                name='RAr', tex_name=r'R_{ATEA,c}',
                                info='rate_a as column vector',)
        self.lub.e_str = 'PTDF @ (pn - pdR) - RAr@cdup'
        self.llb.e_str = '-PTDF @ (pn - pdR) - RAr@cdup'

        # --- ramping ---
        self.Mr = RampSub(u=self.pg, name='Mr', tex_name=r'M_{r}',
                          info='Subtraction matrix for ramping, (ng, ng-1)',)
        self.RR30 = NumHstack(u=self.R30, ref=self.Mr,
                              name='RR30', tex_name=r'R_{30,R}',
                              info='Repeated ramp rate as 2D matrix, (ng, ng-1)',)
        self.rgu = Constraint(name='rgu', info='Gen ramping up',
                              e_str='pg @ Mr - RR30',
                              type='uq',)
        self.rgd = Constraint(name='rgd',
                              info='Gen ramping down',
                              e_str='-pg @ Mr - RR30',
                              type='uq',)
        self.rgu0 = Constraint(name='rgu0',
                               info='Initial gen ramping up',
                               e_str='pg[:, 0] - pg0 - R30',
                               type='uq',)
        self.rgd0 = Constraint(name='rgd0',
                               info='Initial gen ramping down',
                               e_str='- pg[:, 0] + pg0 - R30',
                               type='uq',)

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
