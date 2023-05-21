"""
OPF routines.
"""
from collections import OrderedDict
from ams.routines.base import BaseRoutine, timer
import numpy as np
from scipy.optimize import linprog

from ams.core.param import RParam
from ams.core.var import RAlgeb

from ams.routines.routinedata import RoutineData
from ams.routines.routine import Routine

from ams.opt.omodel import Constraint, Objective


class DCOPFData(RoutineData):
    """
    DCOPF parameters and variables.
    """

    def __init__(self):
        RoutineData.__init__(self)
        # --- generator cost ---
        self.c2 = RParam(info='Gen cost coefficient 2',
                         name='c2',
                         tex_name=r'c_{2}',
                         unit=r'$/MW (MVar)',
                         owner_name='GCost',
                         )
        self.c1 = RParam(info='Gen cost coefficient 1',
                         name='c1',
                         tex_name=r'c_{1}',
                         unit=r'$/MW (MVar)',
                         owner_name='GCost',
                         )
        self.c0 = RParam(info='Gen cost coefficient 0',
                         name='c0',
                         tex_name=r'c_{0}',
                         unit=r'$/MW (MVar)',
                         owner_name='GCost',
                         )
        # --- generator output ---
        self.pmax = RParam(info='generator maximum active power in system base',
                           name='pmax',
                           tex_name=r'p_{max}',
                           unit='p.u.',
                           owner_name='StaticGen',
                           )
        self.pmin = RParam(info='generator minimum active power in system base',
                           name='pmin',
                           tex_name=r'p_{min}',
                           unit='p.u.',
                           owner_name='StaticGen',
                           )

        # --- load ---
        self.pd = RParam(info='active power load in system base',
                         name='pd',
                         src='p0',
                         tex_name=r'p_{d}',
                         unit='p.u.',
                         owner_name='PQ',
                         )
        # --- line ---
        self.rate_a = RParam(info='long-term flow limit flow limit',
                             name='rate_a',
                             tex_name=r'R_{ATEA}',
                             unit='MVA',
                             owner_name='Line',
                             )


class DCOPFModel(Routine):
    """
    DCOPF dispatch model.
    """

    def __init__(self, system, config):
        Routine.__init__(self, system, config)
        self.pg = RAlgeb(info='actual active power generation',
                         unit='p.u.',
                         name='pg',
                         tex_name=r'p_{g}',
                         owner_name='StaticGen',
                         lb=self.pmin,
                         ub=self.pmax,
                         )

        # --- constraints ---
        self.pb = Constraint(name='pb',
                             info='power balance',
                             e_str='sum(pd) - sum(pg)',
                             type='eq',
                             )
        # self.lub = Constraint(name='lub',
        #                       info='line limits upper bound',
        #                       e_str='GSF * (G - D) - rate_aLine',
        #                       type='uq',
        #                       )
        # self.llb = Constraint(name='llb',
        #                       info='line limits lower bound',
        #                       e_str='- GSF * (G - D) - rate_aLine',
        #                       type='uq',
        #                       )

        # --- objective ---
        self.obj = Objective(e_str='sum(c2 * pg**2 + c1 * pg + c0)',
                             sense='min',)


class DCOPF(DCOPFData, DCOPFModel):
    """
    DCOPF dispatch routine.
    """

    def __init__(self, system, config):
        DCOPFData.__init__(self)
        DCOPFModel.__init__(self, system, config)
