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
                         is_group=False,
                         )
        self.c1 = RParam(info='Gen cost coefficient 1',
                         name='c1',
                         tex_name=r'c_{1}',
                         unit=r'$/MW (MVar)',
                         is_group=False,
                         )
        self.c0 = RParam(info='Gen cost coefficient 0',
                         name='c0',
                         tex_name=r'c_{0}',
                         unit=r'$/MW (MVar)',
                         is_group=False,
                         )
        # --- generator output ---
        self.p = RAlgeb(info='actual active power generation',
                        unit='p.u.',
                        name='p',
                        tex_name=r'p_{g}',
                        is_group=True,
                        group_name='StaticGen',
                        )
        # --- load ---
        self.pd = RParam(info='active power load in system base',
                         name='pd',
                         tex_name=r'p_{d}',
                         unit='p.u.',
                         )
        # --- line ---
        self.rate_a = RParam(info='long-term flow limit flow limit',
                             name='rate_a',
                             tex_name=r'R_{ATEA}',
                             unit='MVA')


class DCOPFModel(Routine):
    """
    DCOPF dispatch model.
    """

    def __init__(self, system, config):
        Routine.__init__(self, system, config)


class DCOPF(DCOPFData, DCOPFModel):
    """
    DCOPF dispatch routine.
    """

    def __init__(self, system, config):
        DCOPFData.__init__(self)
        DCOPFModel.__init__(self, system, config)
