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
        self.pg = RParam()


class DCOPFModel(Routine):
    """
    DCOPF dispatch model.
    """

    def __init__(self, system, config):
        Routine.__init__(self, system, config)
        self.p = RAlgeb(info='actual active power generation',
                        unit='p.u.',
                        tex_name='p',
                        name='p',
                        is_group = True,
                        group_name='StaticGen',
                        )
        self.q = RAlgeb(info='actual reactive power generation',
                        unit='p.u.',
                        tex_name='q',
                        name='q',
                        is_group = True,
                        group_name='StaticGen',
                        )


class DCOPF(DCOPFData, DCOPFModel):
    """
    DCOPF dispatch routine.
    """

    def __init__(self, system, config):
        DCOPFData.__init__(self)
        DCOPFModel.__init__(self, system, config)
