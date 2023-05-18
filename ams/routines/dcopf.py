"""
OPF routines.
"""
from collections import OrderedDict
from ams.routines.base import BaseRoutine, timer
import numpy as np
from scipy.optimize import linprog

from ams.routines.routinedata import RoutineData
from ams.routines.routine import Routine


class DCOPFData(RoutineData):
    """
    DCOPF parameters and variables.
    """

    def __init__(self):
        RoutineData.__init__(self)
        self.aBus = None
        self.vBus = None


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
