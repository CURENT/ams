"""
Distributional optimal power flow (DOPF) routine.
"""
from collections import OrderedDict
import numpy as np
from scipy.optimize import linprog

from ams.core.param import RParam

from ams.routines.routine import RoutineData, RoutineModel
from ams.routines.dcopf import DCOPFData, DCOPFBase, DCOPFModel

from ams.opt.omodel import Var, Constraint, Objective


class DOPFData(DCOPFData):
    """
    DOPF data.
    """

    def __init__(self):
        DCOPFData.__init__(self)


class DOPFBase(DCOPFBase):
    """
    Base class for DOPF dispatch model.

    Overload the ``solve``, ``unpack``, and ``run`` methods.
    """

    def __init__(self, system, config):
        DCOPFBase.__init__(self, system, config)


class DOPFModel(DOPFBase, DCOPFModel):
    """
    DOPF model.
    """

    def __init__(self, system, config):
        DOPFBase.__init__(self, system, config)
        DCOPFModel.__init__(self, system, config)
        self.info = 'Distributional Optimal Power Flow'
        self.type = 'DED'


class DOPF(DOPFData, DOPFModel):
    """
    Distributional optimal power flow (DOPF) routine.
    """

    def __init__(self, system, config):
        DOPFData.__init__(self)
        DOPFModel.__init__(self, system, config)
