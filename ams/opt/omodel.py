"""
Module for optimization models.
"""

import logging

from collections import OrderedDict

from andes.core.common import Config

from ams.opt.ovar import OVar
from ams.opt.constraint import Constraint
from ams.opt.objective import Objective

logger = logging.getLogger(__name__)


class OModel:
    """
    Base class for optimization models.
    """

    def __init__(self):
        self.vars = OrderedDict()
        self.constraints = OrderedDict()
        self.objective = None

    def init(self):
        pass

    def add_vars(self, name='var', type=None, n=1, lb=None, ub=None):
        var = OVar(name=name, type=type, n=n, lb=lb, ub=ub)
        self.vars[name] = var
        return var

    def add_constraints(self, *args, **kwargs):
        self.constraints.add(*args, **kwargs)

    def add_objective(self, *args, **kwargs):
        self.objectives.add(*args, **kwargs)
