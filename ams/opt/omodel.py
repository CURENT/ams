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

    def __init__(self, system=None, config=None):

        # --- Model ---
        self.system = system
        # TODO: check config
        self.config = Config(name=self.class_name)  # `config` that can be exported

        self.vars = OVar()
        self.constraints = Constraint()
        self.objectives = Objective()

    def add_constraints(self, *args, **kwargs):
        self.constraints.add(*args, **kwargs)
