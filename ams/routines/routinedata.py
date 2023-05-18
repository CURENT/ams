"""
Module for routine data.
"""

import logging  # NOQA
from collections import OrderedDict  # NOQA

import numpy as np

from andes.core import Config
from andes.shared import deg2rad
from andes.utils.misc import elapsed
from ams.core.param import RParam
from ams.core.var import RAlgeb
from ams.opt.omodel import OModel, Var, Constraint

from ams.core.symprocessor import SymProcessor

logger = logging.getLogger(__name__)


class RoutineData:
    """
    CLass to hold routine parameters and variables for a dispatch model.
    """

    def __init__(self):
        self.rparams = OrderedDict()  # list out RParam in a routine
