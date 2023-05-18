"""
Module for routine data.
"""

import logging  # NOQA
from collections import OrderedDict  # NOQA

import numpy as np

from andes.core import Config
from andes.shared import deg2rad
from andes.utils.misc import elapsed
from ams.opt.omodel import OModel

from ams.core.symprocessor import SymProcessor

logger = logging.getLogger(__name__)


class RoutineData:
    """
    CLass to hold routine parameters and variables for a dispatch model.
    """

    def __init__(self):
        pass
