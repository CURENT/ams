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
from ams.opt.omodel import OModel

from ams.core.symprocessor import SymProcessor

logger = logging.getLogger(__name__)


class RoutineData:
    """
    CLass to hold routine parameters and variables for a dispatch model.
    """

    def __init__(self):
        self.rparams = OrderedDict()  # list out RParam in a routine
        self.ralgebs = OrderedDict()  # list out RAlgebs in a routine

    def __setattr__(self, key, value):
        """
        Overload the setattr function to register attributes.

        Parameters
        ----------
        key : str
            name of the attribute
        value : [Algeb]
            value of the attribute
        """

        # NOTE: value.id is not in use yet
        if isinstance(value, RAlgeb):
            value.id = len(self.ralgebs)
        elif isinstance(value, RParam):
            value.id = len(self.rparams)
        self._register_attribute(key, value)

        super(RoutineData, self).__setattr__(key, value)

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to the model instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        Block attributes will be exported and registered recursively.
        """
        if isinstance(value, RAlgeb):
            self.ralgebs[key] = value
            logger.debug('Registering algeb %s', key)
        elif isinstance(value, RParam):
            self.rparams[key] = value
            logger.debug('Registering parameter %s', key)
            value.id = len(self.rparams)
