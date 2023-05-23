import logging
import importlib
import inspect
import copy
from collections import OrderedDict

import numpy as np

from andes.models.group import GroupBase as andes_GroupBase

from ams.core.var import Algeb

logger = logging.getLogger(__name__)


class TypeBase:
    """
    Base class for types.
    """

    def __init__(self):

        self.common_rparams = ['u', 'name']
        self.common_ralgebs = []

        self.routines = OrderedDict()

    @property
    def class_name(self):
        return self.__class__.__name__

class Undefined(TypeBase):
    """
    The undefined type. Holds routines with no ``type``.
    """
    pass


class PowerFlow(TypeBase):
    """
    Power flow type. Holds routines with ``type`` of ``pflow``.
    """

    def __init__(self):
        super().__init__()
