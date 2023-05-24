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

        self.common_rparams = []
        self.common_ralgebs = []
        self.common_constrs = []

        self.routines = OrderedDict()

    @property
    def class_name(self):
        return self.__class__.__name__

    @property
    def n(self):
        """
        Total number of devices.
        """
        return len(self.routines)

    def __repr__(self):
        dev_text = 'routine' if self.n == 1 else 'routines'
        return f'{self.class_name} ({self.n} {dev_text}) at {hex(id(self))}'


class Undefined(TypeBase):
    """
    The undefined type.
    """
    pass


class PF(TypeBase):
    """
    Type for power flow routines.
    """

    def __init__(self):
        super().__init__()
        self.common_rparams.extend(('pd',))
        self.common_ralgebs.extend(('pg',))


class DC(TypeBase):
    """
    Type for DCOPF routines.
    """

    def __init__(self):
        super().__init__()
        self.common_rparams.extend(('c2', 'c1', 'c0', 'pmax', 'pmin', 'pd', 'rate_a',))
        self.common_ralgebs.extend(('pg',))
        self.common_constrs.extend(('pb', 'lub', 'llb'))


class AC(DC):
    """
    Type for ACOPF routines.
    """

    def __init__(self):
        DC.__init__(self)
        self.common_rparams.extend(('qd',))
        self.common_ralgebs.extend(('aBus', 'vBus', 'qg',))


class ED(TypeBase):
    """
    Type for economic dispatch routines.
    """

    def __init__(self):
        super().__init__()
        # TODO: add common parameters and variables


class UC(TypeBase):
    """
    Type for unit commitment routines.
    """

    def __init__(self):
        super().__init__()
        # TODO: add common parameters and variables
