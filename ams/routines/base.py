"""
Base module for routines.
"""

import logging # NOQA
from collections import OrderedDict  # NOQA
from andes.routines.base import BaseRoutine as andes_BaseRoutine  # NOQA

logger = logging.getLogger(__name__)


class BaseTable:
    """
    Base class for holding dispatch results of a device.
    """
    def __init__(self, columns=None):
        self.columns = columns
        self.data = OrderedDict()
    
    def as_df(self):
        """
        Convert the data to a pandas DataFrame.
        """
        pass


class BaseResults:
    """
    Base class for holding dispatch results.
    """
    def __init__(self):
        pass

    def as_df(self):
        """
        Convert the data to a pandas DataFrame.
        """
        pass


class BaseRoutine(andes_BaseRoutine):
    """
    Base routine class.
    """
    def __init__(self, system=None, config=None):
        # TODO: may need revision on the __init__()
        super().__init__()
        self.name = 'BaseRoutine'
        # TODO: how to organize the results?
        self.res = BaseResults()

    def run(self, **kwargs):
        """
        Run the routine.
        """
        pass
