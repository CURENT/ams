"""
Module for optimization variables.
"""

from ams.core.var import RAlgeb

class OVar:
    """
    Base class for optimization variables.
    """
    def __init__(self):
        pass

    def add(self, RAlgeb=None, type=None, lb=None, ub=None, value=None):
        """
        Add variables.
        """
        # setattr(self, name, value)
        pass
