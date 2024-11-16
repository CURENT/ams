"""
Module for non-linear equations.
"""

from ams.routines.pflow2 import PFlow2


class PFlowSolver:
    """
    Base class for power flow solver.
    """

    def __init__(self, rtn: PFlow2):
        """
        Initialize the non-linear equations with an instance of PFlow2.
        """
        self.rtn = rtn

    def init(self):
        """
        Initialize the power flow solver.
        """
        raise NotImplementedError

    def parse(self):
        """
        Parse the power flow equations.
        """
        raise NotImplementedError

    def evaluate(self):
        """
        Evaluate the power flow equations.
        """
        raise NotImplementedError
