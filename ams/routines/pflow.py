"""
Module for power flow calculation.
"""

from ams.routines.base import BaseRoutine


class PF(BaseRoutine):
    """
    Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)

    def run(self, **kwargs):
        """
        Run power flow.
        """
        pass

    def summary(self, **kwargs):
        """
        Print power flow summary.
        """
        pass

    def report(self, **kwargs):
        """
        Print power flow report.
        """
        pass


class DCPF(BaseRoutine):
    """
    Power flow routine.
    """
    def __init__(self, system=None, config=None):
        super().__init__(system, config)

    def run(self, **kwargs):
        """
        Run power flow.
        """
        pass

    def summary(self, **kwargs):
        """
        Print power flow summary.
        """
        pass

    def report(self, **kwargs):
        """
        Print power flow report.
        """
        pass
