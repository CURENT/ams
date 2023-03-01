"""
Module for power flow calculation.
"""

from ams.routines.base import BaseRoutine
from ams.solver.pypower.runpf import runpf, rundcpf


class PF(BaseRoutine):
    """
    Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.converged = False

    def run(self, **kwargs):
        """
        Run power flow.
        """
        res, exit_code = runpf(self.system._ppc, **kwargs)
        self.converged = bool(exit_code)

        # TODO: organize the results
        # bus, gen, line

        ppc = self.system._ppc
        return self.converged, ppc

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
