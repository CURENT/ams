"""
Module for power flow calculation.
"""

import numpy as np

from ams.routines.base import BaseRoutine
from ams.solver.pypower.runpf import runpf, rundcpf

from ams.io.pypower import system2ppc, ppc2system


class PFlow(BaseRoutine):
    """
    Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "AC Power flow"
        self.converged = False

    def run(self, **kwargs):
        """
        Run power flow.
        """
        ppc = system2ppc(self.system)
        res, exit_code = runpf(ppc, **kwargs)
        self.converged = bool(exit_code)

        # # --- Update variables ---
        # a_Bus, v_Bus, q_PV, p_Slack, q_Slack = res2system(self.system, res)
        # for aname in self.algebs.keys():
        #     a = self.algebs[aname]['a']  # array index
        #     if len(a) > 1:
        #         self.v[a[0]:a[-1]+1] = locals()[aname]
        #         self.algebs[aname]['algeb'].v = locals()[aname]
        #     else:
        #         self.v[a] = locals()[aname]
        #         self.algebs[aname]['algeb'].v = locals()[aname]

        return res, self.converged

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
