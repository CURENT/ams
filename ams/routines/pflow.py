"""
Power flow routines.
"""

import numpy as np

from andes.shared import deg2rad

from ams.routines.base import BaseRoutine
from ams.solver.pypower.runpf import runpf, rundcpf

from ams.io.pypower import system2ppc


class PFlow(BaseRoutine):
    """
    AC Power flow routine.
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

        # --- Update variables ---
        system = self.system

        # --- Bus ---
        system.Bus.v.v = a_Bus = ppc['bus'][:, 7]  # voltage magnitude
        system.Bus.a.v = v_Bus = ppc['bus'][:, 8] * deg2rad # voltage angle

        # --- PV ---
        system.PV.q.v = q_PV = ppc['gen'][system.Slack.n:, 2]  # reactive power

        # --- Slack ---
        system.Slack.p.v = p_Slack = ppc['gen'][:system.Slack.n, 1]  # active power
        system.Slack.q.v = q_Slack = ppc['gen'][:system.Slack.n, 2]  # reactive power

        # --- routine ---
        for aname in self.algebs.keys():
            a = self.algebs[aname]['a']  # array index
            if len(a) > 1:
                self.v[a[0]:a[-1]+1] = locals()[aname]
                self.algebs[aname]['algeb'].v = locals()[aname]
            else:
                self.v[a] = locals()[aname]
                self.algebs[aname]['algeb'].v = locals()[aname]

        return self.converged

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
    DC Power flow routine.
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
