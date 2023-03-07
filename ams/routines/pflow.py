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
    AC Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "AC Power flow"
        self._algeb_models = ['Bus', 'PV', 'Slack']

    def run(self, **kwargs):
        """
        Run power flow.
        """
        ppc = system2ppc(self.system)
        res, exit_code = runpf(ppc, **kwargs)
        self.exit_code = int(exit_code)

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

        # --- store results into routine algeb ---
        for raname, ralgeb in self.ralgebs.items():
            ralgeb.v = ralgeb.Algeb.v.copy()

        return bool(exit_code)

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


class DCPF(PFlow):
    """
    DC Power Flow routine.
    """
    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "DC Power Flow"

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
