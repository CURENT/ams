"""
Power flow routines.
"""
import logging
from collections import OrderedDict

import numpy as np

from andes.shared import deg2rad

from ams.routines.base import BaseRoutine, timer
from ams.solver.pypower.runpf import runpf, rundcpf

from ams.io.pypower import system2ppc

logger = logging.getLogger(__name__)


class PFlow(BaseRoutine):
    """
    AC Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "AC Power flow"
        self.rtn_models = OrderedDict([
            ('Bus', ['vmax', 'vmin']),
            ('Slack', ['pmax', 'pmin', 'qmax', 'qmin']),
            ('PV', ['pmax', 'pmin', 'qmax', 'qmin']),
        ])

    @timer
    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        res, success = runpf(ppc, **kwargs)
        return ppc, success
    
    def unpack(self, ppc):
        """
        Unpack results from ppc to system.
        """
        system = self.system

        # --- Bus ---
        system.Bus.v.v = ppc['bus'][:, 7]  # voltage magnitude
        system.Bus.a.v = ppc['bus'][:, 8] * deg2rad  # voltage angle

        # --- PV ---
        system.PV.q.v = ppc['gen'][system.Slack.n:, 2]  # reactive power

        # --- Slack ---
        system.Slack.p.v = ppc['gen'][:system.Slack.n, 1]  # active power
        system.Slack.q.v = ppc['gen'][:system.Slack.n, 2]  # reactive power

        # --- store results into routine algeb ---
        for raname, oalgeb in self.oalgebs.items():
            oalgeb.v = oalgeb.Algeb.v.copy()

    def run(self, **kwargs):
        """
        Run power flow.
        """
        (ppc, success), elapsed_time = self.solve(**kwargs)
        self.exit_code = int(not success)
        if success:
            info = f'{self.class_name} completed in {elapsed_time} seconds with exit code {self.exit_code}.'
        else:
            info = f'{self.class_name} failed in {elapsed_time} seconds with exit code {self.exit_code}.'
        logger.info(info)
        self.exec_time = float(elapsed_time.split(' ')[0])
        self.unpack(ppc)
        return self.exit_code

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
