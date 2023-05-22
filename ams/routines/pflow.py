"""
Power flow routines.
"""
import logging
from collections import OrderedDict

import numpy as np

from andes.shared import deg2rad

from ams.solver.pypower.runpf import runpf, rundcpf

from ams.io.pypower import system2ppc
from ams.core.param import RParam
from ams.core.var import RAlgeb

from ams.routines.dcpf import DCPFlowData, DCPFlowBase

logger = logging.getLogger(__name__)


class PFlowData(DCPFlowData):
    """
    AC Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        DCPFlowData.__init__(self, system, config)

    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        res, success = runpf(ppc, **kwargs)
        return ppc, success

class PFlowBase(DCPFlowBase):
    """
    Base class for AC Power Flow model.
    """

    def __init__(self, system, config):
        DCPFlowBase.__init__(self, system, config)


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
        for raname, ralgeb in self.ralgebs.items():
            ralgeb.v = ralgeb.Algeb.v.copy()

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
