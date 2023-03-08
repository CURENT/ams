"""
OPF routines.
"""
from ams.routines.base import BaseRoutine
import numpy as np


class OPF(BaseRoutine):
    """
    AC Optimal Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system=system, config=config)
        self.info = "AC Optimal Power Flow"
        self._algeb_models = ['Bus', 'PV', 'Slack']

    def setup_om(self):
        # --- optimization modeling ---

        # --- decision variables ---
        self.om.add_Rvars(RAlgeb=self.aBus,
                          lb=None,
                          ub=None,)
        self.om.add_Rvars(RAlgeb=self.vBus,
                          lb=self.system.Bus.vmax,
                          ub=self.system.Bus.vmin,)
        self.om.add_Rvars(RAlgeb=self.qPV,
                          lb=self.system.PV.qmin,
                          ub=self.system.PV.qmax,)
        self.om.add_Rvars(RAlgeb=self.pSlack,
                          lb=self.system.Slack.pmin,
                          ub=self.system.Slack.pmax,)
        self.om.add_Rvars(RAlgeb=self.qSlack,
                          lb=self.system.Slack.qmin,
                          ub=self.system.Slack.qmax,)

        # --- constraints ---
        # self.om.add_constraints()
        # self.om.add_objective()


class DCOPF(OPF):
    """
    DC Optimal Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "DC Optimal Power Flow"
