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
        self.om.AddOVars(OAlgeb=self.aBus,
                         lb=None,
                         ub=None,)
        self.om.AddOVars(OAlgeb=self.vBus,
                         lb=self.system.Bus.vmax,
                         ub=self.system.Bus.vmin,)
        self.om.AddOVars(OAlgeb=self.pPV,
                         lb=self.system.PV.pmin,
                         ub=self.system.PV.pmax,)
        self.om.AddOVars(OAlgeb=self.qPV,
                         lb=self.system.PV.qmin,
                         ub=self.system.PV.qmax,)
        self.om.AddOVars(OAlgeb=self.pSlack,
                         lb=self.system.Slack.pmin,
                         ub=self.system.Slack.pmax,)
        self.om.AddOVars(OAlgeb=self.qSlack,
                         lb=self.system.Slack.qmin,
                         ub=self.system.Slack.qmax,)

        # --- constraints ---
        self.om.AddOConstrs(name='pb',
                            n=self.system.Bus.n,
                            expr='Sum(p0) = Sum(pPV) + pSlack',
                            )
        self.om.AddOConstrs(name='qb',
                            n=self.system.Bus.n,
                            expr='Sum(q0) = Sum(qPV) + qSlack',
                            )
        self.om.AddOConstrs(name='v',
                            n=self.system.Bus.n,
                            expr='vBus**2 = v0**2 + 2 * v0 * q0 / x0',
                            )
        self.om.AddOConstrs(name='p',
                            n=self.system.Line.n,
                            expr='p0 = p1 + p2',
                            )
        # self.om.add_constraints()
        # self.om.add_objective()


class DCOPF(BaseRoutine):
    """
    DC Optimal Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "DC Optimal Power Flow"
        self._algeb_models = ['Bus', 'PV', 'Slack']

    def setup_om(self):
        # --- optimization modeling ---

        # --- decision variables ---
        self.om.AddOVars(OAlgeb=self.pPV,
                         lb=self.system.PV.pmin,
                         ub=self.system.PV.pmax,)
        self.om.AddOVars(OAlgeb=self.pSlack,
                         lb=self.system.Slack.pmin,
                         ub=self.system.Slack.pmax,)

        # --- constraints ---
        self.om.AddOConstrs(name='pb',
                            n=self.system.Bus.n,
                            expr='Sum(p0PQ) = Sum(pPV) + pSlack',
                            )
        self.om.AddOConstrs(name='llim',
                            n=self.system.Line.n,
                            expr='p0 = p1 + p2', # TODO; fix eqn
                            )
