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
        self.om.add_vars(name='aBus',
                         type=np.float64,
                         n=self.system.Bus.n,
                         lb=np.full(self.system.Bus.n, -np.inf),
                         ub=np.full(self.system.Bus.n, np.inf),
                         )
        self.om.add_vars(name='vBus',
                         type=np.float64,
                         n=self.system.Bus.n,
                         lb=self.system.Bus.vmin.v,
                         ub=self.system.Bus.vmin.v,
                         )
        # self.om.add_constraints()
        # self.om.add_objective()


class DCOPF(OPF):
    """
    DC Optimal Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "DC Optimal Power Flow"
