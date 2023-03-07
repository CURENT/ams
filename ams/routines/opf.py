"""
OPF routines.
"""
from ams.routines.base import BaseRoutine

class OPF(BaseRoutine):
    """
    AC Optimal Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "AC Optimal Power Flow"
        self._algeb_models = ['Bus', 'PV', 'Slack']

class DCOPF(OPF):
    """
    DC Optimal Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "DC Optimal Power Flow"
