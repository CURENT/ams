"""
OPF routines.
"""
from ams.routines.base import BaseRoutine, timer, BaseFormulation


class OPFFormulation(BaseFormulation):
    """
    Optimal Power Flow formulation.
    """

    def __init__(self):
        super().__init__()


class OPF(BaseRoutine, OPFFormulation):
    """
    AC Optimal Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        BaseRoutine.__init__(self, system, config)
        self.info = "AC Optimal Power Flow"
        self._algeb_models = ['Bus', 'PV', 'Slack']
        OPFFormulation.__init__(self)


    @timer
    def _solve(self, **kwargs):
        return None

    def _unpack(self):
        """
        Unpack the results.
        """
        return None

    def run(self, **kwargs):
        """
        Run OPF.
        """
        super().run(**kwargs)


class DCOPF(OPF):
    """
    DC Optimal Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "DC Optimal Power Flow"
