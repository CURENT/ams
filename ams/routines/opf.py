"""
OPF routines.
"""
from ams.routines.base import BaseRoutineModel, timer, BaseFormulation


class OPFFormulation(BaseFormulation):
    """
    OPF Formulation.
    """

    def __init__(self, routine=None):
        super().__init__(routine)
    
    def add_vars(self, type=None, lb=None, ub=None, value=None):
        """
        Add variables.
        """
        self.vars.add(type=type, lb=lb, ub=ub, value=value)
    
    def add_constraints(self, type=None, lb=None, ub=None, value=None):
        """
        Add constraints.
        """
        self.constraints.add(type=type, lb=lb, ub=ub, value=value)
    
    def add_objective(self, type=None, lb=None, ub=None, value=None):
        """
        Add objective.
        """
        self.objective.add(type=type, lb=lb, ub=ub, value=value)


class OPF(BaseRoutineModel, OPFFormulation):
    """
    AC Optimal Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        BaseRoutineModel.__init__(self, system=system, config=config)
        OPFFormulation.__init__(self, routine=self)
        self.info = "AC Optimal Power Flow"
        self._algeb_models = ['Bus', 'PV', 'Slack']


class DCOPF(OPF):
    """
    DC Optimal Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "DC Optimal Power Flow"
