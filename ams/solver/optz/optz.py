"""
Module for optimization modeling elements.

This module is developed from andes.core.var.
"""

from andes.core.var import BaseVar


class Var(BaseVar):
    """
    The variable class for dispatch modeling.

    This class connects the variables in AMS with those in the solver.

    This class is revised from andes.core.var.BaseVar.
    """

    def __init__(self):
        BaseVar.__init(self)
        # TODO: the definition might need to be revised: 1) block unused methods; 2) add new attributes


class Obj(BaseOptz):
    """
    The objective class.
    """

    def __init__(self):
        BaseOptz.__init(self)


class Constr(BaseOptz):
    """
    The constraint class.
    """

    def __init__(self):
        BaseOptz.__init(self)
