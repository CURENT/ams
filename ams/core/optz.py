"""
Module for optimization modeling elements.
"""


class BaseOptz():
    """
    The base optimization class.
    """

    def __init__(self):
        pass


class Var(BaseOptz):
    """
    The variable class.
    """

    def __init__(self):
        BaseOptz.__init(self)


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
