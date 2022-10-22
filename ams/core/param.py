"""
Module for parameters used for describing models.
"""


class BaseParam():
    """
    The base parameter class.
    """

    def __init__(self):
        pass


class IdxParam(BaseParam):
    """
    The parameter class for indexing.
    """

    def __init__(self):
        pass


class NumParam(BaseParam):
    def __init__(self):
        BaseParam.__init(self)


class ExtParam(NumParam):
    def __init__(self):
        NumParam.__init(self)
