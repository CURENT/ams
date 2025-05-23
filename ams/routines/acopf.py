"""
ACOPF routines.
"""

from ams.routines.pypower import ACOPF1


class ACOPF(ACOPF1):
    """
    Alias for ACOPF1.
    """

    def __init__(self, system, config):
        super().__init__(system, config)
