"""
Module for optimization constraints.
"""


class Constraint:
    """
    Base class for constraints.
    """
    def __init__(self, name, sense):
        self.name = name
        self.sense = sense
