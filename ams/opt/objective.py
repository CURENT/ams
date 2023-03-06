"""
Module for optimization objectives.
"""


class Objective:
    """
    Base class for objective functions.
    """
    def __init__(self, name, sense):
        self.name = name
        self.sense = sense
