"""
Module for optimization constraints.
"""
import logging

from typing import Optional

logger = logging.getLogger(__name__)


class Constraint:
    """
    Base class for constraints.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 n: Optional[int] = 1,
                 expr: Optional[str] = None,
                 ):
        self.name = name
        self.n = n
        self.expr = expr

    @property
    def class_name(self):
        return self.__class__.__name__

    def __repr__(self):
        dev_text = 'Constraint' if self.n == 1 else 'Constraints'
        return f'{self.name} ({self.n} {dev_text}) at {hex(id(self))}'
