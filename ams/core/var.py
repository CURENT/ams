"""
Base class for variables.
"""

from typing import Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Algeb:
    """
    Algebraic variable class.

    This class is simplified from ``andes.core.var.Algeb``.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 ):
        self.name = name
        self.info = info
        self.unit = unit

        self.tex_name = tex_name if tex_name else name
        self.owner = None  # instance of the owner Model
        self.id = None     # variable internal index inside a model (assigned in run time)

        # TODO: set a
        # address into the variable and equation arrays (dae.f/dae.g and dae.x/dae.y)
        self.a: np.ndarray = np.array([], dtype=int)

        self.v: np.ndarray = np.array([], dtype=float)  # variable value array

    def __repr__(self):
        if self.owner.n == 0:
            span = []

        elif 1 <= self.owner.n <= 20:
            span = f'a={self.a}, v={self.v}'

        else:
            span = []
            span.append(self.a[0])
            span.append(self.a[-1])
            span.append(self.a[1] - self.a[0])
            span = ':'.join([str(i) for i in span])
            span = 'a=[' + span + ']'

        return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}, {span}'

    @property
    def class_name(self):
        return self.__class__.__name__
