"""
Base class for variables.
"""

from typing import Optional

import numpy as np


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
        self.v = np.empty(0)  # variable value

    def __repr__(self):
        n = self.owner.n
        dev_text = 'Algeb' if n == 1 else 'Algebs'
        return f'{self.owner.class_name}.{self.name} ({n} {dev_text}) at {hex(id(self))}'


class GAlgeb(Algeb):
    """
    Group Algebraic variable class.

    Extends ``Algeb`` to adapt to group algebraic variables.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, info=info, unit=unit)

        self.tex_name = tex_name if tex_name else name
        self.owner = None  # instance of the owner group

    def __repr__(self):
        n = self.owner.n
        dev_text = 'Algeb' if n == 1 else 'Algebs'
        return f'{self.owner.class_name}.{self.name} ({n} {dev_text}) at {hex(id(self))}'
