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

class RAlgeb(Algeb):
    """
    Class for algebraic variables used in a routine.

    Extends ``Algeb`` to adapt to group algebraic variables.

    Parameters
    ----------
    info : str, optional
        Descriptive information
    unit : str, optional
        Unit
    tex_name : str
        LaTeX-formatted variable symbol. If is None, the value of `name` will be
        used.
    discrete : Discrete
        Discrete component on which this variable depends. ANDES will call
        `check_var()` of the discrete component before initializing this
        variable.
    name : str, optional
        Variable name. One should typically assigning the name directly because
        it will be automatically assigned by the model. The value of ``name``
        will be the symbol name to be used in expressions.
    is_group : bool, optional
        Indicate if this variable is a group variable. If True, `group_name`
        must be provided.
    group_name : str, optional
        Name of the group. Must be provided if `is_group` is True.
    
    Attributes
    ----------
    a : array-like
        variable address
    v : array-like
        local-storage of the variable value
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 owner_name: Optional[str] = None,
                 lb: Optional[str] = None,
                 ub: Optional[str] = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, info=info, unit=unit)
        self.is_group = False
        self.owner_name = owner_name  # indicate if this variable is a group variable
        self.owner = None  # instance of the owner model or group
        self.lb = lb
        self.ub = ub
        self.id = None     # variable internal index inside a model (assigned in run time)

        # TODO: set a
        # address into the variable and equation arrays (dae.f/dae.g and dae.x/dae.y)
        self.a: np.ndarray = np.array([], dtype=int)

        self.v: np.ndarray = np.array([], dtype=float)  # variable value array

    def get_idx(self):
        if self.is_group:
            return self.owner.get_idx()
        else:
            return self.owner.idx.v

    @property
    def n(self):
        return self.owner.n

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
