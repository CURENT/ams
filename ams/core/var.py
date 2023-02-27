"""
Base class for variables.
"""

from typing import Optional
from andes.shared import np


class Algeb:
    """
    Algebraic variable class.
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

        self.a: np.ndarray = np.array([], dtype=int)
        self.v: np.ndarray = np.array([], dtype=float)

    def set_address(self, addr: np.ndarray, contiguous=False):
        """
        Set the address of internal variables.

        Parameters
        ----------
        addr : np.ndarray
            The assigned address for this variable
        contiguous : bool, optional
            If the addresses are contiguous
        """

        self.a = addr
        self.n = len(self.a)

        # NOT IN USE
        self.ae = np.array(self.a)
        self.av = np.array(self.a)
        # -----------

        self._contiguous = contiguous

        if self._contiguous:
            if self.e_setter is False:
                self.e_inplace = True

            if self.v_setter is False:
                self.v_inplace = True
