"""
Symbolic processor class for AMS models.

This module is revised from ``andes.core.symprocessor``.
"""

import logging
from collections import OrderedDict

import sympy as sp

logger = logging.getLogger(__name__)

class SymProcessor:
    """
    Class for symbolic processing in AMS routine.
    """

    def __init__(self, parent):
        self.parent = parent
        self.inputs_dict = OrderedDict()

    def generate_symbols(self):
        """
        Generate symbols for all variables.
        """
        logger.debug(f'Generating symbols for {self.parent.class_name}')
        
        for oalgeb in self.parent.oalgebs.keys():
            self.inputs_dict[oalgeb] = sp.Symbol(oalgeb)
