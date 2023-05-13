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
        self.config = parent.config
        self.class_name = parent.class_name
        self.tex_names = OrderedDict()

    def generate_symbols(self):
        """
        Generate symbols for all variables.
        """
        logger.debug(f'- Generating symbols for {self.parent.class_name}')

        # process tex_names defined in model
        # -----------------------------------------------------------
        for key in self.parent.tex_names.keys():
            self.tex_names[key] = sp.Symbol(self.parent.tex_names[key])
        
        for oname, oalgeb in self.parent.oalgebs.items():
            self.inputs_dict[oname] = sp.MatrixSymbol(oalgeb.tex_name, oalgeb.v.shape[0], 1)