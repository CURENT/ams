"""
Module for system matrix make.
"""

import logging

from andes.shared import np

from ams.solver.pypower.makePTDF import makePTDF, makeBdc
from ams.io.pypower import system2ppc

logger = logging.getLogger(__name__)


class MatProcessor:
    """
    Class for matrix processing in AMS system.
    """

    def __init__(self, system):
        self.system = system

    def make(self):
        """
        Make the PTDF matrix and connectivity matrix.
        """
        ppc = system2ppc(self.system)
        PTDF = makePTDF(ppc['baseMVA'], ppc['bus'], ppc['branch'])
        _, _, _, _, Cft = makeBdc(ppc['baseMVA'], ppc['bus'], ppc['branch'])
        return PTDF, Cft
