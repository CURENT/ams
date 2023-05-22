"""
Module for system matrix make.
"""

import logging

from ams.solver.pypower.makePTDF import makePTDF
from ams.io.pypower import system2ppc

logger = logging.getLogger(__name__)


class MatProcessor:
    """
    Class for matrix processing in AMS system.
    """

    def __init__(self, system):
        self.system = system

    @property
    def PTDF(self):
        """
        Build the power transfer distribution factor matrix.
        """
        ppc = system2ppc(self.system)
        return makePTDF(ppc['baseMVA'], ppc['bus'], ppc['branch'])
