"""
Module for system matrix make.
"""

import logging

from andes.shared import np

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

    @property
    def Cft(self):
        """
        Build the connection matrix.

        # TODO: add support for sparsity
        """
        # size
        nb = self.system.Bus.n
        nl = self.system.Line.n
        # Bus matrix location
        fbus = np.array(self.system.Bus.idx2uid(self.system.Line.bus1.v))
        tbus = np.array(self.system.Bus.idx2uid(self.system.Line.bus2.v))
        # connection matrix
        Cft = np.zeros((nl, nb))
        Cft[fbus - 1, np.arange(nl)] = 1
        Cft[tbus - 1, np.arange(nl)] = -1
        return Cft

    def rePTDF(self):
        """
        Restructure the PTDF matrix to be used in the following equation:
        .. math::

            [PTDF_{1}, PTDF_{2}] \cdot [ [pg_{1}, 0]^T - [pd_{1}, pd_{2}]^T ]
        """
        gen_bus = self.system.StaticGen.get(src='bus', attr='v',
                                           idx=self.system.StaticGen.get_idx())
        all_bus = self.system.Bus.idx.v
        regBus = [int(bus) if isinstance(bus, (int, float)) else bus for bus in gen_bus]
        redBus = [int(bus) if isinstance(bus, (int, float)) else bus for bus in all_bus if bus not in gen_bus]

        uid_regBus = self.system.Bus.idx2uid(regBus)
        uid_redBus = self.system.Bus.idx2uid(redBus)

        PTDF = self.PTDF
        return PTDF[:, uid_regBus], PTDF[:, uid_redBus]
