"""
Module for system matrix make.
"""

import logging

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye, linalg

from andes.shared import rad2deg

logger = logging.getLogger(__name__)


class MatProcessor:
    """
    Class for matrix processing in AMS system.
    """

    def __init__(self, system):
        self.system = system

    @property
    def b(self):
        """
        Build the reduced branch resistance matrix.
        """
        x = self.system.Line.x.v[self.system.Line.u.v.astype(bool)]

        tap = np.ones(self.system.Line.n)

        return np.diag(1/x)

    @property
    def PTDF(self):
        """
        Build the power transfer distribution factor matrix.
        """
        nb = self.system.Bus.n
        nl = self.system.Line.n

        u = self.system.Line.u.v.astype(bool)       # in-service lines
        b = self.system.Line.x.v[u]                 # series susceptance
        tap = self.system.Line.tap.v[u] * rad2deg   # tap ratio
        b = b / tap

        f = self.system.Bus.idx2uid(self.system.Line.bus1.v)    # List of "from" buses
        t = self.system.Bus.idx2uid(self.system.Line.bus2.v)    # List of "to" buses
        f = np.array(f)[u]
        t = np.array(t)[u]
        i = np.concatenate((np.arange(nl), np.arange(nl)))      # Double set of row indices
        v = np.concatenate((np.ones(nl), -np.ones(nl)))         # Values for the connection matrix

        # Connection matrix
        Cft = csr_matrix((v, (i, np.concatenate((f, t)))), shape=(nl, nb))

        # Build Bf matrix
        Bf = csr_matrix((np.concatenate((b, -b)), (i, np.concatenate((f, t)))), shape=(nl, nb))

        Bbus = Cft.T.dot(Bf)
        logger.debug(f'Bbus:{Bbus.shape}')

        dP = eye(nb, format='csr')

        noref = self.system.Bus.idx2uid(self.system.PV.bus.v)
        noslack = self.system.Bus.idx2uid(self.system.PV.bus.v)
        dTheta = csr_matrix((nb, nb), dtype=float)
        dTheta[noref, :] = linalg.spsolve(Bbus[np.ix_(noslack, noref)], dP[noslack, :])

        logger.debug(f"Bf:{Bf.shape}, dTheta:{dTheta.shape}")
        ptdf = Bf.dot(dTheta)

        return Bbus, Bf, ptdf
