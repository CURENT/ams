"""
Builds the B matrices and phase shift injections for DC power flow.

This module is adapted from ``pypower.makeBdc``
"""

import logging

import numpy as np
from scipy.sparse import csr_matrix as sparse

from andes.shared import deg2rad

from ams.solver.pypower.idx_bus import BUS_I
from ams.solver.pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, SHIFT, BR_STATUS

logger = logging.getLogger(__name__)


def makeBdc(baseMVA, bus, branch):
    """
    Builds the B matrices and phase shift injections for DC power flow.

    Returns the B matrices and phase shift injection vectors needed for a
    DC power flow.
    The bus real power injections are related to bus voltage angles by::
        P = Bbus * Va + PBusinj
    The real power flows at the from end the lines are related to the bus
    voltage angles by::
        Pf = Bf * Va + Pfinj
    Does appropriate conversions to p.u.
    
    Parameters
    ----------
    baseMVA : float
        Base MVA.
    bus : ndarray
        Bus matrix.
    branch : ndarray
        Branch matrix.
    """
    # constants
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of lines

    # check that bus numbers are equal to indices to bus (one set of bus nums)
    if any(bus[:, BUS_I] != list(range(nb))):
        logger.warning('makeBdc: buses must be numbered consecutively in '
                       'bus matrix\n')

    # for each branch, compute the elements of the branch B matrix and the phase
    # shift "quiescent" injections, where
    ##
    # | Pf |   | Bff  Bft |   | Vaf |   | Pfinj |
    # |    | = |          | * |     | + |       |
    # | Pt |   | Btf  Btt |   | Vat |   | Ptinj |
    ##
    stat = branch[:, BR_STATUS]  # ones at in-service branches
    b = stat / branch[:, BR_X]  # series susceptance
    tap = np.ones(nl)  # default tap ratio = 1
    i = np.flatnonzero(branch[:, TAP])  # indices of non-zero tap ratios
    tap[i] = branch[i, TAP]  # assign non-zero tap ratios
    b = b / tap

    # build connection matrix Cft = Cf - Ct for line and from - to buses
    f = np.array(branch[:, F_BUS]).astype(int)  # list of "from" buses
    t = np.array(branch[:, T_BUS]).astype(int)  # list of "to" buses
    i = np.r_[range(nl), range(nl)]  # double set of row indices
    # connection matrix
    Cft = sparse((np.r_[np.ones(nl), -np.ones(nl)],      # data
                  (i, np.r_[f, t])),               # row, col
                  shape=(nl, nb))

    # build Bf such that Bf * Va is the vector of real branch powers injected
    # at each branch's "from" bus
    Bf = sparse((np.r_[b, -b],                     # data
                 (i, np.r_[f, t])),                # row, col
                 shape=(nl, nb))  # = spdiags(b, 0, nl, nl) * Cft

    # build Bbus
    Bbus = Cft.T * Bf

    # build phase shift injection vectors
    Pfinj = b * (-branch[:, SHIFT] * deg2rad)  # injected at the from bus ...
    # Ptinj = -Pfinj                            ## and extracted at the to bus
    Pbusinj = Cft.T * Pfinj  # Pbusinj = Cf * Pfinj + Ct * Ptinj

    return Bbus, Bf, Pbusinj, Pfinj
