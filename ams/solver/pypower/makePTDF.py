"""
Builds the DC PTDF matrix for a given choice of slack.

This module is adapted from PYPOWER.
"""
import logging

import numpy as np

from scipy.sparse.linalg import spsolve, factorized
from scipy.sparse import csr_matrix as sparse

from ams.solver.pypower.idx_bus import BUS_TYPE, REF, BUS_I
from ams.solver.pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, SHIFT, BR_STATUS

from andes.shared import deg2rad

logger = logging.getLogger(__name__)


def makePTDF(baseMVA, bus, branch,
             slack=None,
             result_side=0,
             using_sparse_solver=False,
             branch_id=None,
             reduced=False):
    """
    Builds the DC PTDF matrix for a given choice of slack.

    The matrix is
    C{nbr x nb}, where C{nbr} is the number of branches and C{nb} is the
    number of buses. The C{slack} can be a scalar (single slack bus) or an
    C{nb x 1} column vector of weights specifying the proportion of the
    slack taken up at each bus. If the C{slack} is not specified the
    reference bus is used by default.
    For convenience, C{slack} can also be an C{nb x nb} matrix, where each
    column specifies how the slack should be handled for injections
    at that bus.
    To restrict the PTDF computation to a subset of branches, supply a list of ppci branch indices in C{branch_id}.
    If C{reduced==True}, the output is reduced to the branches given in C{branch_id}, otherwise the complement rows are set to NaN.

    Parameters
    ----------
    baseMVA : float
        Base MVA.
    bus : ndarray
        Bus matrix.
    branch : ndarray
        Branch matrix.
    slack : int or ndarray, optional
        Slack bus index or vector of slack weights.
    result_side : int, optional
        Side of the branch results to use for PTDF calculation.
        0 - from side, 1 - to side, 2 - average of from and to.
    using_sparse_solver : bool, optional
        Use sparse solver for PTDF calculation.
    branch_id : list, optional
        List of branch indices to use for PTDF calculation.
    reduced : bool, optional
        Return reduced PTDF matrix.

    Returns
    -------
    H : ndarray
        PTDF matrix.
    """
    if reduced and branch_id is None:
        raise ValueError("'reduced=True' is only valid if branch_id is not None")

    # use reference bus for slack by default
    if slack is None:
        slack = np.flatnonzero(bus[:, BUS_TYPE] == REF)
        slack = slack[0]

    # set the slack bus to be used to compute initial PTDF
    if np.isscalar(slack):
        slack_bus = slack
    else:
        slack_bus = 0  # use bus 1 for temp slack bus

    nb = bus.shape[0]
    nbr = branch.shape[0]
    noref = np.arange(1, nb)  # use bus 1 for voltage angle reference
    noslack = np.flatnonzero(np.arange(nb) != slack_bus)

    # check that bus numbers are equal to indices to bus (one set of bus numbers)
    if any(bus[:, BUS_I] != np.arange(nb)):
        stderr.write('makePTDF: buses must be numbered consecutively')

    if reduced:
        H = np.zeros((len(branch_id), nb))
    else:
        H = np.zeros((nbr, nb))
    # compute PTDF for single slack_bus
    if using_sparse_solver:
        Bbus, Bf, *_ = makeBdc(baseMVA, bus, branch, return_csr=False)

        Bbus = Bbus.real
        if result_side == 1:
            Bf *= -1
        if branch_id is not None:
            Bf = Bf.real.toarray()
            if reduced:
                H[:, noslack] = spsolve(Bbus[np.ix_(noslack, noref)].T, Bf[np.ix_(branch_id, noref)].T).T
            else:
                H[np.ix_(branch_id, noslack)] = spsolve(
                    Bbus[np.ix_(noslack, noref)].T, Bf[np.ix_(branch_id, noref)].T).T
        elif Bf.shape[0] < 2000:
            Bf = Bf.real.toarray()
            H[:, noslack] = spsolve(Bbus[np.ix_(noslack, noref)].T, Bf[:, noref].T).T
        else:
            # Use memory saving modus
            Bbus_fact = factorized(Bbus[np.ix_(noslack, noref)].T)
            for i in range(0, Bf.shape[0], 32):
                H[i:i+32, noslack] = Bbus_fact(Bf[i:i+32, noref].real.toarray().T).T
    else:
        Bbus, Bf, *_ = makeBdc(baseMVA, bus, branch)
        Bbus, Bf = np.real(Bbus.toarray()), np.real(Bf.toarray())
        if result_side == 1:
            Bf *= -1
        if branch_id is not None:
            if reduced:
                H[:, noslack] = np.linalg.solve(Bbus[np.ix_(noslack, noref)].T, Bf[np.ix_(branch_id, noref)].T).T
            else:
                H[np.ix_(branch_id, noslack)] = np.linalg.solve(
                    Bbus[np.ix_(noslack, noref)].T, Bf[np.ix_(branch_id, noref)].T).T
        else:
            H[:, noslack] = np.linalg.solve(Bbus[np.ix_(noslack, noref)].T, Bf[:, noref].T).T

    # distribute slack, if requested
    if not np.isscalar(slack):
        if len(slack.shape) == 1:  # slack is a vector of weights
            slack = slack / sum(slack)  # normalize weights

            # conceptually, we want to do ...
            # H = H * (eye(nb, nb) - slack * ones((1, nb)))
            # ... we just do it more efficiently
            v = np.dot(H, slack)
            for k in range(nb):
                H[:, k] = H[:, k] - v
        else:
            H = np.dot(H, slack)

    return H


def makeBdc(baseMVA, bus, branch):
    """
    Builds the B matrices and phase shift injections for DC power flow.

    The bus real power injections are related to bus voltage angles by::
        P = Bbus * Va + PBusinj
    The real power flows at the from end the lines are related to the bus
    voltage angles by::
        Pf = Bf * Va + Pfinj

    Parameters
    ----------
    baseMVA : float
        Base MVA.
    bus : ndarray
        Bus matrix.
    branch : ndarray
        Branch matrix.

    Returns
    -------
    Bbus : sparse matrix
        Bus susceptance matrix.
    Bf : sparse matrix
        From branch susceptance matrix.
    Pbusinj : ndarray
        Bus real power injection vector.
    Pfinj : ndarray
        From end real power injection vector.
    """
    # constants
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of lines

    # check that bus numbers are equal to indices to bus (one set of bus nums)
    if any(bus[:, BUS_I] != list(range(nb))):
        logger.warning('buses must be numbered consecutively in bus matrix.')

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
