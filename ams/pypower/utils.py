"""
PYPOWER utility functions.
"""
import logging  # NOQA
from copy import deepcopy  # NOQA

import numpy as np  # NOQA
from numpy import flatnonzero as find  # NOQA
import scipy.sparse as sp  # NOQA
from scipy.sparse import csr_matrix as c_sparse  # NOQA

from ams.pypower.idx import IDX  # NOQA

logger = logging.getLogger(__name__)


def bustypes(bus, gen):
    """
    Builds index lists of each type of bus (REF, PV, PQ).

    Generators with "out-of-service" status are treated as PQ buses with
    zero generation (regardless of Pg/Qg values in gen). Expects bus
    and gen have been converted to use internal consecutive bus numbering.

    Parameters
    ----------
    bus : ndarray
        Bus data.
    gen : ndarray
        Generator data.

    Returns
    -------
    ref : ndarray
        Index list of reference (REF) buses.
    pv : ndarray
        Index list of PV buses.
    pq : ndarray
        Index list of PQ buses.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    # get generator status
    nb = bus.shape[0]
    ng = gen.shape[0]
    # gen connection matrix, element i, j is 1 if, generator j at bus i is ON
    Cg = c_sparse((gen[:, IDX.gen.GEN_STATUS] > 0,
                   (gen[:, IDX.gen.GEN_BUS], range(ng))), (nb, ng))
    # number of generators at each bus that are ON
    bus_gen_status = (Cg * np.ones(ng, int)).astype(bool)

    # form index lists for slack, PV, and PQ buses
    ref = find((bus[:, IDX.bus.BUS_TYPE] == IDX.bus.REF) & bus_gen_status)  # ref bus index
    pv = find((bus[:, IDX.bus.BUS_TYPE] == IDX.bus.PV) & bus_gen_status)  # PV bus indices
    pq = find((bus[:, IDX.bus.BUS_TYPE] == IDX.bus.PQ) | ~bus_gen_status)  # PQ bus indices

    # pick a new reference bus if for some reason there is none (may have been
    # shut down)
    if (len(ref) == 0) & (len(pv) > 0):
        ref = np.zeros(1, dtype=int)
        ref[0] = pv[0]      # use the first PV bus
        pv = pv[1:]      # take it off PV list
    return ref, pv, pq


def sub2ind(shape, I, J, row_major=False):
    """
    Returns the linear indices of subscripts.

    Parameters
    ----------
    shape : tuple
        Shape of the grid or matrix.
    I : int
        Row subscript.
    J : int
        Column subscript.
    row_major : bool, optional
        If True, uses row-major order (default is False, using column-major order).

    Returns
    -------
    ind : int
        Linear index corresponding to the subscripts (I, J).
    """
    if row_major:
        ind = (I % shape[0]) * shape[1] + (J % shape[1])
    else:
        ind = (J % shape[1]) * shape[0] + (I % shape[0])

    return ind.astype(int)


def feval(func, *args, **kw_args):
    """
    Evaluates the function func using positional arguments args
    and keyword arguments kw_args.

    Parameters
    ----------
    func : str
        Name of the function to evaluate.
    *args : list
        Positional arguments for the function.
    **kw_args : dict
        Keyword arguments for the function.

    Returns
    -------
    result : any
        Result of evaluating the function.
    """
    return eval(func)(*args, **kw_args)


def have_fcn(name):
    """
    Checks if a Python module with the given name exists.

    Parameters
    ----------
    name : str
        Name of the Python module.

    Returns
    -------
    bool
        True if the module exists, False otherwise.
    """
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def get_reorder(A, idx, dim=0):
    """
    Returns A with one of its dimensions indexed::

        B = get_reorder(A, idx, dim)

    Returns A[:, ..., :, idx, :, ..., :], where dim determines
    in which dimension to place the idx.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ndims = np.ndim(A)
    if ndims == 1:
        B = A[idx].copy()
    elif ndims == 2:
        if dim == 0:
            B = A[idx, :].copy()
        elif dim == 1:
            B = A[:, idx].copy()
        else:
            raise ValueError('dim (%d) may be 0 or 1' % dim)
    else:
        raise ValueError('number of dimensions (%d) may be 1 or 2' % dim)

    return B


def set_reorder(A, B, idx, dim=0):
    """
    Assigns B to A with one of the dimensions of A indexed.

    @return: A after doing A(:, ..., :, IDX, :, ..., :) = B
    where DIM determines in which dimension to place the IDX.

    @see: L{get_reorder}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    A = A.copy()
    ndims = np.ndim(A)
    A = A.astype(B.dtype)
    if ndims == 1:
        A[idx] = B
    elif ndims == 2:
        if dim == 0:
            A[idx, :] = B
        elif dim == 1:
            A[:, idx] = B
        else:
            raise ValueError('dim (%d) may be 0 or 1' % dim)
    else:
        raise ValueError('number of dimensions (%d) may be 1 or 2' % dim)

    return A


def isload(gen):
    """
    Checks for dispatchable loads.

    Parameters
    ----------
    gen: np.ndarray
        The generator matrix.

    Returns
    -------
    array
        A column vector of 1's and 0's. The 1's correspond to rows of the
        C{gen} matrix which represent dispatchable loads. The current test is
        C{Pmin < 0 and Pmax == 0}. This may need to be revised to allow sensible
        specification of both elastic demand and pumped storage units.
    """
    return (gen[:, IDX.gen.PMIN] < 0) & (gen[:, IDX.gen.PMAX] == 0)


def hasPQcap(gen, hilo='B'):
    """
    Checks for P-Q capability curve constraints.

    Parameters
    ----------
    gen: np.ndarray
        The generator matrix.
    hilo : str, optional
        If 'U' this function returns C{True} only for rows corresponding to
        generators that require the upper constraint on Q.
        If 'L', only for those requiring the lower constraint.
        If not specified or has any other value it returns true for rows
        corresponding to gens that require either or both of the constraints.

    Returns
    -------
    array
        A column vector of 1's and 0's. The 1's correspond to rows of the
        C{gen} matrix which correspond to generators which have defined a
        capability curve (with sloped upper and/or lower bound on Q) and require
        that additional linear constraints be added to the OPF.

    Notes
    -----
        The C{gen} matrix in version 2 of the PYPOWER case format includes columns
        for specifying a P-Q capability curve for a generator defined as the
        intersection of two half-planes and the box constraints on P and Q.
        The two half planes are defined respectively as the area below the line
        connecting (Pc1, Qc1max) and (Pc2, Qc2max) and the area above the line
        connecting (Pc1, Qc1min) and (Pc2, Qc2min).

        It is smart enough to return C{True} only if the corresponding linear
        constraint is not redundant w.r.t the box constraints.
    """
    # check for errors capability curve data
    if np.any(gen[:, IDX.gen.PC1] > gen[:, IDX.gen.PC2]):
        logger.debug('hasPQcap: Pc1 > Pc2')
    if np.any(gen[:, IDX.gen.QC2MAX] > gen[:, IDX.gen.QC1MAX]):
        logger.debug('hasPQcap: Qc2max > Qc1max')
    if np.any(gen[:, IDX.gen.QC2MIN] < gen[:, IDX.gen.QC1MIN]):
        logger.debug('hasPQcap: Qc2min < Qc1min')

    L = np.zeros(gen.shape[0], bool)
    U = np.zeros(gen.shape[0], bool)
    k = np.nonzero(gen[:, IDX.gen.PC1] != gen[:, IDX.gen.PC2])

    if hilo != 'U':  # include lower constraint
        Qmin_at_Pmax = gen[k, IDX.gen.QC1MIN] + (gen[k, IDX.gen.PMAX] - gen[k, IDX.gen.PC1]) * (
            gen[k, IDX.gen.QC2MIN] - gen[k, IDX.gen.QC1MIN]) / (gen[k, IDX.gen.PC2] - gen[k, IDX.gen.PC1])
        L[k] = Qmin_at_Pmax > gen[k, IDX.gen.QMIN]

    if hilo != 'L':  # include upper constraint
        Qmax_at_Pmax = gen[k, IDX.gen.QC1MAX] + (gen[k, IDX.gen.PMAX] - gen[k, IDX.gen.PC1]) * (
            gen[k, IDX.gen.QC2MAX] - gen[k, IDX.gen.QC1MAX]) / (gen[k, IDX.gen.PC2] - gen[k, IDX.gen.PC1])
        U[k] = Qmax_at_Pmax < gen[k, IDX.gen.QMAX]

    return L | U


def fairmax(x):
    """
    Same as built-in C{max}, except breaks ties randomly.

    Takes a vector as an argument and returns the same output as the
    built-in function C{max} with two output parameters, except that
    where the maximum value occurs at more than one position in the
    vector, the index is chosen randomly from these positions as opposed
    to just choosing the first occurance.

    @see: C{max}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    val = max(x)  # find max value
    i = np.nonzero(x == val)  # find all positions where this occurs
    n = len(i)  # number of occurences
    idx = i(np.fix(n * np.random()) + 1)  # select index randomly among occurances
    return val, idx
