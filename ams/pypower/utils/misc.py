"""
PYPOWER utilities.
"""

import numpy as np  # NOQA
from numpy import flatnonzero as find  # NOQA
from scipy.sparse import csr_matrix as c_sparse  # NOQA

import ams.pypower.utils.const as IDX  # NOQA

# define EPS as commonly used small number
EPS = np.finfo(float).eps


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
    if len(ref) == 0:
        ref = np.zeros(1, dtype=int)
        ref[0] = pv[0]      # use the first PV bus
        pv = pv[1:]      # take it off PV list

    return ref, pv, pq


def isload(gen):
    """
    Checks for dispatchable loads.

    Returns a column vector of 1's and 0's. The 1's correspond to rows of the
    gen matrix which represent dispatchable loads. The current test is
    Pmin < 0 and Pmax == 0. This may need to be revised to allow sensible
    specification of both elastic demand and pumped storage units.

    Parameters
    ----------
    gen : ndarray
        Generator data.

    Returns
    -------
    is_load : ndarray
        Boolean array indicating if each generator is a dispatchable load.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    return (gen[:, IDX.gen.PMIN] < 0) & (gen[:, IDX.gen.PMAX] == 0)


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
