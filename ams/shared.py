"""
Shared constants and delayed imports.

This module is supplementary to the ``andes.shared`` module.
"""
import logging
from functools import wraps
from datetime import datetime
from collections import OrderedDict

import cvxpy as cp

from andes.shared import pd
from andes.utils.lazyimport import LazyImport

logger = logging.getLogger(__name__)

sps = LazyImport('import scipy.sparse as sps')
np = LazyImport('import numpy as np')

# --- NumPy constants ---
# NOTE: In NumPy 2.0, np.Inf and np.NaN are deprecated.
inf = np.inf
nan = np.nan

# NOTE: copied from CVXPY documentation
MIP_SOLVERS = ['CBC', 'COPT', 'GLPK_MI', 'CPLEX', 'GUROBI',
               'MOSEK', 'SCIP', 'XPRESS', 'SCIPY']

INSTALLED_SOLVERS = cp.installed_solvers()

# NOTE: copyright
year_end = datetime.now().year
copyright_msg = f'Copyright (C) 2023-{year_end} Jinning Wang'


def require_MIP_solver(f):
    """
    Decorator for functions that require MIP solver.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not any(s in MIP_SOLVERS for s in INSTALLED_SOLVERS):
            raise ModuleNotFoundError("No MIP solver is available.")
        return f(*args, **kwargs)

    return wrapper


ppc_cols = OrderedDict([
    ('bus', ['bus_i', 'type', 'pd', 'qd', 'gs', 'bs', 'area', 'vm', 'va',
             'baseKV', 'zone', 'vmax', 'vmin', 'lam_p', 'lam_q',
             'mu_vmax', 'mu_vmin']),
    ('branch', ['fbus', 'tbus', 'r', 'x', 'b', 'rate_a', 'rateB_b', 'rate_c',
                'tap', 'shift', 'status', 'angmin',
                'angmax', 'pf', 'qf', 'pt', 'qt', 'mu_sf', 'mu_st',
                'mu_angmin', 'mu_angmax']),
    ('gen', ['bus', 'pg', 'qg', 'qmax', 'qmin', 'vg', 'mbase', 'status',
             'pmax', 'pmin', 'pc1', 'pc2', 'qc1min', 'qc1max', 'qc2min',
             'qc2max', 'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q',
             'apf', 'mu_pmax', 'mu_pmin', 'mu_qmax', 'mu_qmin']),
    ('gencost', ['model', 'startup', 'shutdown', 'n', 'c2', 'c1', 'c0']),
])


def ppc2df(ppc, model='bus', ppc_cols=ppc_cols):
    """
    Convert PYPOWER dict to pandas DataFrame.

    Parameters
    ----------
    ppc : dict
        PYPOWER case dict.
    model : str
        Model name.
    ppc_cols : dict
        Column names.

    Returns
    -------
    pandas.DataFrame
        DataFrame.

    Examples
    --------
    >>> import ams
    >>> sp = ams.system.example()
    >>> ppc = ams.io.pypower.system2ppc(sp)
    >>> ppc_bus = ams.shared.ppc2df(ppc, 'bus')
    """
    if model not in ppc_cols.keys():
        raise ValueError(f"Invalid model {model}")

    df = pd.DataFrame(ppc[model], columns=ppc_cols[model][0:ppc[model].shape[1]])
    return df
