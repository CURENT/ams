"""
Shared constants and delayed imports.

This module is supplementary to the ``andes.shared`` module.
"""

import logging
from functools import wraps

import cvxpy as cp

from andes.utils.lazyimport import LazyImport


logger = logging.getLogger(__name__)


igraph = LazyImport("import igraph")

# NOTE: copied from CVXPY documentation
MIP_SOLVERS = ['CBC', 'COPT', 'GLPK_MI', 'CPLEX', 'GUROBI',
               'MOSEK', 'SCIP', 'XPRESS', 'SCIPY']

INSTALLED_SOLVERS = cp.installed_solvers()


def require_igraph(f):
    """
    Decorator for functions that require igraph.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            getattr(igraph, "__version__")
        except AttributeError:
            logger.error("Package `igraph` is not installed.")
        return f(*args, **kwargs)

    return wrapper


def require_MIP_solver(f):
    """
    Decorator for functions that require MIP solver.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not any(s in MIP_SOLVERS for s in INSTALLED_SOLVERS):
            raise ImportError("No MIP solver is available.")
        return f(*args, **kwargs)

    return wrapper
