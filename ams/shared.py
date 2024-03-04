"""
Shared constants and delayed imports.

This module is supplementary to the ``andes.shared`` module.
"""
import logging
from functools import wraps
from datetime import datetime

import cvxpy as cp


logger = logging.getLogger(__name__)


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
