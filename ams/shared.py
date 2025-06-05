"""
Shared constants and delayed imports.

This module is supplementary to the ``andes.shared`` module.
"""
import logging
import sys
import unittest
from functools import wraps
from time import strftime

import cvxpy as cp

from andes.utils.lazyimport import LazyImport

from andes.system import System as adSystem

from ._version import get_versions

logger = logging.getLogger(__name__)

sps = LazyImport('import scipy.sparse as sps')
np = LazyImport('import numpy as np')
pd = LazyImport('import pandas as pd')
ppoption = LazyImport('from pypower.ppoption import ppoption')
runpf = LazyImport('from pypower.runpf import runpf')
runopf = LazyImport('from pypower.runopf import runopf')
opf = LazyImport('from gurobi_optimods import opf')
tqdm = LazyImport('from tqdm.auto import tqdm')

# --- an empty ANDES system ---
empty_adsys = adSystem(autogen_stale=False)
ad_models = list(empty_adsys.models.keys())
ad_pf_models = [mname for mname, model in empty_adsys.models.items() if model.flags.pflow]
ad_dyn_models = [mname for mname, model in empty_adsys.models.items() if model.flags.tds and not model.flags.pflow]

# --- NumPy constants ---
# NOTE: In NumPy 2.0, np.Inf and np.NaN are deprecated.
inf = np.inf
nan = np.nan

# --- misc constants ---
_prefix = r" - --------------> | "  # NOQA
_max_length = 80                    # NOQA

# NOTE: copyright
copyright_msg = "Copyright (C) 2023-2025 Jinning Wang"
nowarranty_msg = "AMS comes with ABSOLUTELY NO WARRANTY"
report_time = strftime("%m/%d/%Y %I:%M:%S %p")
version_msg = f"AMS {get_versions()['version']}"

summary_row = {'field': 'Info',
               'comment': version_msg,
               'comment2': report_time,
               'comment3': nowarranty_msg,
               'comment4': copyright_msg}

summary_name = "Summary"  # ensure model Summary's name is consistent

# NOTE: copied from CVXPY documentation, last checked on 2024/10/30, v1.5
mip_solvers = ['CBC', 'COPT', 'GLPK_MI', 'CPLEX', 'GUROBI',
               'MOSEK', 'SCIP', 'XPRESS', 'SCIPY']

misocp_solvers = ['MOSEK', 'CPLEX', 'GUROBI', 'XPRESS', 'SCIP']

installed_solvers = cp.installed_solvers()

installed_mip_solvers = [s for s in installed_solvers if s in mip_solvers]


def require_MISOCP_solver(f):
    """
    Decorator for functions that require MISOCP solver.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not any(s in misocp_solvers for s in installed_solvers):
            raise ModuleNotFoundError("No MISOCP solver is available.")
        return f(*args, **kwargs)

    return wrapper


def require_MIP_solver(f):
    """
    Decorator for functions that require MIP solver.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not any(s in mip_solvers for s in installed_solvers):
            raise ModuleNotFoundError("No MIP solver is available.")
        return f(*args, **kwargs)

    return wrapper


def skip_unittest_without_MIP(f):
    """
    Decorator for skipping tests that require MIP solver.
    """
    def wrapper(*args, **kwargs):
        if any(s in mip_solvers for s in installed_solvers):
            pass
        else:
            raise unittest.SkipTest("No MIP solver is available.")
        return f(*args, **kwargs)
    return wrapper


def skip_unittest_without_MISOCP(f):
    """
    Decorator for skipping tests that require MISOCP solver.
    """
    def wrapper(*args, **kwargs):
        if any(s in misocp_solvers for s in installed_solvers):
            pass
        else:
            raise unittest.SkipTest("No MISOCP solver is available.")
        return f(*args, **kwargs)
    return wrapper


def skip_unittest_without_PYPOWER(f):
    """
    Decorator for skipping tests that require PYPOWER.
    """
    def wrapper(*args, **kwargs):
        try:
            import pypower  # NOQA
        except ImportError:
            raise unittest.SkipTest("PYPOWER is not available.")
        return f(*args, **kwargs)
    return wrapper


def skip_unittest_without_gurobi_optimods(f):
    """
    Decorator for skipping tests that require gurobi_optimods.
    """
    def wrapper(*args, **kwargs):
        try:
            import gurobi_optimods  # NOQA
        except ImportError:
            raise unittest.SkipTest("gurobi_optimods is not available.")
        return f(*args, **kwargs)
    return wrapper


def _init_pbar(total, unit, no_tqdm):
    """Initializes and returns a tqdm progress bar."""
    pbar = tqdm(total=total, unit=unit, ncols=80, ascii=True,
                file=sys.stdout, disable=no_tqdm)
    pbar.update(0)
    return pbar


def _update_pbar(pbar, current, total):
    """Updates and closes the progress bar."""
    perc = np.round(min((current / total) * 100, 100), 2)
    if pbar.total is not None:  # Check if pbar is still valid
        last_pc = pbar.n / pbar.total * 100  # Get current percentage based on updated value
    else:
        last_pc = 0

    perc_diff = perc - last_pc
    if perc_diff >= 1:
        pbar.update(perc_diff)

    # Ensure pbar finishes at 100% and closes
    if pbar.n < pbar.total:  # Check if it's not already at total
        pbar.update(pbar.total - pbar.n)  # Update remaining
    pbar.close()


def ams_params_not_in_andes(mdl_name, am_params):
    """
    Helper function to return parameters not in the ANDES model.
    If the model is not in the ANDES system, it returns an empty list.

    Parameters
    ----------
    mdl_name : str
        The name of the model.
    am_params : list
        A list of parameters from the AMS model.

    Returns
    -------
    list
        A list of parameters that are not in the ANDES model.
    """
    if mdl_name not in ad_models:
        return []
    ad_params = list(empty_adsys.models[mdl_name].params.keys())
    return list(set(am_params) - set(ad_params))


def model2df(instance, skip_empty, to_andes):
    """
    Prepare a DataFrame from the model instance for output.

    Parameters
    ----------
    instance : ams.model.Model
        The model instance to prepare.
    skip_empty : bool
        Whether to skip empty models.
    to_andes : bool
        Whether to prepare the DataFrame for ANDES.

    Returns
    -------
    pd.DataFrame
        The prepared DataFrame.
    """
    name = instance.class_name

    if skip_empty and instance.n == 0:
        return None

    if name not in ad_models and to_andes:
        return None

    instance.cache.refresh("df_in")
    df = instance.cache.df_in

    if to_andes:
        skipped_params = ams_params_not_in_andes(name, df.columns.tolist())
        df = df.drop(skipped_params, axis=1, errors='ignore')

    return df
