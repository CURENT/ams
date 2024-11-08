"""
Benchmark functions.
"""

import datetime
import sys
import importlib_metadata
import logging

import numpy as np

try:
    import pandapower as pdp
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False
    logging.warning("pandapower is not available. Some functionalities will be disabled.")

from andes.utils.misc import elapsed

import ams

logger = logging.getLogger(__name__)

_failed_time = '-1 seconds'
_failed_obj = -1

cols_time = ['ams_mats', 'ams_parse', 'ams_eval', 'ams_final',
             'ams_postinit', 'ams_grb', 'ams_mosek', 'ams_piqp', 'pdp']
cols_obj = ['grb', 'mosek', 'piqp', 'pdp']


def get_tool_versions(tools=None):
    """
    Get the current time, Python version, and versions of specified tools.

    Parameters
    ----------
    tools : list of str, optional
        List of tool names to check versions. If None, a default list of tools is used.

    Returns
    -------
    dict
        A dictionary containing the tool names and their versions.
    """
    if tools is None:
        tools = ['ltbams', 'andes',
                 'cvxpy', 'pandapower',
                 'PYPOWER', 'gurobipy', 'mosek',
                 'piqp', 'numba']

    # Get current time and Python version
    last_run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    python_version = sys.version

    # Collect tool versions
    tool_versions = {}
    for tool in tools:
        try:
            version = importlib_metadata.version(tool)
            tool_versions[tool] = version
        except importlib_metadata.PackageNotFoundError:
            logger.error(f"Package {tool} not found.")
            tool_versions[tool] = "Not installed"

    # Print the results in a formatted way
    logger.warning(f"Last run time: {last_run_time}")
    logger.warning(f"Python: {python_version}\n")

    # Calculate the width of the columns
    max_tool_length = max(len(tool) for tool in tool_versions)
    max_version_length = max(len(version) for version in tool_versions.values())

    # Print the header
    logger.warning(f"{'Tool':<{max_tool_length}}  {'Version':<{max_version_length}}")
    logger.warning(f"{'-' * max_tool_length}  {'-' * max_version_length}")

    # Print each tool and its version
    for tool, version in tool_versions.items():
        logger.warning(f"{tool:<{max_tool_length}}  {version:<{max_version_length}}")

    return tool_versions


def run_routine(system, routine='DCOPF', ignore_dpp=True, solver='GUROBI', method=None):
    """
    Run the specified routine with the given solver and method.

    Parameters
    ----------
    system : ams.System
        The system object containing the routine.
    routine : str, optional
        The name of the routine to run. Defaults to 'DCOPF'.
    ignore_dpp : bool, optional
        Whether to ignore DPP. Defaults to True.
    solver : str, optional
        The solver to use. Defaults to 'GUROBI'.
    method : function, optional
        A custom solve method to use. Defaults to None.

    Returns
    -------
    tuple
        A tuple containing the elapsed time and the objective value.
    """
    rtn = system.routines[routine]
    try:
        t_start, _ = elapsed()
        if method:
            rtn.run(solver=solver, reoptimize=True, Method=method, ignore_dpp=ignore_dpp)
        else:
            rtn.run(solver=solver, ignore_dpp=ignore_dpp)
        _, elapsed_time = elapsed(t_start)
        obj_value = rtn.obj.v
    except Exception as e:
        logger.error(f"Error running routine {routine} with solver {solver}: {e}")
        elapsed_time = _failed_time
        obj_value = _failed_obj
    return elapsed_time, obj_value


def test_time(case, routine='DCOPF', ignore_dpp=True):
    """
    Test the execution time of the specified routine on the given case.

    Parameters
    ----------
    case : str
        The path to the case file.
    routine : str, optional
        The name of the routine to test. Defaults to 'DCOPF'.
    ignore_dpp : bool, optional
        Whether to ignore DPP. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing the list of times and the list of objective values.
    """
    sp = ams.load(case, setup=True, default_config=True, no_output=True)

    # NOTE: the line flow limits are relaxed for the large cases
    # otherwise the DCOPF will fail in pandapower and MATPOWER
    if sp.Bus.n > 4000:
        sp.Line.alter(src='rate_a', idx=sp.Line.idx.v, value=99999)

    rtn = sp.routines[routine]

    # Initialize AMS
    # --- matrices build ---
    t_mats, _ = elapsed()
    sp.mats.build()
    _, s_mats = elapsed(t_mats)

    # --- code generation ---
    t_parse, _ = elapsed()
    rtn.om.parse()
    _, s_parse = elapsed(t_parse)

    # --- code evaluation ---
    t_evaluate, _ = elapsed()
    rtn.om.evaluate()
    _, s_evaluate = elapsed(t_evaluate)

    # --- problem finalization ---
    t_finalize, _ = elapsed()
    rtn.om.finalize()
    _, s_finalize = elapsed(t_finalize)

    # --- rest init process ---
    t_postinit, _ = elapsed()
    rtn.init()
    _, s_postinit = elapsed(t_postinit)

    # --- run solvers ---
    s_ams_grb, obj_grb = run_routine(sp, routine, ignore_dpp, 'GUROBI')
    s_ams_mosek, obj_mosek = run_routine(sp, routine, ignore_dpp, 'MOSEK')
    s_ams_piqp, obj_piqp = run_routine(sp, routine, ignore_dpp, 'PIQP')

    if PANDAPOWER_AVAILABLE:
        # Convert to PYPOWER format
        ppc = ams.io.pypower.system2ppc(sp)
        freq = sp.config.freq
        ppn = pdp.converter.from_ppc(ppc, f_hz=freq)

        del sp

        try:
            t_pdp, _ = elapsed()
            pdp.rundcopp(ppn)
            _, s_pdp = elapsed(t_pdp)
            obj_pdp = ppn.res_cost
        except Exception as e:
            logger.error(f"Error running pandapower: {e}")
            s_pdp = _failed_time
            obj_pdp = _failed_obj
    else:
        s_pdp = _failed_time
        obj_pdp = _failed_obj

    time = [s_mats, s_parse, s_evaluate, s_finalize, s_postinit,
            s_ams_grb, s_ams_mosek, s_ams_piqp, s_pdp]
    time = [float(t.split(' ')[0]) for t in time]
    obj = [obj_grb, obj_mosek, obj_piqp, obj_pdp]

    return time, obj


def run_dcopf_with_load_factors(sp, solver, method=None, load_factors=None, ignore_dpp=False):
    """
    Run the specified solver with varying load factors.

    Parameters
    ----------
    sp : ams.System
        The system object containing the routine.
    solver : str
        The name of the solver to use.
    method : function, optional
        A custom solve method to use. Defaults to None.
    load_factors : list of float, optional
        List of load factors to apply. Defaults to None.

    Returns
    -------
    tuple
        A tuple containing the elapsed time and the cumulative objective value.
    """
    if load_factors is None:
        load_factors = []

    obj_value = 0
    try:
        t_start, _ = elapsed()
        pq_idx = sp.PQ.idx.v
        pd0 = sp.PQ.get(src='p0', attr='v', idx=pq_idx).copy()
        for lf_k in load_factors:
            sp.PQ.set(src='p0', attr='v', idx=pq_idx, value=lf_k * pd0)
            sp.DCOPF.update(params=['pd'])
            if method:
                sp.DCOPF.run(solver=solver, reoptimize=True, Method=method, ignore_dpp=ignore_dpp)
            else:
                sp.DCOPF.run(solver=solver, ignore_dpp=ignore_dpp)
            obj_value += sp.DCOPF.obj.v
        _, elapsed_time = elapsed(t_start)
    except Exception as e:
        logger.error(f"Error running solver {solver} with load factors: {e}")
        elapsed_time = _failed_time
        obj_value = _failed_obj
    return elapsed_time, obj_value


def test_mtime(case, load_factors, ignore_dpp=True):
    """
    Test the execution time of the specified routine on the given case with varying load factors.

    Parameters
    ----------
    case : str
        The path to the case file.
    load_factors : list of float
        List of load factors to apply.

    Returns
    -------
    tuple
        A tuple containing the list of times and the list of objective values.
    """
    sp = ams.load(case, setup=True, default_config=True, no_output=True)

    # Record original load
    pq_idx = sp.PQ.idx.v
    pd0 = sp.PQ.get(src='p0', attr='v', idx=pq_idx).copy()

    # Initialize AMS
    # --- matrices build ---
    t_mats, _ = elapsed()
    sp.mats.build()
    _, s_mats = elapsed(t_mats)

    # --- code generation ---
    t_parse, _ = elapsed()
    sp.DCOPF.om.parse()
    _, s_parse = elapsed(t_parse)

    # --- code evaluation ---
    t_evaluate, _ = elapsed()
    sp.DCOPF.om.evaluate()
    _, s_evaluate = elapsed(t_evaluate)

    # --- problem finalization ---
    t_finalize, _ = elapsed()
    sp.DCOPF.om.finalize()
    _, s_finalize = elapsed(t_finalize)

    # --- rest init process ---
    t_postinit, _ = elapsed()
    sp.DCOPF.init()
    _, s_postinit = elapsed(t_postinit)

    # Run solvers with load factors
    s_ams_grb, obj_grb = run_dcopf_with_load_factors(
        sp, 'GUROBI', method=3, load_factors=load_factors, ignore_dpp=ignore_dpp)
    sp.PQ.set(src='p0', attr='v', idx=pq_idx, value=pd0)  # Reset the load in AMS

    s_ams_mosek, obj_mosek = run_dcopf_with_load_factors(
        sp, 'MOSEK', load_factors=load_factors, ignore_dpp=ignore_dpp)
    sp.PQ.set(src='p0', attr='v', idx=pq_idx, value=pd0)

    s_ams_piqp, obj_piqp = run_dcopf_with_load_factors(
        sp, 'PIQP', load_factors=load_factors, ignore_dpp=ignore_dpp)
    sp.PQ.set(src='p0', attr='v', idx=pq_idx, value=pd0)

    if PANDAPOWER_AVAILABLE:
        # --- PANDAPOWER ---
        ppc = ams.io.pypower.system2ppc(sp)
        freq = sp.config.freq

        del sp

        ppc_pd0 = ppc['bus'][:, 2].copy()

        ppn = pdp.converter.from_ppc(ppc, f_hz=freq)
        obj_pdp = 0
        t_pdp_series = ['0 seconds'] * len(load_factors)
        try:
            for i, lf_k in enumerate(load_factors):
                ppc['bus'][:, 2] = lf_k * ppc_pd0
                ppn = pdp.converter.from_ppc(ppc, f_hz=freq)
                t0_pdp, _ = elapsed()
                pdp.rundcopp(ppn)
                obj_pdp += ppn.res_cost
                _, t_pdp_series[i] = elapsed(t0_pdp)
            t_pdp_series = [float(t.split(' ')[0]) for t in t_pdp_series]
            s_pdp = f'{np.sum(t_pdp_series):.4f} seconds'
        except Exception:
            s_pdp = _failed_time
            obj_pdp = _failed_obj
    else:
        s_pdp = _failed_time
        obj_pdp = _failed_obj

    time = [s_mats, s_parse, s_evaluate, s_finalize, s_postinit,
            s_ams_grb, s_ams_mosek, s_ams_piqp, s_pdp]
    time = [float(t.split(' ')[0]) for t in time]
    obj = [obj_grb, obj_mosek, obj_piqp, obj_pdp]

    return time, obj
