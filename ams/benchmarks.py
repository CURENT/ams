"""
Benchmark functions.
"""

import datetime
import sys
import importlib.metadata as importlib_metadata
import logging

try:
    import pandapower as pdp
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False
    logging.warning("pandapower is not available. Some functionalities will be disabled.")

from andes.utils.misc import elapsed

import ams

logger = logging.getLogger(__name__)

_failed_time = -1
_failed_obj = -1

cols_pre = ['ams_mats', 'ams_parse', 'ams_eval', 'ams_final', 'ams_postinit']


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
        tools = ['ltbams', 'andes', 'cvxpy',
                 'gurobipy', 'mosek', 'piqp',
                 'pandapower', 'numba']

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


def time_routine_solve(system, routine='DCOPF', **kwargs):
    """
    Run the specified routine with the given solver and method.

    Parameters
    ----------
    system : ams.System
        The system object containing the routine.
    routine : str, optional
        The name of the routine to run. Defaults to 'DCOPF'.

    Other Parameters
    ----------------
    solver : str, optional
        The solver to use.
    ignore_dpp : bool, optional
        Whether to ignore DPP. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing the elapsed time (s) and the objective value ($).
    """
    rtn = system.routines[routine]
    solver = kwargs.get('solver', None)
    try:
        t, _ = elapsed()
        rtn.run(**kwargs)
        _, s0 = elapsed(t)
        elapsed_time = float(s0.split(' ')[0])
        obj_value = rtn.obj.v
    except Exception as e:
        logger.error(f"Error running routine {routine} with solver {solver}: {e}")
        elapsed_time = _failed_time
        obj_value = _failed_obj
    return elapsed_time, obj_value


def pre_solve(system, routine):
    """
    Time the routine preparation process.

    Parameters
    ----------
    system : ams.System
        The system object containing the routine.
    routine : str
        The name of the routine to prepare

    Returns
    -------
    dict
        A dictionary containing the preparation times in seconds for each step:
        'mats', 'parse', 'evaluate', 'finalize', 'postinit'.
    """
    rtn = system.routines[routine]

    # Initialize AMS
    # --- matrices build ---
    t_mats, _ = elapsed()
    system.mats.build(force=True)
    _, s_mats = elapsed(t_mats)

    # --- code generation ---
    t_parse, _ = elapsed()
    rtn.om.parse(force=True)
    _, s_parse = elapsed(t_parse)

    # --- code evaluation ---
    t_evaluate, _ = elapsed()
    rtn.om.evaluate(force=True)
    _, s_evaluate = elapsed(t_evaluate)

    # --- problem finalization ---
    t_finalize, _ = elapsed()
    rtn.om.finalize(force=True)
    _, s_finalize = elapsed(t_finalize)

    # --- rest init process ---
    t_postinit, _ = elapsed()
    rtn.init()
    _, s_postinit = elapsed(t_postinit)

    s_float = [float(s.split(' ')[0]) for s in [s_mats, s_parse, s_evaluate, s_finalize, s_postinit]]

    pre_time = dict(zip(cols_pre, s_float))
    return pre_time


def time_pdp_dcopf(ppn):
    """
    Test the execution time of DCOPF using pandapower.

    Parameters
    ----------
    ppn : pandapowerNet
        The pandapower network object.

    Returns
    -------
    tuple
        A tuple containing the elapsed time (s) and the objective value ($).
    """
    try:
        t_pdp, _ = elapsed()
        pdp.rundcopp(ppn)
        _, s_pdp = elapsed(t_pdp)
        elapsed_time = float(s_pdp.split(' ')[0])
        obj_value = ppn.res_cost
    except Exception as e:
        logger.error(f"Error running pandapower: {e}")
        elapsed_time = _failed_time
        obj_value = _failed_obj
    return elapsed_time, obj_value


def time_routine(system, routine='DCOPF', solvers=['CLARABEL'],
                 **kwargs):
    """
    Time the specified routine with the given solvers.

    Parameters
    ----------
    system : ams.System
        The system object containing the routine.
    routine : str, optional
        The name of the routine to run. Defaults to 'DCOPF'.
    solvers : list of str, optional
        List of solvers to use. Defaults to ['CLARABEL'].

    Other Parameters
    ----------------
    ignore_dpp : bool, optional
        Whether to ignore DPP. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing the preparation times and the solution times in
        seconds for each solver.
    """
    pre_time = pre_solve(system, routine)
    sol = {f'{solver}': {'time': 0, 'obj': 0} for solver in solvers}

    for solver in solvers:
        if solver != 'pandapower':
            s, obj = time_routine_solve(system, routine, solver=solver, **kwargs)
            sol[solver]['time'] = s
            sol[solver]['obj'] = obj
        elif solver == 'pandapower' and PANDAPOWER_AVAILABLE and routine == 'DCOPF':
            ppc = ams.io.pypower.system2ppc(system)
            ppn = pdp.converter.from_ppc(ppc, f_hz=system.config.freq)
            s, obj = time_pdp_dcopf(ppn)
            sol[solver]['time'] = s
            sol[solver]['obj'] = obj
        else:
            sol[solver]['time'] = _failed_time
            sol[solver]['obj'] = _failed_obj

    return pre_time, sol


def time_dcopf_with_lf(system, solvers=['CLARABEL'], load_factors=[1], ignore_dpp=False):
    """
    Time the execution of DCOPF with varying load factors.

    Parameters
    ----------
    system : ams.System
        The system object containing the routine.
    solvers : list of str, optional
        List of solvers to use. Defaults to ['CLARABEL'].
    load_factors : list of float, optional
        List of load factors to apply. Defaults to None.
    ignore_dpp : bool, optional
        Whether to ignore DPP.

    Returns
    -------
    tuple
        A tuple containing the list of times and the list of objective values.
    """
    pre_time = pre_solve(system, 'DCOPF')
    sol = {f'{solver}': {'time': 0, 'obj': 0} for solver in solvers}

    pd0 = system.PQ.p0.v.copy()
    pq_idx = system.PQ.idx.v

    for solver in solvers:
        if solver != 'pandapower':
            obj_all = 0
            t_all, _ = elapsed()
            for lf_k in load_factors:
                system.PQ.set(src='p0', attr='v', idx=pq_idx, value=lf_k * pd0)
                system.DCOPF.update(params=['pd'])
                _, obj = time_routine_solve(system, 'DCOPF',
                                            solver=solver, ignore_dpp=ignore_dpp)
                obj_all += obj
            _, s_all = elapsed(t_all)
            system.PQ.set(src='p0', attr='v', idx=pq_idx, value=pd0)
            s = float(s_all.split(' ')[0])
            sol[solver]['time'] = s
            sol[solver]['obj'] = obj_all
        elif solver == 'pandapower' and PANDAPOWER_AVAILABLE:
            ppc = ams.io.pypower.system2ppc(system)
            ppn = pdp.converter.from_ppc(ppc, f_hz=system.config.freq)
            p_mw0 = ppn.load['p_mw'].copy()
            t_all, _ = elapsed()
            obj_all = 0
            for lf_k in load_factors:
                ppn.load['p_mw'] = lf_k * p_mw0
                _, obj = time_pdp_dcopf(ppn)
                obj_all += obj
            _, s_all = elapsed(t_all)
            s = float(s_all.split(' ')[0])
            sol[solver]['time'] = s
            sol[solver]['obj'] = obj_all
        else:
            sol[solver]['time'] = _failed_time
            sol[solver]['obj'] = _failed_obj

    return pre_time, sol
