"""
Interface to PyPower
"""
import logging

from numpy import array  # NOQA
import pandas as pd

logger = logging.getLogger(__name__)


def load_ppc(case) -> dict:
    """
    Load PyPower case file into a dict.

    Parameters
    ----------
    case : str
        The path to the PyPower case file.

    Returns
    -------
    ppc : dict
        The PyPower case dict.
    """
    exec(open(f"{case}").read())
    # NOTE: the following line is not robust
    func_name = case.split('/')[-1].rstrip('.py')
    ppc = eval(f"{func_name}()")
    source_type = 'ppc'
    return ppc


def to_ppc(ssp) -> dict:
    """
    Convert the AMS system to a PyPower case dict.

    Parameters
    ----------
    ssp : ams.system
        The AMS system.

    Returns
    -------
    ppc : dict
        The PyPower case dict.
    """
    # TODO: convert the AMS system to a PyPower case dict

    if not ssp.is_setup:
        logger.warning('System has not been setup. Conversion aborted.')
        return None

    # --- output ppc ---
    ppc = {"version": '2'}

    ## system MVA base
    ppc["baseMVA"] = ssp.config.mva

    ## bus data
    ssp_bus = ssp.Bus.as_df()
    ssp_pq = ssp.PQ.as_df()
    ssp_pq[['p0', 'q0']] = ssp_pq[['p0', 'q0']] * ssp.config.mva
    ssp_pq['type'] = 1

    # NOTE: bus data: bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    # NOTE: bus type, 1 = PQ, 2 = PV, 3 = ref, 4 = isolated
    bus_cols = ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area', 'Vm', 'Va',
                'baseKV', 'zone', 'Vmax', 'Vmin']
    ppc_bus = pd.DataFrame(columns=bus_cols)

    ppc_bus['bus_i'] = ssp_bus['idx']  # TODO: will run into error if idx type is string
    ppc_bus['type'] = 4

    ppc_load = pd.merge(ssp_bus[['idx']].rename(columns={'idx': 'bus'}), 
            ssp_pq[['bus', 'p0', 'q0', 'type']].rename(columns={'p0': 'Pd', 'q0': 'Qd'}), 
            on='bus', how='left').fillna(0)
    ppc_bus['Pd'] = ppc_load['Pd']
    ppc_bus['Qd'] = ppc_load['Qd']

    ppc_bus['Gs'] = 0
    ppc_bus['Bs'] = 0
    ppc_bus['area'] = 1
    ppc_bus['Vm'] = ssp_bus['v0']
    ppc_bus['Va'] = ssp_bus['a0']
    ppc_bus['baseKV'] = ssp_bus['Vn']
    ppc_bus['zone'] = ssp_bus['owner']
    ppc_bus['Vmax'] = ssp_bus['vmax']
    ppc_bus['Vmin'] = ssp_bus['vmin']
    ppc["bus"] = ppc_bus.values



    return ppc
