"""
Interface to PyPower
"""
import logging

from collections import OrderedDict  # NOQA

from numpy import array  # NOQA
import pandas as pd  # NOQA

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

    idx = OrderedDict()

    # --- initialize ppc ---
    ppc = {"version": '2'}
    mva = ssp.config.mva  # system MVA base
    ppc["baseMVA"] = mva

    # --- bus data ---
    ssp_bus = ssp.Bus.as_df().rename(columns={'idx': 'bus'}).reset_index(drop=True)
    idx['bus'] = {ssp: ppc for ssp, ppc in enumerate(ssp_bus['bus'].tolist())}

    # NOTE: bus data: bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    # NOTE: bus type, 1 = PQ, 2 = PV, 3 = ref, 4 = isolated
    bus_cols = ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area', 'Vm', 'Va',
                'baseKV', 'zone', 'Vmax', 'Vmin']
    ppc_bus = pd.DataFrame(columns=bus_cols)

    ppc_bus['bus_i'] = idx['bus'].values()
    ppc_bus['type'] = 4

    # load data
    ssp_pq = ssp.PQ.as_df()
    ssp_pq[['p0', 'q0']] = ssp_pq[['p0', 'q0']].mul(mva)
    ssp_pq['type'] = 1
    ppc_load = pd.merge(ssp_bus,
                        ssp_pq[['bus', 'p0', 'q0', 'type']].rename(columns={'p0': 'Pd', 'q0': 'Qd'}),
                        on='bus', how='left').fillna(0)
    ppc_bus[['type', 'Pd', 'Qd']] = ppc_load[['type', 'Pd', 'Qd']]

    # shunt data
    ssp_shunt = ssp.Shunt.as_df()
    ssp_shunt['g'] = ssp_shunt['g'] * ssp_shunt['u']
    ssp_shunt['b'] = ssp_shunt['b'] * ssp_shunt['u']
    ssp_shunt[['g', 'b']] = ssp_shunt[['g', 'b']] * mva
    ppc_y = pd.merge(ssp_bus,
                     ssp_shunt[['bus', 'g', 'b']].rename(columns={'g': 'Gs', 'b': 'Bs'}),
                     on='bus', how='left').fillna(0)
    ppc_bus[['Gs', 'Bs']] = ppc_y[['Gs', 'Bs']]

    # rest of the bus data
    ppc_bus_cols = ['area', 'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin']
    ssp_bus_cols = ['area', 'v0', 'a0', 'Vn', 'owner', 'vmax', 'vmin']
    ppc_bus[ppc_bus_cols] = ssp_bus[ssp_bus_cols]

    # --- generator data ---
    pv_df = ssp.PV.as_df()
    slack_df = ssp.Slack.as_df()
    gen_df = pd.concat([pv_df, slack_df], ignore_index=True)
    idx['gen'] = {ssp: ppc for ssp, ppc in enumerate(gen_df['idx'].tolist())}

    # NOTE: gen data:
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    gen_cols = ['bus', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status',
                'Pmax', 'Pmin', 'Pc1', 'Pc2',
                'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max',
                'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q', 'apf']
    ppc_gen = pd.DataFrame(columns=gen_cols)

    # bus idx in ppc
    gen_bus_ppc = [idx['bus'][bus_idx] for bus_idx in gen_df['bus'].tolist()]
    ppc_gen['bus'] = gen_bus_ppc
    # gen_mva_cols = OrderedDict([
    #     ('Pg', 'p0'), ('Qg', 'q0'),
    #     ('Qmax', 'qmax'), ('Qmin', 'qmin'),
    #     ('Pmax', 'pmax'), ('Pmin', 'pmin'),
    # ])
    # gen_mva_cols = ['Pg', 'Qg', 'Qmax', 'Qmin', 'Pmax', 'Pmin', 'Pc1', 'Pc2',
    #                 'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max',
    #                 'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q']

    # ppc_gen['Pg'] = gen_df['p0'] * mva
    # ppc_gen['Qg'] = gen_df['q0'] * mva
    # ppc_gen['Qmax'] = gen_df['qmax'] * mva
    # ppc_gen['Qmin'] = gen_df['qmin'] * mva
    # ppc_gen['Vg'] = gen_df['v0']

    # branch

    # areas

    # gencost

    # output bus data
    ppc["bus"] = ppc_bus.values

    return ppc, ppc_bus, idx
