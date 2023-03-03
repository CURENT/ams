"""
Interface to PYPOWER
"""
import logging

from collections import OrderedDict  # NOQA

from numpy import array  # NOQA
import pandas as pd  # NOQA

from andes.shared import rad2deg  # NOQA

logger = logging.getLogger(__name__)


def load_ppc(case) -> dict:
    """
    Load PYPOWER case file into a dict.

    Parameters
    ----------
    case : str
        The path to the PYPOWER case file.

    Returns
    -------
    ppc : dict
        The PYPOWER case dict.
    """
    exec(open(f"{case}").read())
    # NOTE: the following line is not robust
    func_name = case.split('/')[-1].rstrip('.py')
    ppc = eval(f"{func_name}()")
    source_type = 'ppc'
    return ppc


def to_ppc(ssp) -> dict:
    """
    Convert the AMS system to a PYPOWER case dict.

    Parameters
    ----------
    ssp : ams.system
        The AMS system.

    Returns
    -------
    ppc : dict
        The PYPOWER case dict.
    key_dict : OrderedDict
        Mapping dict from AMS system to PYPOWER case.
    col_dict : OrderedDict
        Dict of columns of PYPOWER case.
    """
    # TODO: convert the AMS system to a PYPOWER case dict

    if not ssp.is_setup:
        logger.warning('System has not been setup. Conversion aborted.')
        return None

    key_dict = OrderedDict()
    col_dict = OrderedDict()

    # --- initialize ppc ---
    ppc = {"version": '2'}
    mva = ssp.config.mva  # system MVA base
    ppc["baseMVA"] = mva

    # --- bus data ---
    bus_df = ssp.Bus.as_df().rename(columns={'idx': 'bus'}).reset_index(drop=True)
    key_dict['Bus'] = OrderedDict(
        {ssp: ppc for ssp, ppc in enumerate(bus_df['bus'].tolist(), start=1)})

    # NOTE: bus data: bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    # NOTE: bus type, 1 = PQ, 2 = PV, 3 = ref, 4 = isolated
    bus_cols = ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area', 'Vm', 'Va',
                'baseKV', 'zone', 'Vmax', 'Vmin']
    ppc_bus = pd.DataFrame(columns=bus_cols)

    ppc_bus['bus_i'] = key_dict['Bus'].values()
    ppc_bus['type'] = 1  # default to PQ bus
    # TODO: add check for isolated buses

    # load data
    ssp_pq = ssp.PQ.as_df()
    ssp_pq[['p0', 'q0']] = ssp_pq[['p0', 'q0']].mul(mva)
    ppc_load = pd.merge(bus_df,
                        ssp_pq[['bus', 'p0', 'q0']].rename(columns={'p0': 'Pd', 'q0': 'Qd'}),
                        on='bus', how='left').fillna(0)
    ppc_bus[['Pd', 'Qd']] = ppc_load[['Pd', 'Qd']]

    # shunt data
    ssp_shunt = ssp.Shunt.as_df()
    ssp_shunt['g'] = ssp_shunt['g'] * ssp_shunt['u']
    ssp_shunt['b'] = ssp_shunt['b'] * ssp_shunt['u']
    ssp_shunt[['g', 'b']] = ssp_shunt[['g', 'b']] * mva
    ppc_y = pd.merge(bus_df,
                     ssp_shunt[['bus', 'g', 'b']].rename(columns={'g': 'Gs', 'b': 'Bs'}),
                     on='bus', how='left').fillna(0)
    ppc_bus[['Gs', 'Bs']] = ppc_y[['Gs', 'Bs']]

    # rest of the bus data
    ppc_bus_cols = ['area', 'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin']
    bus_df_cols = ['area', 'v0', 'a0', 'Vn', 'zone', 'vmax', 'vmin']
    ppc_bus[ppc_bus_cols] = bus_df[bus_df_cols]
    if (ppc_bus['baseKV'] == 0).any():
        logger.warning('PPC Bus baseKV contains 0. Replaced with system mva.')
        ppc_bus['baseKV'].replace(0, mva, inplace=True)

    # --- generator data ---
    pv_df = ssp.PV.as_df()
    slack_df = ssp.Slack.as_df()
    gen_df = pd.concat([pv_df, slack_df], ignore_index=True)
    gen_df = pd.merge(left=gen_df, right=bus_df[['bus', 'area']],
                      on='bus', how='left',)
    key_dict['Slack'] = OrderedDict(
        {ssp: ppc for ssp, ppc in enumerate(slack_df['idx'].tolist(), start=1)})
    key_dict['PV'] = OrderedDict(
        {ssp: ppc for ssp, ppc in enumerate(pv_df['idx'].tolist(), start=1)})

    # NOTE: gen data:
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    gen_cols = ['bus', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status',
                'Pmax', 'Pmin', 'Pc1', 'Pc2',
                'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max',
                'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q', 'apf']
    ppc_gen = pd.DataFrame(columns=gen_cols)

    # idx of bus in ppc
    gen_bus_ppc = [key_dict['Bus'][bus_idx] for bus_idx in gen_df['bus'].tolist()]
    ppc_gen['bus'] = gen_bus_ppc
    # define bus type
    type_pv = ppc_bus['bus_i'].isin(pv_df['bus']).astype(int)
    type_slack = ppc_bus['bus_i'].isin(slack_df['bus']).astype(int) * 2
    ppc_bus['type'] = ppc_bus['type'] + type_pv + type_slack

    dcols_gen = OrderedDict([
        ('Pg', 'p0'), ('Qg', 'q0'), ('Qmax', 'qmax'), ('Qmin', 'qmin'),
        ('Pmax', 'pmax'), ('Pmin', 'pmin'), ('ramp_agc', 'Ragc'),
        ('ramp_10', 'R10'), ('ramp_30', 'R30'), ('ramp_q', 'Rq'),
        ('Pc1', 'Pc1'), ('Pc2', 'Pc2'), ('Qc1min', 'Qc1min'), ('Qc1max', 'Qc1max'),
        ('Qc2min', 'Qc2min'), ('Qc2max', 'Qc2max'),
    ])
    ppc_gen[list(dcols_gen.keys())] = gen_df[list(dcols_gen.values())]

    # rest of the gen data
    ppc_gen[['mBase', 'status', 'Vg', 'apf']] = gen_df[['Sn', 'u', 'v0', 'apf']]

    # data type
    ppc_bus[['bus_i', 'type', 'zone']] = ppc_bus[['bus_i', 'type', 'zone']].astype(int)
    ppc_gen[['bus', 'status']] = ppc_gen[['bus', 'status']].astype(int)
    ppc["bus"] = ppc_bus.values
    ppc["gen"] = ppc_gen.values

    # --- branch data ---
    line_df = ssp.Line.as_df()
    key_dict['Line'] = OrderedDict(
        {ssp: ppc for ssp, ppc in enumerate(line_df['idx'].tolist(), start=1)})

    branch_cols = ['fbus', 'tbus', 'r', 'x', 'b', 'rateA', 'rateB', 'rateC',
                   'ratio', 'angle', 'status', 'angmin', 'angmax']
    ppc_line = pd.DataFrame(columns=branch_cols)
    dcols_line = OrderedDict([
        ('fbus', 'bus1'), ('tbus', 'bus2'),
        ('r', 'r'), ('x', 'x'), ('b', 'b'), ('ratio', 'tap'),
        ('status', 'u'), ('rateA', 'rate_a'),
        ('rateB', 'rate_b'), ('rateC', 'rate_c'),
    ])
    ppc_line[list(dcols_line.keys())] = line_df[list(dcols_line.values())]
    ppc_line[['angmin', 'angmax', 'angle']] = line_df[['amin', 'amax', 'phi']] * rad2deg
    ppc_line[['fbus', 'tbus']] = ppc_line[['fbus', 'tbus']].replace(key_dict['Bus'])
    ppc["branch"] = ppc_line.values

    # --- area data ---
    # area_df = ssp.Area.as_df()
    # key_dict['Area'] = OrderedDict(
    #     {ssp: ppc for ssp, ppc in enumerate(area_df['idx'].tolist(), start=1)})
    # ppc['areas'] = array([list(key_dict['Area'].values())])

    # --- gencost data ---
    gc_df = ssp.GCost.as_df()
    key_dict['GCost'] = OrderedDict(
        {ssp: ppc for ssp, ppc in enumerate(gc_df['idx'].tolist(), start=1)})

    # TODO: Add Model 1
    gcost_cols = ['type', 'startup', 'shutdown', 'n', 'c2', 'c1', 'c0']
    ppc_gcost = pd.DataFrame(columns=gcost_cols)
    gc_df = pd.merge(left=gen_df[['idx']].astype(str).rename(columns={'idx': 'gen'}),
                     right=gc_df, on='gen', how='left')
    dcols_cost = gcost_cols.copy()
    dcols_cost.remove('n')
    ppc_gcost[dcols_cost] = gc_df[dcols_cost]
    ppc_gcost['n'] = 3
    ppc["gencost"] = ppc_gcost.values

    col_dict['bus'] = bus_cols
    col_dict['gen'] = gen_cols
    col_dict['branch'] = branch_cols

    return ppc, key_dict, col_dict
