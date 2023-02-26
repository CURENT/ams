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
        Mapping dict between AMS system and PYPOWER.
    """
    # TODO: convert the AMS system to a PYPOWER case dict

    if not ssp.is_setup:
        logger.warning('System has not been setup. Conversion aborted.')
        return None

    key_dict = OrderedDict()

    # --- initialize ppc ---
    ppc = {"version": '2'}
    mva = ssp.config.mva  # system MVA base
    ppc["baseMVA"] = mva

    # --- bus data ---
    ssp_bus = ssp.Bus.as_df().rename(columns={'idx': 'bus'}).reset_index(drop=True)
    key_dict['Bus'] = OrderedDict(
        {ssp: ppc for ssp, ppc in enumerate(ssp_bus['bus'].tolist(), start=1)})

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
    ppc_load = pd.merge(ssp_bus,
                        ssp_pq[['bus', 'p0', 'q0']].rename(columns={'p0': 'Pd', 'q0': 'Qd'}),
                        on='bus', how='left').fillna(0)
    ppc_bus[['Pd', 'Qd']] = ppc_load[['Pd', 'Qd']]

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
    # bus type
    bus_type = pd.DataFrame(columns=['bus', 'type'])
    bus_type['bus'] = key_dict['PV'].keys()
    bus_type['type'] = 2
    bus_type = pd.concat([bus_type,
                        pd.DataFrame({'bus': key_dict['Slack'].keys(), 'type': 3})],
                        axis=0, ignore_index=True)
    # define bus type
    ppc_bus = pd.merge(ppc_bus, bus_type.rename(columns={'bus': 'bus_i'}),
                on='bus_i', how='left', suffixes=('_x', '_y'))
    ppc_bus['type'] = ppc_bus['type_y'].fillna(1).astype(int)
    ppc_bus.drop(columns=['type_x', 'type_y'], inplace=True)
    ppc_bus = ppc_bus.reindex(columns=bus_cols)

    # data that needs to be converted
    dcols_gen = OrderedDict([
        ('Pg', 'p0'), ('Qg', 'q0'), ('Qmax', 'qmax'), ('Qmin', 'qmin'),
        ('Pmax', 'pmax'), ('Pmin', 'pmin'), ('ramp_agc', 'Ragc'),
        ('ramp_10', 'R10'), ('ramp_30', 'R30'), ('ramp_q', 'Rq')
    ])
    scols_gen = ['Pc1', 'Pc2', 'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max', 'apf']
    ppc_gen[list(dcols_gen.keys())] = gen_df[list(dcols_gen.values())].mul(mva)
    ppc_gen[scols_gen] = gen_df[scols_gen].mul(mva)
    ppc_gen['Vg'] = gen_df['v0']

    ppc["bus"] = ppc_bus.values
    ppc["gen"] = ppc_gen.values

    # rest of the gen data
    ppc_gen[['mBase', 'status', 'Vg']] = gen_df[['Sn', 'u', 'v0']]

    # --- branch data ---
    line_df = ssp.Line.as_df()
    key_dict['Line'] = OrderedDict(
        {ssp: ppc for ssp, ppc in enumerate(line_df['idx'].tolist(), start=1)})

    line_cols = ['fbus', 'tbus', 'r', 'x', 'b', 'rateA', 'rateB', 'rateC',
                    'ratio', 'angle', 'status', 'angmin', 'angmax']
    ppc_line = pd.DataFrame(columns=line_cols)
    dcols_line = OrderedDict([
        ('fbus', 'bus1'), ('tbus', 'bus2'),
        ('r', 'r'), ('x', 'x'), ('b', 'b'), ('ratio', 'tap'),
        ('status', 'u'), ('rateA', 'rate_a'),
        ('rateB', 'rate_b'), ('rateC', 'rate_c'),
    ])
    ppc_line[list(dcols_line.keys())] = line_df[list(dcols_line.values())]
    ppc_line[['angmin', 'angmax', 'angle']] = line_df[['amin', 'amax', 'phi']] * rad2deg
    ppc_line[['fbus', 'tbus']] = ppc_line[['fbus', 'tbus']].replace(key_dict['Bus'])

    # -- area data --
    area_df = ssp.Area.as_df()
    key_dict['Area'] = OrderedDict(
        {ssp: ppc for ssp, ppc in enumerate(area_df['idx'].tolist(), start=1)})


    # gencost

    # --- output ---

    return ppc, key_dict, bus_type, ppc_bus, ppc_gen, ppc_line
