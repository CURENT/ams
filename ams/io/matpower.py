"""
MATPOWER parser.
"""
import logging
import re
import numpy as np

from andes.io import read_file_like
from andes.io.xlsx import confirm_overwrite
from andes.shared import deg2rad, rad2deg

from ams import __version__ as version
from ams.shared import copyright_msg, nowarranty_msg, report_time

logger = logging.getLogger(__name__)


def testlines(infile):
    """
    Test if this file is in the MATPOWER format.

    NOT YET IMPLEMENTED.
    """

    return True  # hard coded


def read(system, file):
    """
    Read a MATPOWER data file into mpc, and build AMS device elements.
    """

    mpc = m2mpc(file)
    return mpc2system(mpc, system)


def m2mpc(infile: str) -> dict:
    """
    Parse a MATPOWER file and return a dictionary containing the parsed data.

    This function processes MATPOWER case files and extracts relevant fields
    into a structured dictionary. It is revised from ``andes.io.matpower.m2mpc``.

    Supported fields include:
    - `baseMVA`: The system base power in MVA.
    - `bus`: Bus data, including voltage, load, and generation information.
    - `bus_name`: Names of the buses (if available).
    - `gen`: Generator data, including power limits and voltage setpoints.
    - `branch`: Branch data, including line impedances and ratings.
    - `gencost`: Generator cost data (parsed but not used in this implementation).
    - `areas`: Area data (parsed but not used in this implementation).
    - `gentype`: Generator type information (if available).
    - `genfuel`: Generator fuel type information (if available).

    Parameters
    ----------
    infile : str
        Path to the MATPOWER file to be parsed.

    Returns
    -------
    dict
        A dictionary containing the parsed MATPOWER data, where keys correspond
        to MATPOWER struct names and values are numpy arrays or lists.
    """

    func = re.compile(r'function\s')
    mva = re.compile(r'\s*mpc.baseMVA\s*=\s*')
    bus = re.compile(r'\s*mpc.bus\s*=\s*\[?')
    gen = re.compile(r'\s*mpc.gen\s*=\s*\[')
    branch = re.compile(r'\s*mpc.branch\s*=\s*\[')
    area = re.compile(r'\s*mpc.areas\s*=\s*\[')
    gencost = re.compile(r'\s*mpc.gencost\s*=\s*\[')
    bus_name = re.compile(r'\s*mpc.bus_name\s*=\s*{')
    gentype = re.compile(r'\s*mpc.gentype\s*=\s*{')
    genfuel = re.compile(r'\s*mpc.genfuel\s*=\s*{')
    end = re.compile(r'\s*[\];}]')
    has_digit = re.compile(r'.*\d+\s*]?;?')

    field = None
    info = True

    mpc = {
        'version': 2,  # not in use
        'baseMVA': 100,
        'bus': [],
        'gen': [],
        'branch': [],
        'area': [],
        'gencost': [],
        'bus_name': [],
        'gentype': [],
        'genfuel': [],
    }

    input_list = read_file_like(infile)

    for line in input_list:
        line = line.strip().rstrip(';')
        if not line:
            continue
        elif func.search(line):  # skip function declaration
            continue
        elif len(line.split('%')[0]) == 0:
            if info is True:
                logger.info(line[1:])
                info = False
            else:
                continue
        elif mva.search(line):
            mpc["baseMVA"] = float(line.split('=')[1])

        if not field:
            if bus.search(line):
                field = 'bus'
            elif gen.search(line):
                field = 'gen'
            elif branch.search(line):
                field = 'branch'
            elif area.search(line):
                field = 'area'
            elif gencost.search(line):
                field = 'gencost'
            elif bus_name.search(line):
                field = 'bus_name'
            elif gentype.search(line):
                field = 'gentype'
            elif genfuel.search(line):
                field = 'genfuel'
            else:
                continue
        elif end.search(line):
            field = None
            continue

        # parse mpc sections
        if field:
            if line.find('=') >= 0:
                line = line.split('=')[1]
            if line.find('[') >= 0:
                line = re.sub(r'\[', '', line)
            elif line.find('{') >= 0:
                line = re.sub(r'{', '', line)

            if field in ['bus_name', 'gentype', 'genfuel']:
                # Handle string-based fields
                line = line.split(';')
                data = [i.strip('\'').strip() for i in line if i.strip()]
                mpc[field].extend(data)
            else:
                if not has_digit.search(line):
                    continue
                line = line.split('%')[0].strip()
                line = line.split(';')
                for item in line:
                    if not has_digit.search(item):
                        continue
                    try:
                        data = np.array([float(val) for val in item.split()])
                    except Exception as e:
                        logger.error('Error parsing "%s"', infile)
                        raise e
                    mpc[field].append(data)

    # convert mpc to np array
    mpc_array = dict()
    for key, val in mpc.items():
        if isinstance(val, (float, int)):
            mpc_array[key] = val
        elif isinstance(val, list):
            if len(val) == 0:
                continue
            if key in ['bus_name', 'gentype', 'genfuel']:
                mpc_array[key] = np.array(val, dtype=object)
            else:
                mpc_array[key] = np.array(val)
        else:
            raise NotImplementedError("Unknown type for mpc, ", type(val))
    return mpc_array


def mpc2system(mpc: dict, system) -> bool:
    """
    Load an mpc dict into an empty AMS system.

    Revised from ``andes.io.matpower.mpc2system``.

    Note that `mbase` in mpc is converted to `Sn`, but it is not actually used in
    MATPOWER nor AMS.

    In converted AMS system, StaticGen idxes are 1-based, while the sequence follow
    the order of the original MATPOWER data.

    Parameters
    ----------
    system : ams.system.System
        Empty system to load the data into.
    mpc : dict
        mpc struct names : numpy arrays

    Returns
    -------
    bool
        True if successful, False otherwise.
    """

    # list of buses with slack gen
    sw = []

    system.config.mva = base_mva = mpc['baseMVA']

    for data in mpc['bus']:
        # idx  ty   pd   qd  gs  bs  area  vmag  vang  baseKV  zone  vmax  vmin
        # 0    1    2    3   4   5   6     7     8     9       10    11    12
        idx = int(data[0])
        ty = data[1]
        if ty == 3:
            sw.append(idx)
        pd = data[2] / base_mva
        qd = data[3] / base_mva
        gs = data[4] / base_mva
        bs = data[5] / base_mva
        area = data[6]
        vmag = data[7]
        vang = data[8] * deg2rad
        baseKV = data[9]
        if baseKV == 0:
            baseKV = 110
        zone = data[10]
        vmax = data[11]
        vmin = data[12]

        system.add('Bus', idx=idx, name=None,
                   type=ty, Vn=baseKV,
                   v0=vmag, a0=vang,
                   vmax=vmax, vmin=vmin,
                   area=area, zone=zone)
        if pd != 0 or qd != 0:
            system.add('PQ', bus=idx, name=None, Vn=baseKV, p0=pd, q0=qd)
        if gs or bs:
            system.add('Shunt', bus=idx, name=None, Vn=baseKV, g=gs, b=bs)

    gen_idx = 0
    if mpc['gen'].shape[1] <= 10:  # missing data
        mpc_gen = np.zeros((mpc['gen'].shape[0], 21), dtype=np.float64)
        mpc_gen[:, :10] = mpc['gen']
        mpc_gen[:, 16] = system.PV.Ragc.default * base_mva / 60
        mpc_gen[:, 17] = system.PV.R10.default * base_mva
        mpc_gen[:, 18] = system.PV.R30.default * base_mva
        mpc_gen[:, 19] = system.PV.Rq.default * base_mva / 60
    else:
        mpc_gen = mpc['gen']

    # Ensure 'gentype' and 'genfuel' keys exist in mpc, with default values if missing
    gentype = mpc.get('gentype', [''] * mpc_gen.shape[0])
    genfuel = mpc.get('genfuel', [''] * mpc_gen.shape[0])

    # Validate lengths of 'gentype' and 'genfuel' against the number of generators
    if len(gentype) != mpc_gen.shape[0]:
        raise ValueError(
            f"'gentype' length ({len(gentype)}) does not match the number of generators ({mpc_gen.shape[0]})")
    if len(genfuel) != mpc_gen.shape[0]:
        raise ValueError(
            f"'genfuel' length ({len(genfuel)}) does not match the number of generators ({mpc_gen.shape[0]})")

    for data, gt, gf in zip(mpc_gen, gentype, genfuel):
        # bus  pg  qg  qmax  qmin  vg  mbase  status  pmax  pmin
        # 0    1   2   3     4     5   6      7       8     9
        # pc1  pc2  qc1min  qc1max  qc2min  qc2max  ramp_agc  ramp_10
        # 10   11   12      13      14      15      16        17
        # ramp_30  ramp_q  apf
        # 18       19      20
        bus_idx = int(data[0])
        gen_idx += 1
        vg = data[5]
        status = int(data[7])
        pg = data[1] / base_mva
        qg = data[2] / base_mva
        qmax = data[3] / base_mva
        qmin = data[4] / base_mva
        pmax = data[8] / base_mva
        pmin = data[9] / base_mva
        pc1 = data[10] / base_mva
        pc2 = data[11] / base_mva
        qc1min = data[12] / base_mva
        qc1max = data[13] / base_mva
        qc2min = data[14] / base_mva
        qc2max = data[15] / base_mva
        ramp_agc = 60 * data[16] / base_mva  # MW/min -> MW/h
        ramp_10 = data[17] / base_mva  # MW -> MW/h
        ramp_30 = data[18] / base_mva  # MW -> MW/h
        ramp_q = 60 * data[19] / base_mva  # MVAr/min -> MVAr/h
        apf = data[20]

        uid = system.Bus.idx2uid(bus_idx)
        vn = system.Bus.Vn.v[uid]
        a0 = system.Bus.a0.v[uid]

        if bus_idx in sw:
            system.add('Slack', idx=gen_idx, bus=bus_idx, busr=bus_idx,
                       name=None,
                       u=status, Sn=data[6],
                       Vn=vn, v0=vg, p0=pg, q0=qg, a0=a0,
                       pmax=pmax, pmin=pmin,
                       qmax=qmax, qmin=qmin,
                       Pc1=pc1, Pc2=pc2,
                       Qc1min=qc1min, Qc1max=qc1max,
                       Qc2min=qc2min, Qc2max=qc2max,
                       Ragc=ramp_agc, R10=ramp_10,
                       R30=ramp_30, Rq=ramp_q,
                       apf=apf, gentype=gt, genfuel=gf)
        else:
            system.add('PV', idx=gen_idx, bus=bus_idx, busr=bus_idx,
                       name=None,
                       u=status, Sn=data[6],
                       Vn=vn, v0=vg, p0=pg, q0=qg,
                       pmax=pmax, pmin=pmin,
                       qmax=qmax, qmin=qmin,
                       Pc1=pc1, Pc2=pc2,
                       Qc1min=qc1min, Qc1max=qc1max,
                       Qc2min=qc2min, Qc2max=qc2max,
                       Ragc=ramp_agc, R10=ramp_10,
                       R30=ramp_30, Rq=ramp_q,
                       apf=apf, gentype=gt, genfuel=gf)

    for data in mpc['branch']:
        # fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle
        # 0     1       2   3   4   5       6       7       8       9
        # status	angmin	angmax	Pf	Qf	Pt	Qt
        # 10        11      12      13  14  15  16
        fbus = int(data[0])
        tbus = int(data[1])
        r = data[2]
        x = data[3]
        b = data[4]
        rate_a = data[5] / base_mva
        rate_b = data[6] / base_mva
        rate_c = data[7] / base_mva
        amin = data[11] * deg2rad
        amax = data[12] * deg2rad

        status = int(data[10])

        if (data[8] == 0.0) or (data[8] == 1.0 and data[9] == 0.0):
            # not a transformer
            tf = False
            ratio = 1
            angle = 0
        else:
            tf = True
            ratio = data[8]
            angle = data[9] * deg2rad

        vf = system.Bus.Vn.v[system.Bus.idx2uid(fbus)]
        vt = system.Bus.Vn.v[system.Bus.idx2uid(tbus)]
        system.add('Line', u=status, name=f'Line {fbus:.0f}-{tbus:.0f}',
                   Vn1=vf, Vn2=vt,
                   bus1=fbus, bus2=tbus,
                   r=r, x=x, b=b,
                   trans=tf, tap=ratio, phi=angle,
                   rate_a=rate_a, rate_b=rate_b, rate_c=rate_c,
                   amin=amin, amax=amax)

    if ('bus_name' in mpc) and (len(mpc['bus_name']) == len(system.Bus.name.v)):
        system.Bus.name.v[:] = mpc['bus_name']

    # --- gencost ---
    if 'gencost' in mpc:
        gcost_idx = 0
        gen_idx = np.arange(mpc['gen'].shape[0]) + 1
        mpc_cost = mpc['gencost']
        if mpc_cost[0, 0] == 1:
            logger.warning("Type 1 gencost detected, which is not supported in AMS.\n"
                           "Default type 2 cost parameters will be used as a fallback.\n"
                           "It is recommended to manually convert the gencost data to type 2.")
            mpc_cost = np.repeat(np.array([[2, 0, 0, 3, 0, 0, 0]]),
                                 mpc_cost.shape[0], axis=0)
        for data, gen in zip(mpc_cost, gen_idx):
            # NOTE: only type 2 costs are supported for now
            # type  startup shutdown	n	c2  c1  c0
            # 0     1       2           3   4   5   6
            if data[0] != 2:
                raise ValueError('Only MODEL 2 costs are supported')
            gcost_idx += 1
            gctype = int(data[0])
            startup = data[1]
            shutdown = data[2]
            if data[3] == 3:
                c2 = data[4] * base_mva ** 2
                c1 = data[5] * base_mva
                c0 = data[6]
            elif data[3] == 2:
                c2 = 0
                c1 = data[4] * base_mva
                c0 = data[5]
            else:
                raise ValueError('Unrecognized gencost model, please use eighter quadratic or linear cost model')
            system.add('GCost', gen=int(gen),
                       u=1, type=gctype,
                       idx=gcost_idx,
                       name=None,
                       csu=startup, csd=shutdown,
                       c2=c2, c1=c1, c0=c0
                       )

    # --- Area ---
    area = system.Bus.area.v
    area_map = {}
    if area:
        for a in set(area):
            a_new = system.add('Area',
                               param_dict=dict(idx=a, name=a))
            area_map[a] = a_new
        system.Bus.area.v = [area_map[a] for a in area]

    # --- Zone ---
    zone = system.Bus.zone.v
    zone_map = {}
    if zone:
        n_zone = system.Area.n
        for z in set(zone):
            z_new = system.add('Zone',
                               param_dict=dict(idx=int(n_zone + 1),
                                               name=f'{n_zone + 1}'))
            zone_map[z] = z_new
            n_zone += 1
        system.Bus.zone.v = [zone_map[z] for z in zone]

    return True


def _get_bus_id_caller(bus):
    """
    Helper function to get the bus id. Force bus id to be uid+1.

    Parameters
    ----------
    bus : andes.models.bus.Bus
        Bus object

    Returns
    -------
    lambda function to that takes bus idx and returns bus id for matpower case
    """

    if np.array(bus.idx.v).dtype in ['int', 'float']:
        return lambda x: x
    else:
        return lambda x: list(np.array(bus.idx2uid(x)) + 1)


def system2mpc(system) -> dict:
    """
    Convert a **setup** AMS system to a MATPOWER mpc dictionary.

    This function is revised from ``andes.io.matpower.system2mpc``.

    In the ``gen`` section, slack generators are listed before PV generators.

    In the converted MPC, the indices of area (bus[:, 6]) and zone (bus[:, 10])
    may differ from the original MPC. However, the mapping relationship is preserved.
    For example, if the original MPC numbers areas starting from 1, the converted
    MPC may number them starting from 0.

    The coefficients ``c2`` and ``c1`` in the generator cost data are scaled by
    ``base_mva`` to match MATPOWER's unit convention (MW).

    Parameters
    ----------
    system : ams.core.system.System
        The AMS system to be converted.

    Returns
    -------
    mpc : dict
        A dictionary in MATPOWER format representing the converted AMS system.
    """

    mpc = dict(version='2',
               baseMVA=system.config.mva,
               bus=np.zeros((system.Bus.n, 13), dtype=np.float64),
               gen=np.zeros((system.PV.n + system.Slack.n, 21), dtype=np.float64),
               branch=np.zeros((system.Line.n, 17), dtype=np.float64),
               gencost=np.zeros((system.GCost.n, 7), dtype=np.float64),
               )

    if not system.is_setup:
        logger.warning("System is not setup and will be setup now.")
        system.setup()

    if system.Bus.name.v is not None:
        mpc['bus_name'] = system.Bus.name.v

    base_mva = system.config.mva

    # --- Bus ---
    bus = mpc['bus']
    gen = mpc['gen']

    to_busid = _get_bus_id_caller(system.Bus)

    bus[:, 0] = to_busid(system.Bus.idx.v)
    bus[:, 1] = 1
    if system.Area.n > 0:
        bus[:, 6] = system.Area.idx2uid(system.Bus.area.v)
    bus[:, 7] = system.Bus.v0.v
    bus[:, 8] = system.Bus.a0.v * rad2deg
    bus[:, 9] = system.Bus.Vn.v
    if system.Zone.n > 0:
        bus[:, 10] = system.Zone.idx2uid(system.Bus.zone.v)
    bus[:, 11] = system.Bus.vmax.v
    bus[:, 12] = system.Bus.vmin.v

    # --- PQ ---
    if system.PQ.n > 0:
        pq_pos = system.Bus.idx2uid(system.PQ.bus.v)
        u = system.PQ.u.v
        bus[pq_pos, 2] = u * system.PQ.p0.v * base_mva
        bus[pq_pos, 3] = u * system.PQ.q0.v * base_mva

    # --- Shunt ---
    if system.Shunt.n > 0:
        shunt_pos = system.Bus.idx2uid(system.Shunt.bus.v)
        bus[shunt_pos, 4] = system.Shunt.g.v * base_mva
        bus[shunt_pos, 5] = system.Shunt.b.v * base_mva

    # --- PV ---
    if system.PV.n > 0:
        PV = system.PV
        pv_pos = system.Bus.idx2uid(PV.bus.v)
        bus[pv_pos, 1] = 2
        gen[system.Slack.n:, 0] = to_busid(PV.bus.v)
        gen[system.Slack.n:, 1] = PV.p0.v * base_mva
        gen[system.Slack.n:, 2] = PV.q0.v * base_mva
        gen[system.Slack.n:, 3] = PV.qmax.v * base_mva
        gen[system.Slack.n:, 4] = PV.qmin.v * base_mva
        gen[system.Slack.n:, 5] = PV.v0.v
        gen[system.Slack.n:, 6] = PV.Sn.v
        gen[system.Slack.n:, 7] = PV.u.v
        gen[system.Slack.n:, 8] = (PV.ctrl.v * PV.pmax.v + (1 - PV.ctrl.v) * PV.p0.v) * base_mva
        gen[system.Slack.n:, 9] = (PV.ctrl.v * PV.pmin.v + (1 - PV.ctrl.v) * PV.p0.v) * base_mva
        gen[system.Slack.n:, 16] = PV.Ragc.v * base_mva * 60    # MW/h -> MW/min
        gen[system.Slack.n:, 17] = PV.R10.v * base_mva
        gen[system.Slack.n:, 18] = PV.R30.v * base_mva
        gen[system.Slack.n:, 19] = PV.Rq.v * base_mva * 60  # MVAr/h -> MVAr/min

    # --- Slack ---
    if system.Slack.n > 0:
        slack_pos = system.Bus.idx2uid(system.Slack.bus.v)
        bus[slack_pos, 1] = 3
        bus[slack_pos, 8] = system.Slack.a0.v * rad2deg
        gen[:system.Slack.n, 0] = to_busid(system.Slack.bus.v)
        gen[:system.Slack.n, 1] = system.Slack.p0.v * base_mva
        gen[:system.Slack.n, 2] = system.Slack.q0.v * base_mva
        gen[:system.Slack.n, 3] = system.Slack.qmax.v * base_mva
        gen[:system.Slack.n, 4] = system.Slack.qmin.v * base_mva
        gen[:system.Slack.n, 5] = system.Slack.v0.v
        gen[:system.Slack.n, 6] = system.Slack.Sn.v
        gen[:system.Slack.n, 7] = system.Slack.u.v
        gen[:system.Slack.n, 8] = system.Slack.pmax.v * base_mva
        gen[:system.Slack.n, 9] = system.Slack.pmin.v * base_mva
        gen[:system.Slack.n, 16] = system.Slack.Ragc.v * base_mva * 60    # MW/h -> MW/min
        gen[:system.Slack.n, 17] = system.Slack.R10.v * base_mva
        gen[:system.Slack.n, 18] = system.Slack.R30.v * base_mva
        gen[:system.Slack.n, 19] = system.Slack.Rq.v * base_mva * 60    # MVAr/h -> MVAr/min

    if system.Line.n > 0:
        branch = mpc['branch']
        branch[:, 0] = to_busid(system.Line.bus1.v)
        branch[:, 1] = to_busid(system.Line.bus2.v)
        branch[:, 2] = system.Line.r.v
        branch[:, 3] = system.Line.x.v
        branch[:, 4] = system.Line.b.v
        branch[:, 5] = system.Line.rate_a.v * base_mva
        branch[:, 6] = system.Line.rate_b.v * base_mva
        branch[:, 7] = system.Line.rate_c.v * base_mva
        branch[:, 8] = system.Line.tap.v
        branch[:, 9] = system.Line.phi.v * rad2deg
        branch[:, 10] = system.Line.u.v
        branch[:, 11] = system.Line.amin.v * rad2deg
        branch[:, 12] = system.Line.amax.v * rad2deg

    # --- GCost ---
    # NOTE: adjust GCost sequence to match the generator sequence
    if system.GCost.n > 0:
        stg_idx = system.Slack.idx.v + system.PV.idx.v
        gcost_idx = system.GCost.find_idx(keys=['gen'], values=[stg_idx])
        gcost_uid = system.GCost.idx2uid(gcost_idx)
        gencost = mpc['gencost']
        gencost[:, 0] = system.GCost.type.v[gcost_uid]
        gencost[:, 1] = system.GCost.csu.v[gcost_uid]
        gencost[:, 2] = system.GCost.csd.v[gcost_uid]
        gencost[:, 3] = 3
        gencost[:, 4] = system.GCost.c2.v[gcost_uid] / base_mva / base_mva
        gencost[:, 5] = system.GCost.c1.v[gcost_uid] / base_mva
        gencost[:, 6] = system.GCost.c0.v[gcost_uid]
    else:
        mpc.pop('gencost')

    # --- gentype ---
    stg = system.StaticGen.get_all_idxes()
    gentype = system.StaticGen.get(src='gentype', attr='v', idx=stg)
    if any(gentype):
        mpc['gentype'] = np.array(gentype)

    # --- genfuel ---
    genfuel = system.StaticGen.get(src='genfuel', attr='v', idx=stg)
    if any(genfuel):
        mpc['genfuel'] = np.array(genfuel)

    # --- Bus Name ---
    if any(system.Bus.name.v):
        mpc['bus_name'] = np.array(system.Bus.name.v)

    return mpc


def mpc2m(mpc: dict, outfile: str) -> str:
    """
    Write a MATPOWER mpc dict to a M-file.

    Parameters
    ----------
    mpc : dict
        MATPOWER mpc dictionary.
    outfile : str
        Path to the output M-file.
    """
    with open(outfile, 'w') as f:
        # Add version info
        f.write(f"%% Converted by AMS {version}\n")
        f.write(f"%% {copyright_msg}\n\n")
        f.write(f"%% {nowarranty_msg}\n")
        f.write(f"%% Convert time: {report_time}\n\n")

        f.write("function mpc = mpc_case\n")
        f.write("mpc.version = '2';\n\n")

        # Write baseMVA
        f.write(f"%% system MVA base\nmpc.baseMVA = {mpc['baseMVA']};\n\n")

        # Write bus data
        f.write("%% bus data\n")
        f.write("%% bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin\n")
        f.write("mpc.bus = [\n")
        for row in mpc['bus']:
            f.write("    " + "\t".join(f"{val:.6g}" for val in row) + ";\n")
        f.write("];\n\n")

        # Write generator data
        f.write("%% generator data\n")
        f.write("%% bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin\n")
        f.write("%% Pc1 Pc2 Qc1min Qc1max Qc2min Qc2max ramp_agc ramp_10 ramp_30 ramp_q apf\n")
        f.write("mpc.gen = [\n")
        for row in mpc['gen']:
            f.write("    " + "\t".join(f"{val:.6g}" for val in row) + ";\n")
        f.write("];\n\n")

        # Write branch data
        f.write("%% branch data\n")
        f.write("%% fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax PF QF PT QT\n")
        f.write("mpc.branch = [\n")
        for row in mpc['branch']:
            f.write("    " + "\t".join(f"{val:.6g}" for val in row) + ";\n")
        f.write("];\n\n")

        # Write generator cost data if available
        if 'gencost' in mpc:
            f.write("%% generator cost data\n")
            f.write("%% 1 startup shutdown n x1 y1 ... xn yn\n")
            f.write("%% 2 startup shutdown n c(n-1) ... c0\n")
            f.write("mpc.gencost = [\n")
            for row in mpc['gencost']:
                f.write("    " + "\t".join(f"{val:.6g}" for val in row) + ";\n")
            f.write("];\n\n")

        # Write bus names if available and not all None
        if 'bus_name' in mpc and any(mpc['bus_name']):
            f.write("%% bus names\n")
            f.write("mpc.bus_name = {\n")
            for name in mpc['bus_name']:
                f.write(f"    '{name}';\n")
            f.write("};\n\n")

        # Write generator types if available and not all None
        if 'gentype' in mpc and any(mpc['gentype']):
            f.write("%% generator types\n")
            f.write("mpc.gentype = {\n")
            for gentype in mpc['gentype']:
                f.write(f"    '{gentype}';\n")
            f.write("};\n\n")

        # Write generator fuels if available and not all None
        if 'genfuel' in mpc and any(mpc['genfuel']):
            f.write("%% generator fuels\n")
            f.write("mpc.genfuel = {\n")
            for genfuel in mpc['genfuel']:
                f.write(f"    '{genfuel}';\n")
            f.write("};\n\n")

    logger.info(f"Finished writing MATPOWER case to {outfile}")
    return outfile


def write(system, outfile: str, overwrite: bool = None) -> bool:
    """
    Export an AMS system to a MATPOWER M-file.

    This function converts an AMS system object into a MATPOWER-compatible
    mpc dictionary and writes it to a specified output file in MATPOWER format.

    In the converted MPC, the indices of area (bus[:, 6]) and zone (bus[:, 10])
    may differ from the original MPC. However, the mapping relationship is preserved.
    For example, if the original MPC numbers areas starting from 1, the converted
    MPC may number them starting from 0.

    Parameters
    ----------
    system : ams.system.System
        A loaded system.
    outfile : str
        Path to the output file.
    overwrite : bool, optional
        None to prompt for overwrite selection; True to overwrite; False to not overwrite.

    Returns
    -------
    bool
        True if the file was successfully written, False otherwise.
    """
    if not confirm_overwrite(outfile, overwrite=overwrite):
        return False

    mpc = system2mpc(system)
    mpc2m(mpc, outfile)
    logger.info('MATPOWER m case file written to "%s"', outfile)
    return True
