"""
PSS/E .raw reader for AMS.
This module is the existing module in ``andes.io.psse``.
"""

from andes.io.psse import testlines  # NOQA
from andes.io.psse import read as ad_read
from andes.io.xlsx import confirm_overwrite
from andes.shared import rad2deg, pd

from ams import __version__ as version
from ams.shared import copyright_msg, nowarranty_msg, report_time


def read(system, file):
    """
    Read PSS/E RAW file v32/v33 formats.

    Revised from ``andes.io.psse.read`` to complete model ``Zone`` when necessary.
    """
    ret = ad_read(system, file)
    # Extract zone data
    zone = system.Bus.zone.v
    zone_map = {}

    # Check if there are zones to process
    # NOTE: since `Zone` and `Area` below to group `Collection`, we add
    # numerical Zone idx after the last Area idx.
    if zone:
        n_zone = system.Area.n
        for z in set(zone):
            # Add new zone and update the mapping
            z_new = system.add(
                'Zone',
                param_dict=dict(idx=n_zone + 1, name=f'{n_zone + 1}')
            )
            zone_map[z] = z_new
            n_zone += 1

        # Update the zone values in the system
        system.Bus.zone.v = [zone_map[z] for z in zone]
    return ret


def write_raw(system, outfile: str, overwrite: bool = None):
    """
    Convert AMS system to PSS/E RAW file.

    Parameters
    ----------
    system : System
        The AMS system to be converted.
    outfile : str
        The output file path.
    overwrite : bool, optional
        If True, overwrite the file if it exists. If False, do not overwrite.
    """
    if not confirm_overwrite(outfile, overwrite=overwrite):
        return False

    mva = system.config.mva
    freq = system.config.freq
    name = system.files.name

    with open(outfile, 'w') as f:
        # PSS/E version and date
        f.write(f"0, {mva:.2f}, 33, 0, 1, {freq:.2f}     ")

        f.write(f"/ PSS/E 33 RAW, {report_time}\n")
        f.write(f"Created by AMS {version}\n")
        f.write(f"{copyright_msg}\n")
        f.write(f"{nowarranty_msg}\n")
        f.write(f"{name.upper()} \n")

        # --- Bus ---
        bus = system.Bus.cache.df_in
        # 0,  1,    2,      3,    4,    5,    6,     7,  8,  9,    10
        # ID, Name, BaseKV, Type, Area, Zone, Owner, Vm, Va, Vmax, Vmin
        # Define column widths for alignment
        column_widths = [6, 8, 8, 2, 3, 3, 3, 8, 8, 8, 8]
        bus_owner = {}
        for i, o in enumerate(set(bus['owner'])):
            bus_owner[o] = i + 1
        for row in bus.itertuples():
            # Prepare the data for each column
            data = [
                int(row.idx), row.name, row.Vn, int(row.type),
                int(system.Collection.idx2uid(row.area) + 1),
                int(system.Collection.idx2uid(row.zone) + 1),
                int(bus_owner[row.owner]),
                float(row.v0), float(row.a0 * rad2deg),
                float(row.vmax), float(row.vmin)]
            # Format each column with ',' as the delimiter
            formatted_row = ",".join(
                f"{value:>{width}}" if isinstance(value, (int, float)) else f"'{value:>{width - 2}}'"
                for value, width in zip(data, column_widths)
            ) + "\n"
            # Write the formatted row to the file
            f.write(formatted_row)

        # --- Load ---
        f.write("0 / END OF BUS DATA, BEGIN LOAD DATA\n")
        load = system.PQ.cache.df_in
        # 0,   1,  2,      3,    4,    5,      6,       7,  8,  9,  10, 11
        # Bus, Id, Status, Area, Zone, PL(MW), QL (MW), IP, IQ, YP, YQ, Owner
        # NOTE: load are converted to constant load, IP, IQ, YP, YQ are ignored by setting to 0
        column_widths = [6, 8, 2, 3, 3, 10, 10, 2, 2, 10, 10, 3]
        for row in load.itertuples():
            # Prepare the data for each column
            data = [
                int(row.bus),  # Bus number
                int(system.PQ.idx2uid(row.idx) + 1),  # Load ID (unique index + 1)
                int(row.u),  # Status
                int(system.Collection.idx2uid(system.PQ.get('area', row.idx, 'v')) + 1),  # Area number
                int(system.Collection.idx2uid(system.PQ.get('zone', row.idx, 'v')) + 1),  # Zone number
                float(row.p0 * mva),  # PL (MW)
                float(row.q0 * mva),  # QL (MVar)
                0.0,  # IP (ignored, set to 0)
                0.0,  # IQ (ignored, set to 0)
                0.0,  # YP (ignored, set to 0)
                0.0,  # YQ (ignored, set to 0)
                int(bus_owner[row.owner])  # Owner
            ]
            # Format each column with ',' as the delimiter
            formatted_row = ",".join(
                f"{value:>{width}}" if isinstance(value, (int, float)) else f"'{value:>{width - 2}}'"
                for value, width in zip(data, column_widths)
            ) + "\n"
            # Write the formatted row to the file
            f.write(formatted_row)

        # --- Fixed Shunt ---
        f.write("0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA\n")
        shunt = system.Shunt.cache.df_in
        # 0,   1,  2,      3,      4
        # Bus, Id, Status, G (MW), B (Mvar)
        # NOTE: ANDES parse v33 swshunt into fixed shunt
        column_widths = [6, 8, 2, 10, 10]
        for row in shunt.itertuples():
            # Prepare the data for each column
            data = [
                int(row.bus),                                   # Bus number
                int(system.Shunt.idx2uid(row.idx) + 1),         # Shunt ID (unique index + 1)
                int(row.u),                                     # Status
                float(row.g * mva),                             # Conductance (MW at system base)
                float(row.b * mva)                              # Susceptance (Mvar at system base)
            ]

            # Format each column with ',' as the delimiter
            formatted_row = ",".join(
                f"{value:>{width}}" if isinstance(value, (int, float)) else f"'{value:>{width - 2}}'"
                for value, width in zip(data, column_widths)
            ) + "\n"
            # Write the formatted row to the file
            f.write(formatted_row)

        # --- Generator ---
        f.write("0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA\n")
        pv = system.PV.cache.df_in
        slack = system.Slack.cache.df_in

        gen = pd.concat([pv, slack.drop(columns=['a0'])], axis=0)
        gen["subidx"] = gen.groupby('bus').cumcount() + 1

        column_widths = [6, 8, 2, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8]
        # 0,  1,  2,  3,  4,  5,  6,    7,     8,  9, 10, 11, 12,   13,   14,
        # I, ID, PG, QG, QT, QB, VS, IREG, MBASE, ZR, ZX, RT, XT, GTAP, STAT,
        #    15, 16, 17, 18, 19, ...,           26,  27
        # RMPCT, PT, PB, O1, F1, ..., O4, F4, WMOD, WPF
        # The columns above for v33 is different from the manual of v34.5, which includes two new columns:
        # `NREG`` at 8 and `BSLOD` before `O1`
        for row in gen.itertuples():
            # Prepare the data for each column
            data = [
                int(row.bus),                  # I: Bus number
                int(row.subidx),               # ID: Generator ID (subindex)
                float(row.p0 * mva),           # PG: Generated MW
                float(row.q0 * mva),           # QG: Generated MVar
                float(row.qmax * mva),         # QT: Max Q (MVar)
                float(row.qmin * mva),         # QB: Min Q (MVar)
                float(row.v0),                 # VS: Setpoint voltage (p.u.)
                int(0),                        # IREG: Regulated bus (not tracked)
                float(row.Sn if hasattr(row, 'Sn') else mva),  # MBASE: Machine base MVA
                float(row.ra),                 # ZR: Armature resistance (p.u.)
                float(row.xs),                 # ZX: Synchronous reactance (p.u.)
                float(0),                      # RT: Step-up transformer resistance (not tracked)
                float(0),                      # XT: Step-up transformer reactance (not tracked)
                int(0),                        # GTAP: Step-up transformer off-nominal turns ratio (not tracked)
                int(row.u)                     # STAT: Status
            ]
            # Format each column with ',' as the delimiter
            formatted_row = ",".join(
                f"{value:>{width}}" if isinstance(value, (int, float)) else f"'{value:>{width - 2}}'"
                for value, width in zip(data, column_widths)
            ) + "\n"
            # Write the formatted row to the file
            f.write(formatted_row)

        # --- Line ---
        f.write("0 / END OF GENERATOR DATA, BEGIN BRANCH DATA\n")
        line = system.Line.cache.df_in
        branch = line[line['trans'] == 0].reset_index(drop=True)
        transf = line[line['trans'] == 1].reset_index(drop=True)
        # 1) branch
        # 0,  1,   2,   3,   4,   5,     6,     7,     8,   9,  10, 11, 12, 13,  14, 15, 16
        # I,  J,  CKT,  R,   X,   B,  RATEA, RATEB, RATEC, GI,  BI, GJ, BJ, ST, LEN, O1, F1, ..., O4, F4
        column_widths = [6, 6, 4, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8, 3, 8]
        for row in branch.itertuples():
            data = [
                int(row.bus1),                      # I: From bus number
                int(row.bus2),                      # J: To bus number
                "'1'",                              # CKT: Circuit ID (default '1')
                float(row.r),                       # R: Resistance (p.u.)
                float(row.x),                       # X: Reactance (p.u.)
                float(row.b),                       # B: Total line charging (p.u.)
                float(row.rate_a),                  # RATEA: Rating A (MVA)
                float(row.rate_b),                  # RATEB: Rating B (MVA)
                float(row.rate_c),                  # RATEC: Rating C (MVA)
                0.0,                                # GI: From bus conductance (not tracked)
                0.0,                                # BI: From bus susceptance (not tracked)
                0.0,                                # GJ: To bus conductance (not tracked)
                0.0,                                # BJ: To bus susceptance (not tracked)
                int(row.u),                         # ST: Status
                0.0                                 # LEN: Line length (not tracked)
                # O1, F1, ..., O4, F4 omitted for v33
            ]
            formatted_row = ",".join(
                f"{value:>{width}}" if isinstance(value, (int, float)) else f"{value:>{width}}"
                for value, width in zip(data, column_widths)
            ) + "\n"
            f.write(formatted_row)
        # 2) transformer
        f.write("0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA\n")
        for row in transf.itertuples():
            # Map AMS Line (trans) fields to PSSE RAW transformer columns
            # Only 2-winding transformers are supported here
            # 0,  1,   2,   3,   4,   5,     6,     7,     8,   9,  10, 11, 12, 13,  14, 15, 16
            # I,  J,  CKT,  R,   X,   B,  RATEA, RATEB, RATEC, GI,  BI, GJ, BJ, ST, LEN
            data = [
                int(row.bus1),                      # I: From bus number
                int(row.bus2),                      # J: To bus number
                "'1'",                              # CKT: Circuit ID (default '1')
                float(row.r),                       # R: Resistance (p.u.)
                float(row.x),                       # X: Reactance (p.u.)
                float(row.b),                       # B: Total line charging (p.u.)
                float(row.rate_a),                  # RATEA: Rating A (MVA)
                float(row.rate_b),                  # RATEB: Rating B (MVA)
                float(row.rate_c),                  # RATEC: Rating C (MVA)
                0.0,                                # GI: From bus conductance (not tracked)
                0.0,                                # BI: From bus susceptance (not tracked)
                0.0,                                # GJ: To bus conductance (not tracked)
                0.0,                                # BJ: To bus susceptance (not tracked)
                int(row.u),                         # ST: Status
                0.0                                 # LEN: Line length (not tracked)
                # O1, F1, ..., O4, F4 omitted for v33
            ]
            formatted_row = ",".join(
                f"{value:>{width}}" if isinstance(value, (int, float)) else f"{value:>{width}}"
                for value, width in zip(data, column_widths)
            ) + "\n"
            f.write(formatted_row)

            # --- Area ---
            f.write("0 / END OF TRANSFORMER DATA, BEGIN AREA DATA\n")
            area = system.Area.cache.df_in
            for row in area.itertuples():
                # PSSE expects: ID, ISW, PDES, PTOL, NAME
                # Here, ISW, PDES, PTOL are set to 0 by default
                f.write(f"{int(system.Area.idx2uid(row.idx) + 1):6d}, 0, 0.0, 0.0, '{row.name}'\n")

            # --- Zone ---
            f.write("0 / END OF AREA DATA, BEGIN ZONE DATA\n")
            zone = system.Zone.cache.df_in
            # 0,    1
            # ID, Name
            for row in zone.itertuples():
                f.write(f"{int(system.Zone.idx2uid(row.idx) + 1):6d}, '{row.name}'\n")
            f.write("0 / END OF ZONE DATA\n")

        # End of file
        f.write("Q\n")

    return True
