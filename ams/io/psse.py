"""
PSS/E .raw reader for AMS.
This module is the existing module in ``andes.io.psse``.
"""

import numpy as np

from andes.io.psse import (read, testlines)  # NOQA
from andes.io.xlsx import confirm_overwrite

from andes.shared import rad2deg

from ams import __version__ as version
from ams.shared import copyright_msg, nowarranty_msg, report_time


def write(system, outfile: str, overwrite: bool = None):
    """
    Writes the power flow solution from the andes System object to a PSS/E v33 RAW file.

    Args:
        system (andes.core.System): The andes System object containing the power flow solution.
        filename (str, optional): The name of the output RAW file. Defaults to "output.raw".
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
        for row in bus.itertuples():
            # Prepare the data for each column
            data = [
            int(row.idx), row.name, row.Vn, int(row.type),
            int(system.Collection.idx2uid(row.area) + 1),
            int(system.Collection.idx2uid(row.zone) + 1),
            int(row.owner),
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
                int(row.bus),
                int(system.PQ.idx2uid(row.idx) + 1),
                int(row.u),
                int(system.Collection.idx2uid(system.PQ.get('area', row.idx, 'v')) + 1),
                int(system.Collection.idx2uid(system.PQ.get('zone', row.idx, 'v')) + 1),
                float(row.p0 * mva), float(row.q0 * mva),
                float(0), float(0), float(0), float(0),
                int(row.owner)]
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
                int(row.bus),
                int(system.Shunt.idx2uid(row.idx) + 1),
                int(row.u),
                float(row.g * mva / row.Vn**2),
                float(row.b * mva / row.Vn**2)]

            # Format each column with ',' as the delimiter
            formatted_row = ",".join(
                f"{value:>{width}}" if isinstance(value, (int, float)) else f"'{value:>{width - 2}}'"
                for value, width in zip(data, column_widths)
            ) + "\n"
            # Write the formatted row to the file
            f.write(formatted_row)

        # End of file
        f.write("Q\n")
    
    return True

def to_number(x):
    """
    Converts a string to a float or int if possible, otherwise returns the string.
    """
    try:
        return int(x.strip())
    except ValueError:
        try:
            return float(x.strip())
        except ValueError:
            return x.strip()
