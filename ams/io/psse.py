"""
PSS/E .raw reader for AMS.
This module is the existing module in ``andes.io.psse``.
"""

import re
from collections import defaultdict

import numpy as np

from andes.io.psse import (read, testlines)  # NOQA
from andes.io.xlsx import confirm_overwrite

from andes.shared import deg2rad, pd

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
        f.write(f"0, {mva:.2f}, 33, 0, 1, {freq:.2f}")

        # Copyright and warranty information
        f.write(f"/ PSS/E 33 RAW Created by AMS {version}, {report_time} {name} \n")

        # Bus Data
        bus = system.Bus.cache.df_in
        # f.write("0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA\n")
        for row in bus.itertuples():
            out = f"{row.idx}"
            f.write(out)
            break
        f.write("\n0 / END OF BUS DATA, BEGIN LOAD DATA\n")

        # # Load Data
        # f.write("0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA\n")
        # for load in system.loads:
        #     pl_mw = load.p0 * mva
        #     ql_mvar = load.q0 * mva
        #     f.write(f"{int(load.bus):<6},'{load.subidx:<1}',{int(load.u):<2},{int(load.area):<3},{int(load.zone):<3},{pl_mw:<10.3f},{ql_mvar:<10.3f},{load.ip:<9.3f},{load.iq:<9.3f},{load.yp:<9.3f},{load.yq:<9.3f},{int(load.owner):<3},1,0\n")

        # # Fixed Shunt Data
        # f.write("0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA\n")
        # for shunt in system.shunts:
        #     g_mw = shunt.g * mva
        #     b_mvar = shunt.b * mva
        #     f.write(f"{int(shunt.bus):<6},'{shunt.name:<8}',{int(shunt.u):<2},{g_mw:<10.3f},{b_mvar:<10.3f}\n")

        # # Generator Data
        # f.write("0 / END OF GENERATOR DATA, BEGIN BRANCH DATA\n")
        # for gen in system.generators:
        #     pg_mw = gen.p0 * mva
        #     qg_mvar = gen.q0 * mva
        #     qt_mvar = gen.qmax * mva
        #     qb_mvar = gen.qmin * mva
        #     f.write(f"{int(gen.bus):<6},'{gen.subidx:<1}',{pg_mw:<10.3f},{qg_mvar:<10.3f},{qt_mvar:<10.3f},{qb_mvar:<10.3f},{gen.v0:<8.5f},{int(gen.ireg):<3},{gen.Sn:<9.3f},{gen.ra:<9.5f},{gen.xs:<9.5f},{gen.rt:<9.5f},{gen.xt:<9.5f},{gen.gtap:<8.5f},{int(gen.u):<2},{gen.rmpct:<8.3f},{gen.pt:<10.3f},{gen.pb:<10.3f}")
        #     # Add remaining generator parameters (O1-F4, WMOD, WPF) - assuming they exist as attributes
        #     for i in range(1, 5):
        #         o_val = getattr(gen, f'o{i}', 0.0)
        #         f_val = getattr(gen, f'f{i}', 0.0)
        #         f.write(f",{o_val:<9.3f},{f_val:<9.3f}")
        #     wmod = getattr(gen, 'wmod', 0)
        #     wpf = getattr(gen, 'wpf', 0.0)
        #     f.write(f",{int(wmod):<2},{wpf:<8.5f}\n")

        # # Branch Data (Lines and Transformers)
        # f.write("0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA\n")
        # for line in system.lines:
        #     f.write(f"{int(line.bus1):<6},{int(line.bus2):<6},'{line.ckt:<2}',{line.r:<9.5f},{line.x:<9.5f},{line.b:<9.5f},{line.rate_a:<8.2f},{line.rate_b:<8.2f},{line.rate_c:<8.2f},{line.gi:<9.5f},{line.bi:<9.5f},{line.gj:<9.5f},{line.bj:<9.5f},{int(line.u):<2},{line.length:<8.2f}")
        #     # Add remaining line parameters (O1-F4) - assuming they exist as attributes
        #     for i in range(1, 5):
        #         o_val = getattr(line, f'o{i}', 0.0)
        #         f_val = getattr(line, f'f{i}', 0.0)
        #         f.write(f",{o_val:<9.3f},{f_val:<9.3f}")
        #     f.write("\n")

        # for transf in system.transformers:
        #     f.write(f"{int(transf.bus1):<6},{int(transf.bus2):<6},{int(transf.bus3):<6},'{transf.ckt:<2}',{int(transf.cw):<2},{int(transf.cz):<2},{int(transf.cm):<2},{transf.mag1:<9.3f},{transf.mag2:<9.3f},{int(transf.nmetr):<2},'{transf.name:<16}',{int(transf.u):<2}")
        #     # Add remaining transformer parameters (O1-F4) - assuming they exist as attributes
        #     for i in range(1, 5):
        #         o_val = getattr(transf, f'o{i}', 0.0)
        #         f_val = getattr(transf, f'f{i}', 0.0)
        #         f.write(f",{o_val:<9.3f},{f_val:<9.3f}")
        #     f.write("\n")
        #     f.write(f"{transf.r1_2:<12.5f},{transf.x1_2:<12.5f},{transf.sbase1_2:<10.3f},{transf.r2_3:<12.5f},{transf.x2_3:<12.5f},{transf.sbase2_3:<10.3f},{transf.r3_1:<12.5f},{transf.x3_1:<12.5f},{transf.sbase3_1:<10.3f},{transf.vmstar:<9.3f},{transf.anstar:<9.3f}\n")
        #     f.write(f"{transf.windv1:<12.5f},{transf.nomv1:<9.3f},{transf.ang1:<9.3f},{transf.rata1:<8.2f},{transf.ratb1:<8.2f},{transf.ratc1:<8.2f},{int(transf.cod1):<2},{int(transf.cont1):<2},{transf.rma1:<9.3f},{transf.rmi1:<9.3f},{transf.vma1:<9.3f},{transf.vmi1:<9.3f},{int(transf.ntp1):<3},{transf.tab1:<9.3f},{transf.cr1:<9.3f},{transf.cx1:<9.3f}\n")
        #     f.write(f"{transf.windv2:<12.5f},{transf.nomv2:<9.3f},{transf.ang2:<9.3f},{transf.rata2:<8.2f},{transf.ratb2:<8.2f},{transf.ratc2:<8.2f},{int(transf.cod2):<2},{int(transf.cont2):<2},{transf.rma2:<9.3f},{transf.rmi2:<9.3f},{transf.vma2:<9.3f},{transf.vmi2:<9.3f},{int(transf.ntp2):<3},{transf.tab2:<9.3f},{transf.cr2:<9.3f},{transf.cx2:<9.3f}\n")
        #     if transf.nwind == 3:
        #         f.write(f"{transf.windv3:<12.5f},{transf.nomv3:<9.3f},{transf.ang3:<9.3f},{transf.rata3:<8.2f},{transf.ratb3:<8.2f},{transf.ratc3:<8.2f},{int(transf.cod3):<2},{int(transf.cont3):<2},{transf.rma3:<9.3f},{transf.rmi3:<9.3f},{transf.vma3:<9.3f},{transf.vmi3:<9.3f},{int(transf.ntp3):<3},{transf.tab3:<9.3f},{transf.cr3:<9.3f},{transf.cx3:<9.3f}\n")

        # # Area Data
        # f.write("0 / END OF TRANSFORMER DATA, BEGIN AREA DATA\n")
        # for area in system.areas:
        #     f.write(f"{int(area.idx):<3},{int(area.slack):<6},{area.pdes:<12.3f},{area.ptol:<10.3f},'{area.name:<16}'\n")

        # # Two-Terminal DC Data (assuming no DC lines for simplicity)
        # f.write("0 / END OF AREA DATA, BEGIN TWO-TERMINAL DC DATA\n")
        # f.write("0 / END OF TWO-TERMINAL DC DATA, BEGIN VSC DC LINE DATA\n")
        # f.write("0 / END OF VSC DC LINE DATA, BEGIN IMPEDANCE CORRECTION DATA\n")
        # f.write("0 / END OF IMPEDANCE CORRECTION DATA, BEGIN MULTI-TERMINAL DC DATA\n")
        # f.write("0 / END OF MULTI-TERMINAL DC DATA, BEGIN MULTI-SECTION LINE DATA\n")
        # f.write("0 / END OF MULTI-SECTION LINE DATA, BEGIN ZONE DATA\n")

        # # Zone Data
        # for zone in system.zones:
        #     f.write(f"{int(zone.idx):<3},'{zone.name:<16}'\n")

        # # Inter-Area Transfer Data (assuming none)
        # f.write("0 / END OF ZONE DATA, BEGIN INTER-AREA TRANSFER DATA\n")
        # f.write("0 / END OF INTER-AREA TRANSFER DATA, BEGIN OWNER DATA\n")

        # # Owner Data
        # for owner in system.owners:
        #     f.write(f"{int(owner.idx):<3},'{owner.name:<16}'\n")

        # # FACTS Device Data (assuming none)
        # f.write("0 / END OF OWNER DATA, BEGIN FACTS DEVICE DATA\n")
        # f.write("0 / END OF FACTS DEVICE DATA, BEGIN SWITCHED SHUNT DATA\n")

        # # Switched Shunt Data
        # for swshunt in system.swshunts:
        #     f.write(f"{int(swshunt.bus):<6},{int(swshunt.modsw):<2},{int(swshunt.adjm):<2},{int(swshunt.stat):<2},{swshunt.vswhi:<8.5f},{swshunt.vswlo:<8.5f},{int(swshunt.swrem):<6},{swshunt.rmpct:<8.3f},'{swshunt.rmidnt:<16}',{swshunt.binit:<9.3f}")
        #     for i in range(1, 9):
        #         n_val = getattr(swshunt, f'n{i}', 0)
        #         b_val = getattr(swshunt, f'b{i}', 0.0)
        #         if n_val != 0:
        #             f.write(f",{int(n_val):<3},{b_val/mva:<9.3f}")
        #     f.write("\n")

        # # GNE Data (assuming none)
        # f.write("0 / END OF SWITCHED SHUNT DATA, BEGIN GNE DATA\n")
        # f.write("0 / END OF GNE DATA, BEGIN INDUCTION MACHINE DATA\n")
        # f.write("0 / END OF INDUCTION MACHINE DATA\n")

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
