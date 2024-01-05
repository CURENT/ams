"""
PYPOWER reader for AMS.
"""
import logging

from numpy import array  # NOQA
import numpy as np

from ams.io.matpower import mpc2system, system2mpc

logger = logging.getLogger(__name__)


def testlines(infile):
    """
    Test if this file is in the PYPOWER format.

    NOT YET IMPLEMENTED.
    """
    return True


def read(system, file):
    """
    Read a PYPOWER case file into ppc and return an AMS system by calling ``ppc2system``.

    Parameters
    ----------
    system : ams.system
        Empty AMS system to load data into.
    file : str
        The path to the PYPOWER file.

    Returns
    -------
    system : ams.system.System
        The AMS system that loaded the data.
    """
    ppc = py2ppc(file)
    return ppc2system(ppc, system)


def py2ppc(infile: str) -> dict:
    """
    Parse PYPOWER file and return a dictionary with the data.

    Parameters
    ----------
    infile : str
        The path to the PYPOWER file.

    Returns
    -------
    ppc : dict
        The PYPOWER case dict.
    """
    exec(open(f"{infile}").read())
    for name, value in locals().items():
        # Check if the variable name starts with "case"
        if name.startswith("case"):
            ppc = value()
    return ppc


def ppc2system(ppc: dict, system) -> bool:
    """
    Alias for ``mpc2system``. Refer to :py:mod:`ams.io.matpower.mpc2system` for more details.

    Load an PYPOWER case dict into an empth AMS system.

    Parameters
    ----------
    ppc : dict
        The PYPOWER case dict.
    system : ams.system
        Empty AMS system to load data into.

    Returns
    -------
    bool
        True if successful; False otherwise.
    """
    return mpc2system(ppc, system)


def system2ppc(system) -> dict:
    """
    Alias for ``system2mpc``. Refer to :py:mod:`ams.io.matpower.system2mpc` for more details.

    Convert data from an AMS system to an mpc dict.

    In the ``gen`` section, slack generators preceeds PV generators.
    """
    mpc = system2mpc(system)
    np.set_printoptions(suppress=True)
    # Map the original bus indices to consecutive values
    # Adjust discontinuous bus indices
    BUS_I = mpc['bus'][:, 0].astype(int)
    bus_map = {busi0: i for i, busi0 in enumerate(BUS_I)}
    mpc['bus'][:, 0] = np.array([bus_map[busi0] for busi0 in BUS_I])
    mpc['gen'][:, 0] = np.array([bus_map[busi0] for busi0 in mpc['gen'][:, 0].astype(int)])
    mpc['branch'][:, 0] = np.array([bus_map[busi0] for busi0 in mpc['branch'][:, 0].astype(int)])
    mpc['branch'][:, 1] = np.array([bus_map[busi0] for busi0 in mpc['branch'][:, 1].astype(int)])
    return mpc
