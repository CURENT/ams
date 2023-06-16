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
    mpc2system(ppc, system)
    return True


def system2ppc(system) -> dict:
    """
    Alias for ``system2mpc``. Refer to :py:mod:`ams.io.matpower.system2mpc` for more details.

    Convert data from an AMS system to an mpc dict.

    In the ``gen`` section, slack generators preceeds PV generators.
    """
    mpc = system2mpc(system)
    np.set_printoptions(suppress=True)
    # Map the original bus indices to consecutive values
    # adjust the bus index to start from 0
    BUS_I = mpc['bus'][:, 0].astype(int)
    if np.max(mpc['bus'][:, 0]) > mpc['bus'].shape[0]:
        logger.warning('The bus index is not continuous, adjusted automatically.')
        # Find the unique indices in busi
        old_bus = np.unique(BUS_I)
        # Generate a mapping dictionary to map the unique indices to consecutive values
        mapping = {busi0: i for i, busi0 in enumerate(old_bus)}
        # Map the original bus indices to consecutive values
        # BUS_I
        mpc['bus'][:, 0] = np.array([mapping[busi0] for busi0 in BUS_I])
        # GEN_BUS
        GEN_BUS = mpc['gen'][:, 0].astype(int)
        mpc['gen'][:, 0] = np.array([mapping[busi0] for busi0 in GEN_BUS])
        # F_BUS
        F_BUS = mpc['branch'][:, 0].astype(int)
        mpc['branch'][:, 0] = np.array([mapping[busi0] for busi0 in F_BUS])
        # T_BUS
        T_BUS = mpc['branch'][:, 1].astype(int)
        mpc['branch'][:, 1] = np.array([mapping[busi0] for busi0 in T_BUS])
    if np.min(mpc['bus'][:, 0]) > 0:
        logger.debug('Adjust bus index to start from 0.')
        mpc['bus'][:, 0] -= 1   # BUS_I
        mpc['gen'][:, 0] -= 1   # GEN_BUS
        mpc['branch'][:, 0] -= 1    # F_BUS
        mpc['branch'][:, 1] -= 1    # T_BUS
    return mpc
