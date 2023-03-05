"""
PYPOWER reader for AMS.
"""
import logging

from ams.io.matpower import mpc2system, system2mpc
from ams.solver.ipp import load_ppc

logger = logging.getLogger(__name__)


def testlines(infile):
    """
    Test if this file is in the PYPOWER format.

    NOT YET IMPLEMENTED.
    """
    return True


def read(system, file):
    """
    Read a PYPOWER data file into ppc, and build andes device elements.

    TODO: add support for PYPOWER file in future versions.
    """
    ppc = load_ppc(file)
    return ppc2system(ppc, system)


def ppc2system(ppc: dict, system) -> bool:
    """
    Alias for ``mpc2system``.

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
    Alias for ``system2mpc``.

    Convert data from an AMS system to an mpc dict.

    In the ``gen`` section, slack generators preceeds PV generators.
    """
    return system2mpc(system)
