"""
PYPOWER reader for AMS.
"""


from ams.solver.ipp import load_ppc


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
    raise NotImplementedError('PYPOWER reader is not yet implemented.')
    return True
