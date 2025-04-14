"""
PowerWorld .aux file parser.
"""

import logging
import re

logger = logging.getLogger(__name__)


def testlines(infile):
    """
    Test if this file is in the PowerWorld .aux format.
    
    NOT YET IMPLEMENTED.
    """

    return True


def read(system, file):
    """
    Read a PowerWorld .aux data file into a PowerWorld dictionary (ppd),
    and build AMS device elements.
    """

    ppd = aux2ppd(file)
    return ppd2system(ppd, system)


def aux2ppd(infile: str) -> dict:
    """
    Read a PowerWorld .aux data file and return a PowerWorld dictionary (ppd).
    
    Parameters
    ----------
    infile : str
        Path to the input file.
    
    Returns
    -------
    dict
        PowerWorld dictionary (ppd).
    """

    # TODO: implement
    ppd = {}

    return ppd


def ppd2system(ppd: dict, system) -> bool:
    """
    Load a PowerWorld dictionary (ppd) into the AMS system object.

    Parameters
    ----------
    ppd : dict
        PowerWorld dictionary (ppd).
    system : ams.System.system
        AMS system object.

    Returns
    -------
    bool
        True if successful; False otherwise.
    """
    # TODO: implement

    return True
