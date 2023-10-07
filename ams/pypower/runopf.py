# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Runs an optimal power flow.
"""

import logging

from os.path import dirname, join

from ams.pypower.ppoption import ppoption
from ams.pypower.opf import opf
from ams.pypower.printpf import printpf
from ams.pypower.savecase import savecase


logger = logging.getLogger(__name__)


def runopf(casedata=None, ppopt=None, fname='', solvedcase=''):
    """
    Runs an optimal power flow.

    Parameters
    ----------
    casedata : str or dict, optional
        Input data for the problem. It can be a string (ppc) containing the file name of a PYPOWER case
        which defines the data matrices baseMVA, bus, gen, branch, and gencost (areas is not used at all, it is only included for
        backward compatibility of the API), a dict (ppc) containing the data matrices as fields, or the individual data matrices themselves.
        Default is None.

    ppopt : dict, optional
        PYPOWER options vector specifying options for the OPF solver. Default is None.

    fname : str, optional
        File name for saving the solved case data. Default is an empty string ('').

    solvedcase : str, optional
        File name for loading a previously solved case. Default is an empty string ('').

    Returns
    -------
    results : dict
        A dictionary containing the solved case data and additional fields.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    # default arguments
    if casedata is None:
        casedata = join(dirname(__file__), 'case9')
    ppopt = ppoption(ppopt)

    # -----  run the optimal power flow  -----
    r = opf(casedata, ppopt)

    # -----  output results  -----
    if fname:
        fd = None
        try:
            fd = open(fname, "a")
        except IOError as detail:
            logger.debug("Error opening %s: %s.\n" % (fname, detail))
        finally:
            if fd is not None:
                printpf(r, fd, ppopt)
                fd.close()

    else:
        # printpf(r, stdout, ppopt)
        pass

    # save solved case
    if solvedcase:
        savecase(solvedcase, r)

    return r


def rundcopf(casedata=None, ppopt=None, fname='', solvedcase=''):
    """
    Runs a DC optimal power flow.

    Parameters
    ----------
    casedata : str or dict, optional
        Input data for the problem. It can be a string (ppc) containing the file name of a PYPOWER case
        which defines the data matrices baseMVA, bus, gen, branch, and gencost (areas is not used at all, it is only included for
        backward compatibility of the API), a dict (ppc) containing the data matrices as fields, or the individual data matrices themselves.
        Default is None.

    ppopt : dict, optional
        PYPOWER options vector specifying options for the DC OPF solver. Default is None.

    fname : str, optional
        File name for saving the solved case data. Default is an empty string ('').

    solvedcase : str, optional
        File name for loading a previously solved case. Default is an empty string ('').

    Returns
    -------
    results : dict
        A dictionary containing the solved DC OPF case data and additional fields.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    # default arguments
    if casedata is None:
        casedata = join(dirname(__file__), 'case9')
    ppopt = ppoption(ppopt, PF_DC=True)

    return runopf(casedata, ppopt, fname, solvedcase)
