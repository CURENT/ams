# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Runs an optimal power flow with unit-decommitment heuristic.
"""

from sys import stderr

from os.path import dirname, join

from ams.pypower.ppoption import ppoption
from ams.pypower.uopf import uopf
from ams.pypower.printpf import printpf
from ams.pypower.savecase import savecase


def runuopf(casedata=None, ppopt=None, fname='', solvedcase=''):
    """Runs an optimal power flow with unit-decommitment heuristic.

    @see: L{rundcopf}, L{runuopf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # default arguments
    if casedata is None:
        casedata = join(dirname(__file__), 'case9')
    ppopt = ppoption(ppopt)

    # -----  run the unit de-commitment / optimal power flow  -----
    r = uopf(casedata, ppopt)

    # -----  output results  -----
    if fname:
        fd = None
        try:
            fd = open(fname, "a")
        except Exception as detail:
            logger.debug("Error opening %s: %s.\n" % (fname, detail))
        finally:
            if fd is not None:
                printpf(r, fd, ppopt)
                fd.close()

    else:
        # printpf(r, stdout, ppopt=ppopt)
        pass

    # save solved case
    if solvedcase:
        savecase(solvedcase, r)

    return r


def runduopf(casedata=None, ppopt=None, fname='', solvedcase=''):
    """Runs a DC optimal power flow with unit-decommitment heuristic.

    @see: L{rundcopf}, L{runuopf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # default arguments
    if casedata is None:
        casedata = join(dirname(__file__), 'case9')
    ppopt = ppoption(ppopt, PF_DC=True)

    return runuopf(casedata, ppopt, fname, solvedcase)
