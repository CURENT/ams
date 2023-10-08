"""
Module to solve OPF.
"""

import logging

import os

from ams.pypower.core import ppoption
from ams.pypower.routines.opf import fopf
from ams.pypower.printpf import printpf
from ams.pypower.savecase import savecase

from ams.pypower.uopf import uopf
from ams.pypower.printpf import printpf
from ams.pypower.savecase import savecase

from ams.pypower.loadcase import loadcase
from ams.pypower.toggle_reserves import toggle_reserves

logger = logging.getLogger(__name__)


def runopf(casedata, ppopt):
    """Runs an optimal power flow.

    @see: L{rundcopf}, L{runuopf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    sstats = dict(solver_name='PYPOWER',
                  num_iters=1)  # solver stats
    ppopt = ppoption(ppopt)

    # -----  run the optimal power flow  -----
    r = fopf(casedata, ppopt)
    sstats['solver_name'] = 'PYPOWER-PIPS'
    sstats['num_iters'] = r['raw']['output']['iterations']
    return r, sstats


def runuopf(casedata=None, ppopt=None, fname='', solvedcase=''):
    """Runs an optimal power flow with unit-decommitment heuristic.

    @see: L{rundcopf}, L{runuopf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # default arguments
    if casedata is None:
        casedata = os.path.join(os.path.dirname(__file__), 'case9')
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
        casedata = os.path.join(os.path.dirname(__file__), 'case9')
    ppopt = ppoption(ppopt, PF_DC=True)

    return runuopf(casedata, ppopt, fname, solvedcase)


def runopf_w_res(*args):
    """Runs an optimal power flow with fixed zonal reserves.

    Runs an optimal power flow with the addition of reserve requirements
    specified as a set of fixed zonal reserves. See L{runopf} for a
    description of the input and output arguments, which are the same,
    with the exception that the case file or dict C{casedata} must define
    a 'reserves' field, which is a dict with the following fields:
        - C{zones}   C{nrz x ng}, C{zone(i, j) = 1}, if gen C{j} belongs
        to zone C{i} 0, otherwise
        - C{req}     C{nrz x 1}, zonal reserve requirement in MW
        - C{cost}    (C{ng} or C{ngr}) C{x 1}, cost of reserves in $/MW
        - C{qty}     (C{ng} or C{ngr}) C{x 1}, max quantity of reserves
        in MW (optional)
    where C{nrz} is the number of reserve zones and C{ngr} is the number of
    generators belonging to at least one reserve zone and C{ng} is the total
    number of generators.

    In addition to the normal OPF output, the C{results} dict contains a
    new 'reserves' field with the following fields, in addition to those
    provided in the input:
        - C{R}       - C{ng x 1}, reserves provided by each gen in MW
        - C{Rmin}    - C{ng x 1}, lower limit on reserves provided by
        each gen, (MW)
        - C{Rmax}    - C{ng x 1}, upper limit on reserves provided by
        each gen, (MW)
        - C{mu.l}    - C{ng x 1}, shadow price on reserve lower limit, ($/MW)
        - C{mu.u}    - C{ng x 1}, shadow price on reserve upper limit, ($/MW)
        - C{mu.Pmax} - C{ng x 1}, shadow price on C{Pg + R <= Pmax}
        constraint, ($/MW)
        - C{prc}     - C{ng x 1}, reserve price for each gen equal to
        maximum of the shadow prices on the zonal requirement constraint
        for each zone the generator belongs to

    See L{t.t_case30_userfcns} for an example case file with fixed reserves,
    and L{toggle_reserves} for the implementation.

    Calling syntax options::
        results = runopf_w_res(casedata)
        results = runopf_w_res(casedata, ppopt)
        results = runopf_w_res(casedata, ppopt, fname)
        results = runopf_w_res(casedata, [popt, fname, solvedcase)
        results, success = runopf_w_res(...)

    Example::
        results = runopf_w_res('t_case30_userfcns')

    @see: L{runopf}, L{toggle_reserves}, L{t.t_case30_userfcns}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ppc = loadcase(args[0])
    ppc = toggle_reserves(ppc, 'on')

    r = runopf(ppc, *args[1:])
    r = toggle_reserves(r, 'off')

    return r
