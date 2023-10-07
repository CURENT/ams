"""
Module to solve power flow.
"""

import logging
import os

from time import time

import numpy as np

from numpy import flatnonzero as find
from scipy.sparse.linalg import spsolve
from scipy.sparse import hstack, vstack

from andes.shared import deg2rad, rad2deg

from ams.pypower.bustypes import bustypes
from ams.pypower.loadcase import loadcase
from ams.pypower.core import ppoption
from ams.pypower.ppver import ppver

from ams.pypower.fdpf import fdpf
from ams.pypower.gausspf import gausspf
from ams.pypower.pfsoln import pfsoln
from ams.pypower.printpf import printpf
from ams.pypower.savecase import savecase

import ams.pypower.idx as idx
from ams.pypower.make import (makeB, makeBdc, makeSbus, makeYbus,
                                     dSbus_dV, dIbr_dV, dSbr_dV)


logger = logging.getLogger(__name__)


def runpf(casedata=None, ppopt=None, fname='', solvedcase=''):
    """
    Runs a power flow.

    Runs a power flow (full AC Newton's method by default) and optionally
    returns the solved values in the data matrices, a flag which is True if
    the algorithm was successful in finding a solution, and the elapsed
    time in seconds. All input arguments are optional. If casename is
    provided it specifies the name of the input data file or dict
    containing the power flow data. The default value is 'case9'.

    Parameters
    ----------
    casedata : str, dict, or None, optional
        The name of the input data file or a dict containing the power flow data.
        Default is None.
    ppopt : dict, optional
        PYPOWER options vector. It can be used to specify the solution algorithm
        and output options among other things. Default is None.
    fname : str, optional
        The name of the file to which the pretty printed output will be appended.
        Default is an empty string.
    solvedcase : str, optional
        The name of the solved case file. If it ends with '.mat', it saves the case
        as a MAT-file; otherwise, it saves it as a Python-file. Default is an empty
        string.

    Returns
    -------
    results : dict or None
        Solved power flow results. None if the power flow did not converge.
    success : bool
        True if the algorithm successfully found a solution, False otherwise.
    et : float
        Elapsed time in seconds for running the power flow.

    Notes
    -----
    If the ENFORCE_Q_LIMS option is set to True (default is False), then if any
    generator reactive power limit is violated after running the AC power flow,
    the corresponding bus is converted to a idx.bus['PQ'] bus, with Qg at the limit, and
    the case is re-run. The voltage magnitude at the bus will deviate from the
    specified value in order to satisfy the reactive power limit. If the reference
    bus is converted to idx.bus['PQ'], the first remaining idx.bus['PV'] bus will be used as the slack
    bus for the next iteration. This may result in the real power output at this
    generator being slightly off from the specified values.

    Enforcing of generator Q limits inspired by contributions from Mu Lin,
    Lincoln University, New Zealand (1/14/05).

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    info = dict(name='PYPOWER')  # solver stats

    # default arguments
    if casedata is None:
        casedata = os.path.join(os.path.dirname(__file__), 'case9')
    ppopt = ppoption(ppopt)

    # options
    verbose = ppopt["VERBOSE"]
    qlim = ppopt["ENFORCE_Q_LIMS"]  # enforce Q limits on gens?
    dc = ppopt["PF_DC"]  # use DC formulation?

    # read data
    ppc = loadcase(casedata)

    # add zero columns to branch for flows if needed
    if ppc["branch"].shape[1] < idx.branch['QT']:
        ppc["branch"] = np.c_[ppc["branch"],
                           np.zeros((ppc["branch"].shape[0],
                                  idx.branch['QT'] - ppc["branch"].shape[1] + 1))]

    # convert to internal indexing
    ppc = idx.ext2int(ppc)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    # get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    # generator info
    on = find(gen[:, idx.gen['GEN_STATUS']] > 0)  # which generators are on?
    gbus = gen[on, idx.gen['GEN_BUS']].astype(int)  # what buses are they at?

    # -----  run the power flow  -----
    t0 = time()
    v = ppver('all')
    logger.info('PYPOWER Version %s, %s' % (v["Version"], v["Date"]))

    if dc:                               # DC formulation
        # initial state
        Va0 = bus[:, idx.bus['VA']] * deg2rad

        # build B matrices and phase shift injections
        B, Bf, Pbusinj, Pfinj = makeBdc(baseMVA, bus, branch)

        # compute complex bus power injections [generation - load]
        # adjusted for phase shifters and real shunts
        Pbus = makeSbus(baseMVA, bus, gen).real - Pbusinj - bus[:, idx.bus['GS']] / baseMVA

        # "run" the power flow
        Va = dcpf(B, Pbus, Va0, ref, pv, pq)

        # update data matrices with solution
        branch[:, [idx.branch['QF'], idx.branch['QT']]] = np.zeros((branch.shape[0], 2))
        branch[:, idx.branch['PF']] = (Bf * Va + Pfinj) * baseMVA
        branch[:, idx.branch['PT']] = -branch[:, idx.branch['PF']]
        bus[:, idx.bus['VM']] = np.ones(bus.shape[0])
        bus[:, idx.bus['VA']] = Va * rad2deg
        # update Pg for slack generator (1st gen at ref bus)
        # (note: other gens at ref bus are accounted for in Pbus)
        # Pg = Pinj + Pload + Gs
        # newPg = oldPg + newPinj - oldPinj
        refgen = np.zeros(len(ref), dtype=int)
        for k in range(len(ref)):
            temp = find(gbus == ref[k])
            refgen[k] = on[temp[0]]
        gen[refgen, idx.gen['PG']] = gen[refgen, idx.gen['PG']] + (B[ref, :] * Va - Pbus[ref]) * baseMVA

        success = 1
    else:  # AC formulation
        alg = ppopt['PF_ALG']
        if alg == 1:
            solver = 'Newton'
        elif alg == 2:
            solver = 'fast-decoupled, XB'
        elif alg == 3:
            solver = 'fast-decoupled, BX'
        elif alg == 4:
            solver = 'Gauss-Seidel'
        else:
            solver = 'unknown'
        info['name'] += ' with ' + solver + ' method'

        # initial state
        # V0    = np.ones(bus.shape[0])            ## flat start
        V0 = bus[:, idx.bus['VM']] * np.exp(1j * deg2rad * bus[:, idx.bus['VA']])
        vcb = np.ones(V0.shape)    # create mask of voltage-controlled buses
        vcb[pq] = 0     # exclude idx.bus['PQ'] buses
        k = find(vcb[gbus])     # in-service gens at v-c buses
        V0[gbus[k]] = gen[on[k], idx.gen['VG']] / abs(V0[gbus[k]]) * V0[gbus[k]]

        if qlim:
            ref0 = ref  # save index and angle of
            Varef0 = bus[ref0, idx.bus['VA']]  # original reference bus(es)
            limited = []  # list of indices of gens @ Q lims
            fixedQg = np.zeros(gen.shape[0])  # Qg of gens at Q limits

        # build admittance matrices
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

        repeat = True
        while repeat:
            # compute complex bus power injections [generation - load]
            Sbus = makeSbus(baseMVA, bus, gen)

            # run the power flow
            alg = ppopt["PF_ALG"]
            if alg == 1:
                V, success, _ = newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
            elif alg == 2 or alg == 3:
                Bp, Bpp = makeB(baseMVA, bus, branch, alg)
                V, success, _ = fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, ppopt)
            elif alg == 4:
                V, success, _ = gausspf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
            else:
                logger.debug('Only Newton\'s method, fast-decoupled, and '
                             'Gauss-Seidel power flow algorithms currently '
                             'implemented.\n')

            # update data matrices with solution
            bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq)

            if qlim:  # enforce generator Q limits
                # find gens with violated Q constraints
                gen_status = gen[:, idx.gen['GEN_STATUS']] > 0
                qg_max_lim = gen[:, idx.gen['QG']] > gen[:, idx.gen['QMAX']] + ppopt["OPF_VIOLATION"]
                qg_min_lim = gen[:, idx.gen['QG']] < gen[:, idx.gen['QMIN']] - ppopt["OPF_VIOLATION"]

                mx = find(gen_status & qg_max_lim)
                mn = find(gen_status & qg_min_lim)

                if len(mx) > 0 or len(mn) > 0:  # we have some Q limit violations
                    # first check for INFEASIBILITY (all remaining gens violating)
                    infeas = np.union1d(mx, mn)
                    remaining = find(gen_status &
                                     (bus[gen[:, idx.gen['GEN_BUS']], idx.bus['BUS_TYPE']] == idx.bus['PV'] |
                                      bus[gen[:, idx.gen['GEN_BUS']], idx.bus['BUS_TYPE']] == idx.bus['REF']))
                    if len(infeas) == len(remaining) or all(infeas == remaining):
                        if verbose:
                            print('All %d remaining gens exceed to their Q limits: INFEASIBLE PROBLEM\n' % len(infeas))

                        success = 0
                        break

                    # one at a time?
                    if qlim == 2:  # fix largest violation, ignore the rest
                        k = np.argmax(np.r_[gen[mx, idx.gen['QG']] - gen[mx, idx.gen['QMAX']],
                                      gen[mn, idx.gen['QMIN']] - gen[mn, idx.gen['QG']]])
                        if k > len(mx):
                            mn = mn[k - len(mx)]
                            mx = []
                        else:
                            mx = mx[k]
                            mn = []

                    if verbose and len(mx) > 0:
                        for i in range(len(mx)):
                            print('Gen ' + str(mx[i] + 1) + ' at upper Q limit, converting to PQ bus\n')

                    if verbose and len(mn) > 0:
                        for i in range(len(mn)):
                            print('Gen ' + str(mn[i] + 1) + ' at lower Q limit, converting to PQ bus\n')

                    # save corresponding limit values
                    fixedQg[mx] = gen[mx, idx.gen['QMAX']]
                    fixedQg[mn] = gen[mn, idx.gen['QMIN']]
                    mx = np.r_[mx, mn].astype(int)

                    # convert to idx.bus['PQ'] bus
                    gen[mx, idx.gen['QG']] = fixedQg[mx]  # set Qg to binding
                    for i in range(len(mx)):  # [one at a time, since they may be at same bus]
                        gen[mx[i], idx.gen['GEN_STATUS']] = 0  # temporarily turn off gen,
                        bi = gen[mx[i], idx.gen['GEN_BUS']].astype(int)  # adjust load accordingly,
                        bus[bi, [idx.bus['PD'], idx.bus['QD']]] = (
                            bus[bi, [idx.bus['PD'], idx.bus['QD']]] - gen[mx[i], [idx.gen['PG'], idx.gen['QG']]])

                    if len(ref) > 1 and any(
                        bus[gen[mx, idx.gen['GEN_BUS']],
                            idx.bus['BUS_TYPE']] == idx.bus['REF']):
                        raise ValueError('Sorry, PYPOWER cannot enforce Q '
                                         'limits for slack buses in systems '
                                         'with multiple slacks.')

                    # & set bus type to idx.bus['PQ']
                    bus[gen[mx, idx.gen['GEN_BUS']].astype(int), idx.bus['BUS_TYPE']] = idx.bus['PQ']

                    # update bus index lists of each type of bus
                    ref_temp = ref
                    ref, pv, pq = bustypes(bus, gen)

                    # previous line can modify lists to select new idx.bus['REF'] bus
                    # if there was none, so we should update bus with these
                    # just to keep them consistent
                    if ref != ref_temp:
                        bus[ref, idx.bus['BUS_TYPE']] = idx.bus['REF']
                        bus[pv, idx.bus['BUS_TYPE']] = pv
                        if verbose:
                            print('Bus %d is new slack bus\n' % ref)

                    limited = np.r_[limited, mx].astype(int)
                else:
                    repeat = 0  # no more generator Q limits violated
            else:
                repeat = 0  # don't enforce generator Q limits, once is enough

        if qlim and len(limited) > 0:
            # restore injections from limited gens [those at Q limits]
            gen[limited, idx.gen['QG']] = fixedQg[limited]  # restore Qg value,
            for i in range(len(limited)):  # [one at a time, since they may be at same bus]
                bi = gen[limited[i], idx.gen['GEN_BUS']]  # re-adjust load,
                bus[bi, [idx.bus['PD'], idx.bus['QD']]] = bus[bi, [idx.bus['PD'],
                                                                       idx.bus['QD']]] + gen[limited[i], [idx.gen['PG'], idx.gen['QG']]]
                gen[limited[i], idx.gen['GEN_STATUS']] = 1  # and turn gen back on

            if ref != ref0:
                # adjust voltage angles to make original ref bus correct
                bus[:, idx.bus['VA']] = bus[:, idx.bus['VA']] - bus[ref0, idx.bus['VA']] + Varef0

    ppc["et"] = time() - t0
    ppc["success"] = success

    # -----  output results  -----
    # convert back to original bus numbering & print results
    ppc["bus"], ppc["gen"], ppc["branch"] = bus, gen, branch
    results = idx.int2ext(ppc)

    # zero out result fields of out-of-service gens & branches
    if len(results["order"]["gen"]["status"]["off"]) > 0:
        results["gen"][np.ix_(results["order"]["gen"]["status"]["off"], [idx.gen['PG'], idx.gen['QG']])] = 0

    if len(results["order"]["branch"]["status"]["off"]) > 0:
        results["branch"][np.ix_(results["order"]["branch"]["status"]["off"], [idx.branch['PF'],
                              idx.branch['QF'], idx.branch['PT'], idx.branch['QT']])] = 0

    if fname:
        fd = None
        try:
            fd = open(fname, "a")
        except Exception as detail:
            logger.debug("Error opening %s: %s.\n" % (fname, detail))
        finally:
            if fd is not None:
                printpf(results, fd, ppopt)
                fd.close()
    else:
        # printpf(results, stdout, ppopt)
        pass

    # save solved case
    if solvedcase:
        savecase(solvedcase, results)

    return results, success, info


def rundcpf(casedata=None, ppopt=None, fname='', solvedcase=''):
    """
    Solve DC power flow.
    """
    # default arguments
    if casedata is None:
        casedata = os.path.join(os.path.dirname(__file__), 'case9')
    ppopt = ppoption(ppopt, PF_DC=True)

    return runpf(casedata, ppopt, fname, solvedcase)


def dcpf(B, Pbus, Va0, ref, pv, pq):
    """
    Solves a DC power flow.

    Solves for the bus voltage angles at all but the reference bus, given the
    full system B matrix and the vector of bus real power injections, the
    initial vector of bus voltage angles (in radians), and column vectors with
    the lists of bus indices for the swing bus, idx.bus['PV'] buses, and idx.bus['PQ'] buses,
    respectively. Returns a vector of bus voltage angles in radians.

    Parameters
    ----------
    B : ndarray
        The system B matrix.
    Pbus : ndarray
        Vector of bus real power injections.
    Va0 : ndarray
        Initial vector of bus voltage angles (in radians).
    ref : int
        Index of the reference bus.
    pv : ndarray
        List of bus indices for idx.bus['PV'] buses.
    pq : ndarray
        List of bus indices for idx.bus['PQ'] buses.

    Returns
    -------
    Va : ndarray
        Vector of bus voltage angles in radians.

    Author
    ------
    Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad Autonoma de Manizales)

    Ray Zimmerman (PSERC Cornell)
    """
    pvpq = np.matrix(np.r_[pv, pq])

    # initialize result vector
    Va = np.copy(Va0)

    # update angles for non-reference buses
    Va[pvpq] = spsolve(B[pvpq.T, pvpq], np.transpose(Pbus[pvpq] - B[pvpq.T, ref] * Va0[ref]))

    return Va


def newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt=None):
    """Solves the power flow using a full Newton's method.

    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including ref bus) buses, and the reference angle of the
    swing bus, as well as an initial guess for remaining magnitudes and
    angles. C{ppopt} is a PYPOWER options vector which can be used to
    set the termination tolerance, maximum number of iterations, and
    output options (see L{ppoption} for details). Uses default options if
    this parameter is not given. Returns the final complex voltages, a
    flag which indicates whether it converged or not, and the number of
    iterations performed.

    @see: L{runpf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # default arguments
    if ppopt is None:
        ppopt = ppoption()

    # options
    tol = ppopt['PF_TOL']
    max_it = ppopt['PF_MAX_IT']
    verbose = ppopt['VERBOSE']

    # initialize
    converged = 0
    i = 0
    V = V0
    Va = np.angle(V)
    Vm = abs(V)

    # set up indexing for updating V
    pvpq = np.r_[pv, pq]
    npv = len(pv)
    npq = len(pq)
    j1 = 0
    j2 = npv  # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  # j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  # j5:j6 - V mag of pq buses

    # evaluate F(x0)
    mis = V * np.conj(Ybus * V) - Sbus
    F = np.r_[mis[pv].real,
           mis[pq].real,
           mis[pq].imag]

    # check tolerance
    normF = np.linalg.norm(F, np.Inf)
    if verbose > 1:
        logger.info('\n it    max P & Q mismatch (p.u.)')
        logger.info('\n----  ---------------------------')
        logger.info('\n%3d        %10.3e' % (i, normF))
    if normF < tol:
        converged = 1
        if verbose > 1:
            logger.info('\nConverged!\n')

    # do Newton iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1

        # evaluate Jacobian
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

        J11 = dS_dVa[np.array([pvpq]).T, pvpq].real
        J12 = dS_dVm[np.array([pvpq]).T, pq].real
        J21 = dS_dVa[np.array([pq]).T, pvpq].imag
        J22 = dS_dVm[np.array([pq]).T, pq].imag

        J = vstack([
            hstack([J11, J12]),
            hstack([J21, J22])
        ], format="csr")

        # compute update step
        dx = -1 * spsolve(J, F)

        # update voltage
        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        V = Vm * np.exp(1j * Va)
        Vm = abs(V)  # update Vm and Va again in case
        Va = np.angle(V)  # we wrapped around with a negative Vm

        # evalute F(x)
        mis = V * np.conj(Ybus * V) - Sbus
        F = np.r_[mis[pv].real,
               mis[pq].real,
               mis[pq].imag]

        # check for convergence
        normF = np.linalg.norm(F, np.Inf)
        if verbose > 1:
            logger.info('\n%3d        %10.3e' % (i, normF))
        if normF < tol:
            converged = 1
            if verbose:
                logger.info("\nNewton's method power flow converged in "
                            "%d iterations.\n" % i)

    if verbose:
        if not converged:
            logger.info("\nNewton's method power did not converge in %d "
                        "iterations.\n" % i)

    return V, converged, i
