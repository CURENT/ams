"""
PYPOWER module to solve power flow.
"""

import logging  # NOQA

from time import time  # NOQA

import numpy as np  # NOQA

from numpy import flatnonzero as find  # NOQA
from scipy.sparse.linalg import spsolve, splu  # NOQA
from scipy.sparse import hstack, vstack  # NOQA
from scipy.sparse import csr_matrix as c_sparse  # NOQA

from andes.shared import deg2rad, rad2deg  # NOQA

from ams.pypower.core import ppoption  # NOQA
import ams.pypower.utils as putils  # NOQA
from ams.pypower.idx import IDX  # NOQA
from ams.pypower.io import loadcase  # NOQA
from ams.pypower.eps import EPS  # NOQA
import ams.pypower.routines.opffcns as opfcn  # NOQA

from ams.pypower.make import (makeB, makeBdc, makeSbus, makeYbus, dSbus_dV)  # NOQA


logger = logging.getLogger(__name__)


def runpf(casedata, ppopt):
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
    the corresponding bus is converted to a IDX.bus.PQ bus, with Qg at the limit, and
    the case is re-run. The voltage magnitude at the bus will deviate from the
    specified value in order to satisfy the reactive power limit. If the reference
    bus is converted to IDX.bus.PQ, the first remaining IDX.bus.PV bus will be used as the slack
    bus for the next iteration. This may result in the real power output at this
    generator being slightly off from the specified values.

    Enforcing of generator Q limits inspired by contributions from Mu Lin,
    Lincoln University, New Zealand (1/14/05).

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    sstats = dict(solver_name='PYPOWER',
                  num_iters=1)  # solver stats
    ppopt = ppoption(ppopt)

    # read data
    ppc = loadcase(casedata)

    # add zero columns to branch for flows if needed
    if ppc["branch"].shape[1] < IDX.branch.QT:
        ppc["branch"] = np.c_[ppc["branch"],
                              np.zeros((ppc["branch"].shape[0],
                                        IDX.branch.QT - ppc["branch"].shape[1] + 1))]

    # convert to internal indexing
    ppc = opfcn.ext2int(ppc)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    # get bus index lists of each type of bus
    ref, pv, pq = putils.bustypes(bus, gen)

    # generator info
    on = find(gen[:, IDX.gen.GEN_STATUS] > 0)  # which generators are on?
    gbus = gen[on, IDX.gen.GEN_BUS].astype(int)  # what buses are they at?

    # -----  run the power flow  -----
    t0 = time()

    if ppopt["PF_DC"]:                               # DC formulation
        # initial state
        Va0 = bus[:, IDX.bus.VA] * deg2rad

        # build B matrices and phase shift injections
        B, Bf, Pbusinj, Pfinj, _ = makeBdc(baseMVA, bus, branch)

        # compute complex bus power injections [generation - load]
        # adjusted for phase shifters and real shunts
        Pbus = makeSbus(baseMVA, bus, gen).real - Pbusinj - bus[:, IDX.bus.GS] / baseMVA

        # "run" the power flow
        Va = dcpf(B, Pbus, Va0, ref, pv, pq)

        # update data matrices with solution
        branch[:, [IDX.branch.QF, IDX.branch.QT]] = np.zeros((branch.shape[0], 2))
        branch[:, IDX.branch.PF] = (Bf * Va + Pfinj) * baseMVA
        branch[:, IDX.branch.PT] = -branch[:, IDX.branch.PF]
        bus[:, IDX.bus.VM] = np.ones(bus.shape[0])
        bus[:, IDX.bus.VA] = Va * rad2deg
        # update Pg for slack generator (1st gen at ref bus)
        # (note: other gens at ref bus are accounted for in Pbus)
        # Pg = Pinj + Pload + Gs
        # newPg = oldPg + newPinj - oldPinj
        refgen = np.zeros(len(ref), dtype=int)
        for k in range(len(ref)):
            temp = find(gbus == ref[k])
            refgen[k] = on[temp[0]]
        gen[refgen, IDX.gen.PG] = gen[refgen, IDX.gen.PG] + (B[ref, :] * Va - Pbus[ref]) * baseMVA

        success = 1
    else:  # AC formulation
        method_map = {1: 'Newton', 2: 'fast-decoupled, XB',
                      3: 'fast-decoupled, BX', 4: 'Gauss-Seidel'}
        alg = method_map.get(ppopt['PF_ALG'])
        sstats['solver_name'] = f'PYPOWER-{alg}'
        logger.debug(f"Solution method: {alg}'s method.")
        if alg is None:
            logger.debug('Only Newton\'s method, fast-decoupled, and '
                         'Gauss-Seidel power flow algorithms currently '
                         'implemented.\n')
            raise ValueError

        # initial state
        # V0    = np.ones(bus.shape[0])            ## flat start
        V0 = bus[:, IDX.bus.VM] * np.exp(1j * deg2rad * bus[:, IDX.bus.VA])
        vcb = np.ones(V0.shape)    # create mask of voltage-controlled buses
        vcb[pq] = 0     # exclude IDX.bus.PQ buses
        k = find(vcb[gbus])     # in-service gens at v-c buses
        V0[gbus[k]] = gen[on[k], IDX.gen.VG] / abs(V0[gbus[k]]) * V0[gbus[k]]

        if ppopt["ENFORCE_Q_LIMS"]:
            ref0 = ref  # save index and angle of
            Varef0 = bus[ref0, IDX.bus.VA]  # original reference bus(es)
            limited = []  # list of indices of gens @ Q lims
            fixedQg = np.zeros(gen.shape[0])  # Qg of gens at Q limits

        # build admittance matrices
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

        repeat = True
        while repeat:
            # compute complex bus power injections [generation - load]
            Sbus = makeSbus(baseMVA, bus, gen)

            # run the power flow
            if ppopt['PF_ALG'] == 1:
                V, success, sstats['num_iters'] = newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
            elif ppopt['PF_ALG'] in (2, 3):
                Bp, Bpp = makeB(baseMVA, bus, branch, alg)
                V, success, sstats['num_iters'] = fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, ppopt)
            elif ppopt['PF_ALG'] == 4:
                V, success, sstats['num_iters'] = gausspf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
            else:
                pass
                # update data matrices with solution
            bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq)

            enforce_q_lims = ppopt["ENFORCE_Q_LIMS"]
            logger.debug(f'ENFORCE_Q_LIMS={enforce_q_lims}')
            if ppopt["ENFORCE_Q_LIMS"]:  # enforce generator Q limits
                # find gens with violated Q constraints
                gen_status = gen[:, IDX.gen.GEN_STATUS] > 0
                qg_max_lim = gen[:, IDX.gen.QG] > gen[:, IDX.gen.QMAX] + ppopt["OPF_VIOLATION"]
                qg_min_lim = gen[:, IDX.gen.QG] < gen[:, IDX.gen.QMIN] - ppopt["OPF_VIOLATION"]

                mx = find(gen_status & qg_max_lim)
                mn = find(gen_status & qg_min_lim)

                if len(mx) > 0 or len(mn) > 0:  # we have some Q limit violations
                    # first check for INFEASIBILITY (all remaining gens violating)
                    gen_violate = np.union1d(mx, mn)
                    gen_bus = gen[:, IDX.gen.GEN_BUS].astype(int)
                    cond_pv = bus[gen_bus, IDX.bus.BUS_TYPE] == IDX.bus.PV
                    cond_ref = bus[gen_bus, IDX.bus.BUS_TYPE] == IDX.bus.REF
                    gen_remain = find(gen_status & (cond_pv | cond_ref))
                    msg = f'gen_violate = {gen_violate}\nremaining = {gen_remain}'
                    logger.debug(msg)
                    # TODO: condition can be improved, but not sure now
                    if len(gen_remain) <= 0:
                        msg = f'Infeasible: remaining {len(gen_violate)} gens exceed Q limits.'
                        logger.warning(msg)
                        success = 0
                        break

                    # one at a time?
                    if ppopt["ENFORCE_Q_LIMS"] == 2:  # fix largest violation, ignore the rest
                        k = np.argmax(np.r_[gen[mx, IDX.gen.QG] - gen[mx, IDX.gen.QMAX],
                                      gen[mn, IDX.gen.QMIN] - gen[mn, IDX.gen.QG]])
                        if k > len(mx):
                            mn = mn[k - len(mx)]
                            mx = []
                        else:
                            mx = mx[k]
                            mn = []

                    if len(mx) > 0:
                        msg = 'Following Gen convert to PQ bus because'
                        msg += ' exceed Q upper limits:\n'
                        msg += ', '.join(str(i + 1) for i in mx)
                        logger.debug(msg)
                    if len(mn) > 0:
                        msg = 'Following Gen convert to PQ bus because'
                        msg += ' exceed Q lower limits:\n'
                        msg += ', '.join(str(i + 1) for i in mn)
                        logger.debug(msg)

                    # save corresponding limit values
                    fixedQg[mx] = gen[mx, IDX.gen.QMAX]
                    fixedQg[mn] = gen[mn, IDX.gen.QMIN]
                    mx = np.r_[mx, mn].astype(int)

                    # convert to PQ bus
                    # Convert generators PV bus to PQ bus
                    for i in range(len(mx)):
                        idx = mx[i]
                        gen[idx, IDX.gen.QG] = fixedQg[idx]  # Set Qg to binding
                        gen[idx, IDX.gen.GEN_STATUS] = 0  # Temporarily turn off generator
                        bi = int(gen[idx, IDX.gen.GEN_BUS])  # Get the bus index
                        bus[bi, [IDX.bus.PD, IDX.bus.QD]] -= gen[idx, [IDX.gen.PG, IDX.gen.QG]]  # Adjust load

                    if len(ref) > 1 and any(bus[gen[mx, IDX.gen.GEN_BUS], IDX.bus.BUS_TYPE] == IDX.bus.REF):
                        raise ValueError("PYPOWER cannot enforce Q limits for systems with multiple slack buses. "
                                         "Please ensure there is only one slack bus in the system.")

                    # set bus type to PQ
                    bus[gen[mx, IDX.gen.GEN_BUS].astype(int), IDX.bus.BUS_TYPE] = IDX.bus.PQ

                    # update bus index lists of each type of bus
                    ref_temp = ref
                    ref, pv, pq = putils.bustypes(bus, gen)

                    # previous line can modify lists to select new REF bus
                    # if there was none, so we should update bus with these
                    # just to keep them consistent
                    if ref != ref_temp:
                        bus[ref, IDX.bus.BUS_TYPE] = IDX.bus.REF
                        bus[pv, IDX.bus.BUS_TYPE] = pv
                        logger.debug(f'Bus<{ref[0]}> is new slack bus')

                    limited = np.r_[limited, mx].astype(int)
                else:
                    repeat = 0  # no more generator Q limits violated
            else:
                repeat = 0  # don't enforce generator Q limits, once is enough

            if ppopt["ENFORCE_Q_LIMS"] and len(limited) > 0:
                # Restore injections from limited gens [those at Q limits]
                for i in range(len(limited)):
                    idx = limited[i]
                    gen[idx, IDX.gen.QG] = fixedQg[idx]  # Restore Qg value
                    bi = int(gen[idx, IDX.gen.GEN_BUS])  # Get the bus index
                    bus[bi, [IDX.bus.PD, IDX.bus.QD]] += gen[idx,
                                                             [IDX.gen.PG, IDX.gen.QG]]  # Re-adjust load
                    gen[idx, IDX.gen.GEN_STATUS] = 1  # Turn generator back on

                if ref != ref0:
                    # adjust voltage angles to make original ref bus correct
                    bus[:, IDX.bus.VA] = bus[:, IDX.bus.VA] - bus[ref0, IDX.bus.VA] + Varef0

    ppc["et"] = time() - t0
    ppc["success"] = success

    # -----  output results  -----
    # convert back to original bus numbering & print results
    ppc["bus"], ppc["gen"], ppc["branch"] = bus, gen, branch
    results = opfcn.int2ext(ppc)

    # zero out result fields of out-of-service gens & branches
    if len(results["order"]["gen"]["status"]["off"]) > 0:
        results["gen"][np.ix_(results["order"]["gen"]["status"]["off"], [IDX.gen.PG, IDX.gen.QG])] = 0

    if len(results["order"]["branch"]["status"]["off"]) > 0:
        results["branch"][
            np.ix_(
                results["order"]["branch"]["status"]["off"],
                [IDX.branch.PF,
                 IDX.branch.QF,
                 IDX.branch.PT,
                 IDX.branch.QT])] = 0

    return results, success, sstats


def dcpf(B, Pbus, Va0, ref, pv, pq):
    """
    Solves a DC power flow.

    Solves for the bus voltage angles at all but the reference bus, given the
    full system B matrix and the vector of bus real power injections, the
    initial vector of bus voltage angles (rad), and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively.

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
        List of bus indices for IDX.bus.PV buses.
    pq : ndarray
        List of bus indices for IDX.bus.PQ buses.

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


def newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt):
    """
    Solves the power flow using a full Newton's method.

    Parameters
    ----------
    Ybus : array-like
        Full system admittance matrix (for all buses).
    Sbus : array-like
        Complex bus power injection vector (for all buses).
    V0 : array-like
        Initial vector of complex bus voltages.
    ref : int
        Index of the swing bus.
    pv : list of int
        List of bus indices for PV buses.
    pq : list of int
        List of bus indices for PQ buses.
    ppopt : dict, optional
        PYPOWER options vector which can be used to set the termination tolerance,
        maximum number of iterations, and output options (see ppoption for details).
        Default is None, which uses default options.

    Returns
    -------
    V : array-like
        Final complex voltages.
    converged : bool
        Flag indicating whether the power flow converged or not.
    i : int
        Number of iterations performed.

    Notes
    -----
    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including the reference bus) buses, and the reference angle
    of the swing bus, as well as an initial guess for remaining magnitudes
    and angles. Uses default options if the ppopt parameter is not given.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    ppopt = ppoption(ppopt)
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
    logger.info('%2d: |F(x)| = %.10g', i, normF)
    converged = normF < tol

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
        logger.info('%2d: |F(x)| = %.10g', i, normF)
        converged = normF < tol

    return V, converged, i


def pfsoln(baseMVA, bus0, gen0, branch0, Ybus, Yf, Yt, V, ref, pv, pq):
    """
    Updates bus, gen, and branch data structures to match power flow solution.

    This function takes the following inputs and updates the input data structures:

    Parameters
    ----------
    baseMVA : float
        Base power in MVA.
    bus0 : ndarray
        Initial bus data structure.
    gen0 : ndarray
        Initial gen data structure.
    branch0 : ndarray
        Initial branch data structure.
    Ybus : sparse matrix
        Admittance matrix.
    Yf : sparse matrix
        Admittance matrix for "from" end of branches.
    Yt : sparse matrix
        Admittance matrix for "to" end of branches.
    V : ndarray
        Bus voltage magnitude in per unit.
    ref : int
        Reference bus index.
    pv : ndarray
        PV bus indices.
    pq : ndarray
        PQ bus indices.

    Returns
    -------
    bus : ndarray
        Updated bus data structure.
    gen : ndarray
        Updated gen data structure.
    branch : ndarray
        Updated branch data structure.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    # initialize return values
    bus = bus0
    gen = gen0
    branch = branch0

    # ----- update bus voltages -----
    bus[:, IDX.bus.VM] = abs(V)
    bus[:, IDX.bus.VA] = np.angle(V) * rad2deg

    # ----- update Qg for all gens and Pg for slack bus(es) -----
    # generator info
    on = find(gen[:, IDX.gen.GEN_STATUS] > 0)  # which generators are on?
    gbus = gen[on, IDX.gen.GEN_BUS].astype(int)  # what buses are they at?

    # compute total injected bus powers
    Sbus = V[gbus] * np.conj(Ybus[gbus, :] * V)

    # update Qg for all generators
    gen[:, IDX.gen.QG] = np.zeros(gen.shape[0])  # zero out all Qg
    gen[on, IDX.gen.QG] = Sbus.imag * baseMVA + bus[gbus, IDX.bus.QD]  # inj Q + local Qd
    # ... at this point any buses with more than one generator will have
    # the total Q dispatch for the bus assigned to each generator. This
    # must be split between them. We do it first equally, then in proportion
    # to the reactive range of the generator.

    if len(on) > 1:
        # build connection matrix, element i, j is 1 if gen on(i) at bus j is ON
        nb = bus.shape[0]
        ngon = on.shape[0]
        Cg = c_sparse((np.ones(ngon), (range(ngon), gbus)), (ngon, nb))

        # divide Qg by number of generators at the bus to distribute equally
        ngg = Cg * Cg.sum(0).T  # ngon x 1, number of gens at this gen's bus
        ngg = np.asarray(ngg).flatten()  # 1D array
        gen[on, IDX.gen.QG] = gen[on, IDX.gen.QG] / ngg

        # divide proportionally
        Cmin = c_sparse((gen[on, IDX.gen.QMIN], (range(ngon), gbus)), (ngon, nb))
        Cmax = c_sparse((gen[on, IDX.gen.QMAX], (range(ngon), gbus)), (ngon, nb))
        Qg_tot = Cg.T * gen[on, IDX.gen.QG]  # nb x 1 vector of total Qg at each bus
        Qg_min = Cmin.sum(0).T  # nb x 1 vector of min total Qg at each bus
        Qg_max = Cmax.sum(0).T  # nb x 1 vector of max total Qg at each bus
        Qg_min = np.asarray(Qg_min).flatten()  # 1D array
        Qg_max = np.asarray(Qg_max).flatten()  # 1D array
        # gens at buses with Qg range = 0
        ig = find(Cg * Qg_min == Cg * Qg_max)
        Qg_save = gen[on[ig], IDX.gen.QG]
        gen[on, IDX.gen.QG] = gen[on, IDX.gen.QMIN] + \
            (Cg * ((Qg_tot - Qg_min) / (Qg_max - Qg_min + EPS))) * \
            (gen[on, IDX.gen.QMAX] - gen[on, IDX.gen.QMIN])  # ^ avoid div by 0
        gen[on[ig], IDX.gen.QG] = Qg_save  # (terms are mult by 0 anyway)

    # update Pg for slack bus(es)
    # inj P + local Pd
    for k in range(len(ref)):
        refgen = find(gbus == ref[k])  # which is(are) the reference gen(s)?
        gen[on[refgen[0]], IDX.gen.PG] = \
            Sbus[refgen[0]].real * baseMVA + bus[ref[k], IDX.bus.PD]
        if len(refgen) > 1:  # more than one generator at this ref bus
            # subtract off what is generated by other gens at this bus
            gen[on[refgen[0]], IDX.gen.PG] = \
                gen[on[refgen[0]], IDX.gen.PG] - sum(gen[on[refgen[1:len(refgen)]], IDX.gen.PG])

    # ----- update/compute branch power flows -----
    out = find(branch[:, IDX.branch.BR_STATUS] == 0)  # out-of-service branches
    br = find(branch[:, IDX.branch.BR_STATUS]).astype(int)  # in-service branches

    # complex power at "from" bus
    Sf = V[branch[br, IDX.branch.F_BUS].astype(int)] * np.conj(Yf[br, :] * V) * baseMVA
    # complex power injected at "to" bus
    St = V[branch[br, IDX.branch.T_BUS].astype(int)] * np.conj(Yt[br, :] * V) * baseMVA
    branch[np.ix_(br, [IDX.branch.PF, IDX.branch.QF, IDX.branch.PT, IDX.branch.QT])
           ] = np.c_[Sf.real, Sf.imag, St.real, St.imag]
    branch[np.ix_(out, [IDX.branch.PF, IDX.branch.QF, IDX.branch.PT, IDX.branch.QT])
           ] = np.zeros((len(out), 4))

    return bus, gen, branch


def gausspf(Ybus, Sbus, V0, ref, pv, pq, ppopt):
    """
    Solves the power flow using a Gauss-Seidel method.
    This method seems to be much more slower than Newton's method,
    not fully checked yet.

    Parameters
    ----------
    Ybus : array-like
        Full system admittance matrix (for all buses).
    Sbus : array-like
        Complex bus power injection vector (for all buses).
    V0 : array-like
        Initial vector of complex bus voltages.
    ref : int
        Index of the swing bus.
    pv : list of int
        List of bus indices for PV buses.
    pq : list of int
        List of bus indices for PQ buses.
    ppopt : dict
        PYPOWER options vector which can be used to set the termination tolerance,
        maximum number of iterations, and output options (see ppoption for details).

    Returns
    -------
    V : array-like
        Final complex voltages.
    converged : bool
        Flag indicating whether the power flow converged or not.
    i : int
        Number of iterations performed.

    Notes
    -----
    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including the reference bus) buses, and the reference angle
    of the swing bus, as well as an initial guess for remaining magnitudes
    and angles. Uses default options if the ppopt parameter is not given.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)

    Alberto Borghetti (University of Bologna, Italy)
    """
    ppopt = ppoption(ppopt)
    # options
    tol = ppopt['PF_TOL']
    max_it = ppopt['PF_MAX_IT_GS']

    # initialize
    converged = 0
    i = 0
    V = V0.copy()
    # Va = angle(V)
    Vm = abs(V)

    # set up indexing for updating V
    npv = len(pv)
    npq = len(pq)
    pvpq = np.r_[pv, pq]

    # evaluate F(x0)
    mis = V * np.conj(Ybus * V) - Sbus
    F = np.r_[mis[pvpq].real,
              mis[pq].imag]

    # check tolerance
    normF = np.linalg.norm(F, np.Inf)
    logger.info('%d: |F(x)| = %.10g', i, normF)
    converged = normF < tol

    # do Gauss-Seidel iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1

        # update voltage
        # at PQ buses
        for k in pq[list(range(npq))]:
            tmp = (np.conj(Sbus[k] / V[k]) - Ybus[k, :] * V) / Ybus[k, k]
            V[k] = V[k] + tmp.item()

        # at PV buses
        if npv:
            for k in pv[list(range(npv))]:
                tmp = (V[k] * np.conj(Ybus[k, :] * V)).imag
                Sbus[k] = Sbus[k].real + 1j * tmp.item()
                tmp = (np.conj(Sbus[k] / V[k]) - Ybus[k, :] * V) / Ybus[k, k]
                V[k] = V[k] + tmp.item()
#               V[k] = Vm[k] * V[k] / abs(V[k])
            V[pv] = Vm[pv] * V[pv] / abs(V[pv])

        # evalute F(x)
        mis = V * np.conj(Ybus * V) - Sbus
        F = np.r_[mis[pv].real,
                  mis[pq].real,
                  mis[pq].imag]

        # check for convergence
        normF = np.linalg.norm(F, np.Inf)
        logger.info('%d: |F(x)| = %.10g', i, normF)
        converged = normF < tol

    return V, converged, i


def fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, ppopt):
    """
    Solves the power flow using a fast decoupled method.

    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, the FDPF matrices B prime
    and B double prime, and column vectors with the lists of bus indices
    for the swing bus, PV buses, and PQ buses, respectively. The bus voltage
    vector contains the set point for generator (including the reference bus)
    buses, and the reference angle of the swing bus, as well as an initial
    guess for remaining magnitudes and angles. `ppopt` is a PYPOWER options
    vector which can be used to set the termination tolerance, maximum
    number of iterations, and output options (see `ppoption` for details).
    Uses default options if this parameter is not given. Returns the
    final complex voltages, a flag which indicates whether it converged
    or not, and the number of iterations performed.

    Parameters
    ----------
    Ybus : ndarray
        Full system admittance matrix (for all buses).
    Sbus : ndarray
        Complex bus power injection vector (for all buses).
    V0 : ndarray
        Initial vector of complex bus voltages.
    Bp : ndarray
        FDPF matrix B prime.
    Bpp : ndarray
        FDPF matrix B double prime.
    ref : int
        Index of the reference bus.
    pv : ndarray
        List of bus indices for PV buses.
    pq : ndarray
        List of bus indices for PQ buses.
    ppopt : dict
        PYPOWER options vector.

    Returns
    -------
    V : ndarray
        Final complex voltages.
    converged : bool
        Flag indicating whether the power flow converged.
    i : int
        Number of iterations performed.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    ppopt = ppoption(ppopt)
    # options
    tol = ppopt['PF_TOL']
    max_it = ppopt['PF_MAX_IT_FD']
    verbose = ppopt['VERBOSE']

    # initialize
    converged = 0
    i = 0
    V = V0
    Va = np.angle(V)
    Vm = abs(V)

    # set up indexing for updating V
    # npv = len(pv)
    # npq = len(pq)
    pvpq = np.r_[pv, pq]

    # evaluate initial mismatch
    mis = (V * np.conj(Ybus * V) - Sbus) / Vm
    P = mis[pvpq].real
    Q = mis[pq].imag

    # check tolerance
    normP = np.linalg.norm(P, np.Inf)
    normQ = np.linalg.norm(Q, np.Inf)
    logger.info(f'{i} --: |P| = {normP:.10g}, |Q| = {normQ:.10g}')
    converged = normP < tol and normQ < tol

    # reduce B matrices
    Bp = Bp[np.array([pvpq]).T, pvpq].tocsc()  # splu requires a CSC matrix
    Bpp = Bpp[np.array([pq]).T, pq].tocsc()

    # factor B matrices
    Bp_solver = splu(Bp)
    Bpp_solver = splu(Bpp)

    # do P and Q iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1

        # -----  do P iteration, update Va  -----
        dVa = -Bp_solver.solve(P)

        # update voltage
        Va[pvpq] = Va[pvpq] + dVa
        V = Vm * np.exp(1j * Va)

        # evalute mismatch
        mis = (V * np.conj(Ybus * V) - Sbus) / Vm
        P = mis[pvpq].real
        Q = mis[pq].imag

        # check tolerance
        normP = np.linalg.norm(P, np.Inf)
        normQ = np.linalg.norm(Q, np.Inf)
        logger.info(f'{i} P: |P| = {normP:.10g}, |Q| = {normQ:.10g}')
        if normP < tol and normQ < tol:
            converged = 1
            break

        # -----  do Q iteration, update Vm  -----
        dVm = -Bpp_solver.solve(Q)

        # update voltage
        Vm[pq] = Vm[pq] + dVm
        V = Vm * np.exp(1j * Va)

        # evalute mismatch
        mis = (V * np.conj(Ybus * V) - Sbus) / Vm
        P = mis[pvpq].real
        Q = mis[pq].imag

        # check tolerance
        normP = np.linalg.norm(P, np.Inf)
        normQ = np.linalg.norm(Q, np.Inf)
        logger.info(f'{i} Q: |P| = {normP:.10g}, |Q| = {normQ:.10g}')
        converged = normP < tol and normQ < tol
        if converged:
            break

    return V, converged, i
