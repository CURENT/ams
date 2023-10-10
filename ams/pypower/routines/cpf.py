"""
Runs a full AC continuation power flow.
"""
# --- run cpf ---
import logging

import numpy as np
from numpy import flatnonzero as find

import scipy.sparse as sp  # NOQA

from andes.shared import deg2rad  # NOQA
from andes.utils.misc import elapsed  # NOQA

import ams.pypower.utils as putil  # NOQA
import ams.pypower.io as pio  # NOQA
import ams.pypower.routines.opffcns as opfcn  # NOQA
from ams.pypower.idx import IDX  # NOQA
from ams.pypower.make import (makeSbus, makeYbus, dSbus_dV)  # NOQA
from ams.pypower.routines.pflow import newtonpf, pfsoln  # NOQA
from ams.pypower.core import ppoption  # NOQA
import ams.pypower.routines.cpf_callbacks as cpf_callbacks  # NOQA

logger = logging.getLogger(__name__)


def runcpf(casedata, ppopt=None, scale=1.2):
    """
    Runs a full AC continuation power flow.
    """
    ppopt = ppoption(ppopt)

    # options
    verbose = ppopt["VERBOSE"]
    step = ppopt["CPF_STEP"]
    parameterization = ppopt["CPF_PARAMETERIZATION"]
    adapt_step = ppopt["CPF_ADAPT_STEP"]
    cb_args = ppopt["CPF_USER_CALLBACK_ARGS"]

    # set up callbacks
    callback_names = ["cpf_default_callback"]
    if len(ppopt["CPF_USER_CALLBACK"]) > 0:
        if isinstance(ppopt["CPF_USER_CALLBACK"], list):
            callback_names = np.r_[callback_names, ppopt["CPF_USER_CALLBACK"]]
        else:
            callback_names.append(ppopt["CPF_USER_CALLBACK"])
    callbacks = []
    for callback_name in callback_names:
        callbacks.append(getattr(cpf_callbacks, callback_name))

    # read base case data
    ppcbase = pio.loadcase(casedata)
    nb = ppcbase["bus"].shape[0]

    # add zero columns to branch for flows if needed
    if ppcbase["branch"].shape[1] < IDX.branch.QT:
        ppcbase["branch"] = np.c_[ppcbase["branch"],
                                  np.zeros((ppcbase["branch"].shape[0],
                                            IDX.branch.QT - ppcbase["branch"].shape[1] + 1))]

    # convert to internal indexing
    ppcbase = opfcn.ext2int(ppcbase)
    baseMVAb, busb, genb, branchb = \
        ppcbase["baseMVA"], ppcbase["bus"], ppcbase["gen"], ppcbase["branch"]

    # get bus index lists of each type of bus
    ref, pv, pq = putil.bustypes(busb, genb)

    # generator info
    onb = find(genb[:, IDX.gen.GEN_STATUS] > 0)  # which generators are on?
    gbusb = genb[onb, IDX.gen.GEN_BUS].astype(int)  # what buses are they at?

    # scale the load and generation of base case as target case
    ppctarget = pio.loadcase(casedata)
    ppctarget["bus"][:, IDX.bus.PD] *= scale
    ppctarget["bus"][:, IDX.bus.QD] *= scale
    ppctarget["gen"][:, IDX.gen.PG] *= scale
    ppctarget["gen"][:, IDX.gen.QG] *= scale

    # add zero columns to branch for flows if needed
    if ppctarget["branch"].shape[1] < IDX.branch.QT:
        ppctarget["branch"] = np.c_[ppctarget["branch"],
                                    np.zeros((ppctarget["branch"].shape[0],
                                              IDX.branch.QT - ppctarget["branch"].shape[1] + 1))]

    # convert to internal indexing
    ppctarget = opfcn.ext2int(ppctarget)
    baseMVAt, bust, gent, brancht = \
        ppctarget["baseMVA"], ppctarget["bus"], ppctarget["gen"], ppctarget["branch"]

    # get bus index lists of each type of bus
    # ref, pv, pq = putil.bustypes(bust, gent)

    # generator info
    ont = find(gent[:, IDX.gen.GEN_STATUS] > 0)  # which generators are on?
    # gbust = gent[ont, IDX.gen.GEN_BUS].astype(int)  # what buses are they at?

    # -----  run the power flow  -----
    t0, _ = elapsed()  # start timer

    # initial state
    # V0    = np.ones((bus.shape[0])            ## flat start
    V0 = busb[:, IDX.bus.VM] * np.exp((1j * deg2rad * busb[:, IDX.bus.VA]))
    vcb = np.ones(V0.shape)    # create mask of voltage-controlled buses
    vcb[pq] = 0     # exclude PQ buses
    k = find(vcb[gbusb])    # in-service gens at v-c buses
    V0[gbusb[k]] = genb[onb[k], IDX.gen.VG] / abs(V0[gbusb[k]]) * V0[gbusb[k]]

    # build admittance matrices
    Ybus, Yf, Yt = makeYbus(baseMVAb, busb, branchb)

    # compute base case complex bus power injections (generation - load)
    Sbusb = makeSbus(baseMVAb, busb, genb)
    # compute target case complex bus power injections (generation - load)
    Sbust = makeSbus(baseMVAt, bust, gent)

    # scheduled transfer
    Sxfr = Sbust - Sbusb

    # Run the base case power flow solution
    lam = 0
    logger.info('Solve base case power flow')
    V, success, iterations = newtonpf(Ybus, Sbusb, V0, ref, pv, pq, ppopt)

    logger.info('Start full AC CPF')
    logger.info('%2d: lambda = %6.3f, %2d Newton steps' % (0, 0, iterations))

    lamprv = lam    # lam at previous step
    Vprv = V    # V at previous step
    continuation = 1
    cont_steps = 0

    # input args for callbacks
    cb_data = dict(ppc_base=ppcbase, ppc_target=ppctarget,
                   Sxfr=Sxfr, Ybus=Ybus, Yf=Yf, Yt=Yt,
                   ref=ref, pv=pv, pq=pq, ppopt=ppopt)
    cb_state = {}

    # invoke callbacks
    for k in range(len(callbacks)):
        cb_state, _ = callbacks[k](cont_steps, V, lam, V, lam,
                                   cb_data, cb_state, cb_args)

    if np.linalg.norm(Sxfr) == 0:
        logger.error('base case and target case have identical load and generation!')

        continuation = 0
        V0 = V
        lam0 = lam

    # tangent predictor z = [dx;dlam]
    z = np.zeros(2*len(V)+1)
    z[-1] = 1.0
    while continuation:
        cont_steps = cont_steps + 1
        # prediction for next step
        V0, lam0, z = cpf_predictor(V, lam, Ybus, Sxfr, pv, pq, step, z,
                                    Vprv, lamprv, parameterization)

        # save previous voltage, lambda before updating
        Vprv = V
        lamprv = lam

        # correction
        V, success, i, lam = cpf_corrector(Ybus, Sbusb, V0, ref, pv, pq,
                                           lam0, Sxfr, Vprv, lamprv, z,
                                           step, parameterization, ppopt)

        if not success:
            continuation = 0
            logger.warning('%2d: lambda = %6.3f, corrector did not converge in %d iterations' % (
                cont_steps, lam, i))
            break

        logger.info('%2d: lambda = %6.3f, %2d corrector Newton steps' %
                    (cont_steps, lam, i))

        # invoke callbacks
        for k in range(len(callbacks)):
            cb_state, _ = callbacks[k](cont_steps, V, lam, V0, lam0,
                                       cb_data, cb_state, cb_args)

        if isinstance(ppopt["CPF_STOP_AT"], str):
            if ppopt["CPF_STOP_AT"].upper() == "FULL":
                if abs(lam) < 1e-8:     # traced the full continuation curve
                    logger.info('Reached steady state loading limit in %d continuation steps' % cont_steps)
                    continuation = 0
                elif lam < lamprv and lam - step < 0:    # next step will overshoot
                    step = lam      # modify step-size
                    parameterization = 1    # change to natural parameterization
                    adapt_step = False      # disable step-adaptivity

            else:   # == 'NOSE'
                if lam < lamprv:    # reached the nose point
                    logger.info('Reached steady state loading limit in %d continuation steps' % cont_steps)
                    continuation = 0

        else:
            if lam < lamprv:
                logger.info('Reached steady state loading limit in %d continuation steps' % cont_steps)
                continuation = 0
            elif abs(ppopt["CPF_STOP_AT"] - lam) < 1e-8:     # reached desired lambda
                logger.info('Reached desired lambda %3.2f in %d continuation steps' % (
                    ppopt["CPF_STOP_AT"], cont_steps))
                continuation = 0
            # will reach desired lambda in next step
            elif lam + step > ppopt["CPF_STOP_AT"]:
                step = ppopt["CPF_STOP_AT"] - lam   # modify step-size
                parameterization = 1    # change to natural parameterization
                adapt_step = False      # disable step-adaptivity

        if adapt_step and continuation:
            pvpq = np.r_[pv, pq]
            # Adapt stepsize
            cpf_error = np.linalg.norm(np.r_[np.angle(V[pq]), abs(
                V[pvpq]), lam] - np.r_[np.angle(V0[pq]), abs(V0[pvpq]), lam0], np.Inf)
            if cpf_error < ppopt["CPF_ERROR_TOL"]:
                # Increase stepsize
                step = step * ppopt["CPF_ERROR_TOL"] / cpf_error
                if step > ppopt["CPF_STEP_MAX"]:
                    step = ppopt["CPF_STEP_MAX"]
            else:
                # decrese stepsize
                step = step * ppopt["CPF_ERROR_TOL"] / cpf_error
                if step < ppopt["CPF_STEP_MIN"]:
                    step = ppopt["CPF_STEP_MIN"]

    # invoke callbacks
    if success:
        cpf_results = {}
        for k in range(len(callbacks)):
            cb_state, cpf_results = callbacks[k](cont_steps, V, lam, V0, lam0,
                                                 cb_data, cb_state, cb_args, results=cpf_results, is_final=True)
    else:
        cpf_results = {}
        cpf_results["iterations"] = i

    # update bus and gen matrices to reflect the loading and generation
    # at the noise point
    bust[:, IDX.bus.PD] = busb[:, IDX.bus.PD] + lam * (bust[:, IDX.bus.PD] - busb[:, IDX.bus.PD])
    bust[:, IDX.bus.QD] = busb[:, IDX.bus.QD] + lam * (bust[:, IDX.bus.QD] - busb[:, IDX.bus.QD])
    gent[:, IDX.gen.PG] = genb[:, IDX.gen.PG] + lam * (gent[:, IDX.gen.PG] - genb[:, IDX.gen.PG])

    # update data matrices with solution
    bust, gent, brancht = pfsoln(baseMVAt, bust, gent, brancht,
                                 Ybus, Yf, Yt, V, ref, pv, pq)

    ppctarget["et"] = elapsed(t0)
    ppctarget["success"] = success

    # -----  output results  -----
    # convert back to original bus numbering & print results
    ppctarget["bus"], ppctarget["gen"], ppctarget["branch"] = bust, gent, brancht
    if success:
        n = cpf_results["iterations"] + 1
        cpf_results["V_p"] = opfcn.i2e_data(
            ppctarget, cpf_results["V_p"], np.full((nb, n), np.NaN), "bus", 0)
        cpf_results["V_c"] = opfcn.i2e_data(
            ppctarget, cpf_results["V_c"], np.full((nb, n), np.NaN), "bus", 0)
    results = opfcn.int2ext(ppctarget)
    results["cpf"] = cpf_results

    # zero out result fields of out-of-service gens & branches
    if len(results["order"]["gen"]["status"]["off"]) > 0:
        results["gen"][np.ix_(results["order"]["gen"]
                              ["status"]["off"], [IDX.gen.QF, IDX.gen.QG])] = 0

    if len(results["order"]["branch"]["status"]["off"]) > 0:
        results["branch"][np.ix_(results["order"]["branch"]
                                 ["status"]["off"], [IDX.branch.PF, IDX.branch.QF, IDX.branch.PT, IDX.branch.QT])] = 0

    if ppopt["CPF_PLOT_LEVEL"]:
        import matplotlib.pyplot as plt
        plt.show()
    sstats = dict(solver_name='PYPOWER', num_iters=cpf_results["iterations"])
    return results, success, sstats


def cpf_predictor(V, lam, Ybus, Sxfr, pv, pq,
                  step, z, Vprv, lamprv, parameterization):
    # sizes
    pvpq = np.r_[pv, pq]
    nb = len(V)
    npv = len(pv)
    npq = len(pq)

    # compute Jacobian for the power flow equations
    dSbus_dVm, dSbus_dVa = dSbus_dV(Ybus, V)

    j11 = dSbus_dVa[np.array([pvpq]).T, pvpq].real
    j12 = dSbus_dVm[np.array([pvpq]).T, pq].real
    j21 = dSbus_dVa[np.array([pq]).T, pvpq].imag
    j22 = dSbus_dVm[np.array([pq]).T, pq].imag

    J = sp.vstack([
        sp.hstack([j11, j12]),
        sp.hstack([j21, j22])
    ], format="csr")

    dF_dlam = -np.r_[Sxfr[pvpq].real, Sxfr[pq].imag].reshape((-1, 1))
    dP_dV, dP_dlam = cpf_p_jac(parameterization, z, V, lam, Vprv, lamprv, pv, pq)

    # linear operator for computing the tangent predictor
    J = sp.vstack([
        sp.hstack([J, dF_dlam]),
        sp.hstack([dP_dV, dP_dlam])
    ], format="csr")

    Vaprv = np.angle(V)
    Vmprv = abs(V)

    # compute normalized tangent predictor
    s = np.zeros(npv+2*npq+1)
    s[-1] = 1
    z[np.r_[pvpq, nb+pq, 2*nb]] = sp.linalg.spsolve(J, s)
    z = z / np.linalg.norm(z)

    Va0 = Vaprv
    Vm0 = Vmprv
    lam0 = lam

    # prediction for next step
    Va0[pvpq] = Vaprv[pvpq] + step * z[pvpq]
    Vm0[pq] = Vmprv[pq] + step * z[nb+pq]
    lam0 = lam + step * z[2*nb]
    V0 = Vm0 * np.exp((1j * Va0))

    return V0, lam0, z


def cpf_corrector(Ybus, Sbus, V0, ref, pv, pq,
                  lam0, Sxfr, Vprv, lamprv, z,
                  step, parameterization, ppopt):

    # default arguments
    if ppopt is None:
        ppopt = ppoption(ppopt)

    # options
    verbose = ppopt["VERBOSE"]
    tol = ppopt["PF_TOL"]
    max_it = ppopt["PF_MAX_IT"]

    # initialize
    converged = 0
    i = 0
    V = V0
    Va = np.angle(V)
    Vm = abs(V)
    lam = lam0

    # set up indexing for updating V
    pvpq = np.r_[pv, pq]
    npv = len(pv)
    npq = len(pq)
    nb = len(V)
    j1 = 0
    j2 = npv    # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq   # j1:j2 - V angle of pv buses
    j5 = j4
    j6 = j4 + npq   # j5:j6 - V mag of pq buses
    j7 = j6
    j8 = j6 + 1    # j7:j8 - lambda

    # evaluate F(x0, lam0), including Sxfr transfer/loading

    mis = V * np.conj(Ybus.dot(V)) - Sbus - lam*Sxfr
    F = np.r_[mis[pvpq].real, mis[pq].imag]

    # evaluate P(x0, lambda0)
    P = cpf_p(parameterization, step, z, V, lam, Vprv, lamprv, pv, pq)

    # augment F(x, lambda) with P(x, lambda)
    F = np.r_[F, P]

    # check tolerance
    normF = np.linalg.norm(F, np.Inf)
    logger.debug('CPF correction')
    logger.debug('%2d: |F(x)| = %.10g', i, normF)
    if normF < tol:
        converged = 1

    # do Newton iterations
    while not converged and i < max_it:
        # update iteration counter
        i = i + 1

        # evaluate Jacobian
        dSbus_dVm, dSbus_dVa = dSbus_dV(Ybus, V)

        j11 = dSbus_dVa[np.array([pvpq]).T, pvpq].real
        j12 = dSbus_dVm[np.array([pvpq]).T, pq].real
        j21 = dSbus_dVa[np.array([pq]).T, pvpq].imag
        j22 = dSbus_dVm[np.array([pq]).T, pq].imag

        J = sp.vstack([
            sp.hstack([j11, j12]),
            sp.hstack([j21, j22])
        ], format="csr")

        dF_dlam = -np.r_[Sxfr[pvpq].real, Sxfr[pq].imag].reshape((-1, 1))
        dP_dV, dP_dlam = cpf_p_jac(parameterization, z, V, lam, Vprv, lamprv, pv, pq)

        # augment J with real/imag -Sxfr and z^T
        J = sp.vstack([
            sp.hstack([J, dF_dlam]),
            sp.hstack([dP_dV, dP_dlam])
        ], format="csr")

        # compute update step
        dx = -1 * sp.linalg.spsolve(J, F)

        # update voltage
        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        V = Vm * np.exp((1j * Va))
        Vm = abs(V)
        Va = np.angle(V)

        # update lambda
        lam = lam + dx[j7:j8]

        # evalute F(x, lam)
        mis = V * np.conj(Ybus.dot(V)) - Sbus - lam*Sxfr
        F = np.r_[mis[pv].real, mis[pq].real, mis[pq].imag]

        # evaluate P(x, lambda)
        P = cpf_p(parameterization, step, z, V, lam, Vprv, lamprv, pv, pq)

        # augment F(x, lambda) with P(x, lambda)
        F = np.r_[F, P]

        # check for convergence
        normF = np.linalg.norm(F, np.Inf)
        logger.debug('%2d: |F(x)| = %.10g', i, normF)

        if normF < tol:
            converged = 1

    if not converged:
        logger.error('Newton''s method corrector did not converge in %d iterations.' % i)

    return V, converged, i, lam


def cpf_p_jac(parameterization, z, V, lam, Vprv, lamprv, pv, pq):
    if parameterization == 1:
        npv = len(pv)
        npq = len(pq)
        dP_dV = np.zeros(npv+2*npq)
        if lam >= lamprv:
            dP_dlam = 1.0
        else:
            dP_dlam = -1.0

    elif parameterization == 2:
        pvpq = np.r_[pv, pq]

        Va = np.angle(V)
        Vm = abs(V)
        Vaprv = np.angle(Vprv)
        Vmprv = abs(Vprv)
        dP_dV = 2 * (np.r_[Va[pvpq], Vm[pq]] -
                     np.r_[Vaprv[pvpq], Vmprv[pq]])
        if lam == lamprv:
            dP_dlam = 1.0
        else:
            dP_dlam = 2 * (lam - lamprv)

    elif parameterization == 3:
        nb = len(V)
        dP_dV = z[np.r_[pv, pq, nb+pq]]
        dP_dlam = z[2 * nb]

    return dP_dV, dP_dlam


def cpf_p(parameterization, step, z, V, lam, Vprv, lamprv, pv, pq):
    # evaluate P(x0, lambda0)
    if parameterization == 1:
        if lam >= lamprv:
            P = lam - lamprv - step
        else:
            P = lamprv - lam - step

    elif parameterization == 2:
        pvpq = np.r_[pv, pq]

        Va = np.angle(V)
        Vm = abs(V)
        Vaprv = np.angle(Vprv)
        Vmprv = abs(Vprv)
        P = sum(np.square(np.r_[Va[pvpq], Vm[pq], lam] -
                          np.r_[Vaprv[pvpq], Vmprv[pq], lamprv])) - np.square(step)

    elif parameterization == 3:
        pvpq = np.r_[pv, pq]

        nb = len(V)
        Va = np.angle(V)
        Vm = abs(V)
        Vaprv = np.angle(Vprv)
        Vmprv = abs(Vprv)
        P = np.dot(z[np.r_[pvpq, nb+pq, 2*nb]].reshape((1, -1)),
                   np.array(np.r_[Va[pvpq], Vm[pq], lam] - np.r_[Vaprv[pvpq], Vmprv[pq], lamprv]).T) - step

    return P
