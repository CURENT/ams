"""
Module for OPF functions.
"""
import logging
from copy import deepcopy

import numpy as np  # NOQA
from numpy import flatnonzero as find  # NOQA

import scipy.sparse as sp  # NOQA
from scipy.sparse import csr_matrix as c_sparse  # NOQA
from scipy.sparse import lil_matrix as l_sparse  # NOQA

from ams.pypower.utils import isload, get_reorder, set_reorder  # NOQA
from ams.pypower.idx import IDX  # NOQA
from ams.pypower.make import (d2Sbus_dV2, dSbus_dV, dIbr_dV,
                              d2AIbr_dV2, d2ASbr_dV2, dSbr_dV,
                              makeSbus, dAbr_dV)  # NOQA

logger = logging.getLogger(__name__)


def opf_hessfcn(x, lmbda, om, Ybus, Yf, Yt, ppopt, il=None, cost_mult=1.0):
    """
    Evaluates Hessian of Lagrangian for AC OPF.

    Hessian evaluation function for AC optimal power flow, suitable
    for use with L{pips}.

    Examples::
        Lxx = opf_hessfcn(x, lmbda, om, Ybus, Yf, Yt, ppopt)
        Lxx = opf_hessfcn(x, lmbda, om, Ybus, Yf, Yt, ppopt, il)
        Lxx = opf_hessfcn(x, lmbda, om, Ybus, Yf, Yt, ppopt, il, cost_mult)

    @param x: optimization vector
    @param lmbda: C{eqnonlin} - Lagrange multipliers on power balance
    equations. C{ineqnonlin} - Kuhn-Tucker multipliers on constrained
    branch flows.
    @param om: OPF model object
    @param Ybus: bus admittance matrix
    @param Yf: admittance matrix for "from" end of constrained branches
    @param Yt: admittance matrix for "to" end of constrained branches
    @param ppopt: PYPOWER options vector
    @param il: (optional) vector of branch indices corresponding to
    branches with flow limits (all others are assumed to be unconstrained).
    The default is C{range(nl)} (all branches). C{Yf} and C{Yt} contain
    only the rows corresponding to C{il}.
    @param cost_mult: (optional) Scale factor to be applied to the cost
    (default = 1).

    @return: Hessian of the Lagrangian.

    @see: L{opf_costfcn}, L{opf_consfcn}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    # ----- initialize -----
    # unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]
    cp = om.get_cost_params()
    N, Cw, H, dd, rh, kk, mm = \
        cp["N"], cp["Cw"], cp["H"], cp["dd"], cp["rh"], cp["kk"], cp["mm"]
    vv, _, _, _ = om.get_idx()

    # unpack needed parameters
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of branches
    ng = gen.shape[0]  # number of dispatchable injections
    nxyz = len(x)  # total number of control vars of all types

    # set default constrained lines
    if il is None:
        il = np.arange(nl)  # all lines have limits by default
    nl2 = len(il)  # number of constrained lines

    # grab Pg & Qg
    Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]  # active generation in p.u.
    Qg = x[vv["i1"]["Qg"]:vv["iN"]["Qg"]]  # reactive generation in p.u.

    # put Pg & Qg back in gen
    gen[:, IDX.gen.PG] = Pg * baseMVA  # active generation in MW
    gen[:, IDX.gen.QG] = Qg * baseMVA  # reactive generation in MVAr

    # reconstruct V
    Va = x[vv["i1"]["Va"]:vv["iN"]["Va"]]
    Vm = x[vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    V = Vm * np.exp(1j * Va)
    nxtra = nxyz - 2 * nb
    pcost = gencost[np.arange(ng), :]
    if gencost.shape[0] > ng:
        qcost = gencost[np.arange(ng, 2 * ng), :]
    else:
        qcost = np.array([])

    # ----- evaluate d2f -----
    d2f_dPg2 = np.zeros(ng)  # c_sparse((ng, 1))               ## w.r.t. p.u. Pg
    d2f_dQg2 = np.zeros(ng)  # c_sparse((ng, 1))               ## w.r.t. p.u. Qg
    ipolp = find(pcost[:, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL)
    d2f_dPg2[ipolp] = \
        baseMVA**2 * polycost(pcost[ipolp, :], Pg[ipolp] * baseMVA, 2)
    if np.any(qcost):  # Qg is not free
        ipolq = find(qcost[:, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL)
        d2f_dQg2[ipolq] = \
            baseMVA**2 * polycost(qcost[ipolq, :], Qg[ipolq] * baseMVA, 2)
    i = np.r_[np.arange(vv["i1"]["Pg"], vv["iN"]["Pg"]),
              np.arange(vv["i1"]["Qg"], vv["iN"]["Qg"])]
#    d2f = c_sparse((sp.vstack([d2f_dPg2, d2f_dQg2]).toarray().flatten(),
#                  (i, i)), shape=(nxyz, nxyz))
    d2f = c_sparse((np.r_[d2f_dPg2, d2f_dQg2], (i, i)), (nxyz, nxyz))

    # generalized cost
    if sp.issparse(N) and N.nnz > 0:
        nw = N.shape[0]
        r = N * x - rh  # Nx - rhat
        iLT = find(r < -kk)  # below dead zone
        iEQ = find((r == 0) & (kk == 0))  # dead zone doesn't exist
        iGT = find(r > kk)  # above dead zone
        iND = np.r_[iLT, iEQ, iGT]  # rows that are Not in the Dead region
        iL = find(dd == 1)  # rows using linear function
        iQ = find(dd == 2)  # rows using quadratic function
        LL = c_sparse((np.ones(len(iL)), (iL, iL)), (nw, nw))
        QQ = c_sparse((np.ones(len(iQ)), (iQ, iQ)), (nw, nw))
        kbar = c_sparse((np.r_[np.ones(len(iLT)), np.zeros(len(iEQ)), -np.ones(len(iGT))],
                         (iND, iND)), (nw, nw)) * kk
        rr = r + kbar  # apply non-dead zone shift
        M = c_sparse((mm[iND], (iND, iND)), (nw, nw))  # dead zone or scale
        diagrr = c_sparse((rr, (np.arange(nw), np.arange(nw))), (nw, nw))

        # linear rows multiplied by rr(i), quadratic rows by rr(i)^2
        w = M * (LL + QQ * diagrr) * rr
        HwC = H * w + Cw
        AA = N.T * M * (LL + 2 * QQ * diagrr)

        d2f = d2f + AA * H * AA.T + 2 * N.T * M * QQ * \
            c_sparse((HwC, (np.arange(nw), np.arange(nw))), (nw, nw)) * N
    d2f = d2f * cost_mult

    # ----- evaluate Hessian of power balance constraints -----
    nlam = int(len(lmbda["eqnonlin"]) / 2)
    lamP = lmbda["eqnonlin"][:nlam]
    lamQ = lmbda["eqnonlin"][nlam:nlam + nlam]
    Gpaa, Gpav, Gpva, Gpvv = d2Sbus_dV2(Ybus, V, lamP)
    Gqaa, Gqav, Gqva, Gqvv = d2Sbus_dV2(Ybus, V, lamQ)

    d2G = sp.vstack([
        sp.hstack([
            sp.vstack([sp.hstack([Gpaa, Gpav]),
                    sp.hstack([Gpva, Gpvv])]).real +
            sp.vstack([sp.hstack([Gqaa, Gqav]),
                    sp.hstack([Gqva, Gqvv])]).imag,
            c_sparse((2 * nb, nxtra))]),
        sp.hstack([
            c_sparse((nxtra, 2 * nb)),
            c_sparse((nxtra, nxtra))
        ])
    ], "csr")

    # ----- evaluate Hessian of flow constraints -----
    nmu = int(len(lmbda["ineqnonlin"]) / 2)
    muF = lmbda["ineqnonlin"][:nmu]
    muT = lmbda["ineqnonlin"][nmu:nmu + nmu]
    if ppopt['OPF_FLOW_LIM'] == 2:  # current
        dIf_dVa, dIf_dVm, dIt_dVa, dIt_dVm, If, It = dIbr_dV(branch, Yf, Yt, V)
        Hfaa, Hfav, Hfva, Hfvv = d2AIbr_dV2(dIf_dVa, dIf_dVm, If, Yf, V, muF)
        Htaa, Htav, Htva, Htvv = d2AIbr_dV2(dIt_dVa, dIt_dVm, It, Yt, V, muT)
    else:
        f = branch[il, IDX.branch.F_BUS].astype(int)  # list of "from" buses
        t = branch[il, IDX.branch.T_BUS].astype(int)  # list of "to" buses
        # connection matrix for line & from buses
        Cf = c_sparse((np.ones(nl2), (np.arange(nl2), f)), (nl2, nb))
        # connection matrix for line & to buses
        Ct = c_sparse((np.ones(nl2), (np.arange(nl2), t)), (nl2, nb))
        dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St = \
            dSbr_dV(branch[il, :], Yf, Yt, V)
        if ppopt['OPF_FLOW_LIM'] == 1:  # real power
            Hfaa, Hfav, Hfva, Hfvv = d2ASbr_dV2(dSf_dVa.real, dSf_dVm.real,
                                                Sf.real, Cf, Yf, V, muF)
            Htaa, Htav, Htva, Htvv = d2ASbr_dV2(dSt_dVa.real, dSt_dVm.real,
                                                St.real, Ct, Yt, V, muT)
        else:  # apparent power
            Hfaa, Hfav, Hfva, Hfvv = \
                d2ASbr_dV2(dSf_dVa, dSf_dVm, Sf, Cf, Yf, V, muF)
            Htaa, Htav, Htva, Htvv = \
                d2ASbr_dV2(dSt_dVa, dSt_dVm, St, Ct, Yt, V, muT)

    d2H = sp.vstack([
        sp.hstack([
            sp.vstack([sp.hstack([Hfaa, Hfav]),
                    sp.hstack([Hfva, Hfvv])]) +
            sp.vstack([sp.hstack([Htaa, Htav]),
                    sp.hstack([Htva, Htvv])]),
            c_sparse((2 * nb, nxtra))
        ]),
        sp.hstack([
            c_sparse((nxtra, 2 * nb)),
            c_sparse((nxtra, nxtra))
        ])
    ], "csr")

    # -----  do numerical check using (central) finite differences  -----
    if 0:
        nx = len(x)
        step = 1e-5
        num_d2f = c_sparse((nx, nx))
        num_d2G = c_sparse((nx, nx))
        num_d2H = c_sparse((nx, nx))
        for i in range(nx):
            xp = x
            xm = x
            xp[i] = x[i] + step / 2
            xm[i] = x[i] - step / 2
            # evaluate cost & gradients
            _, dfp = opf_costfcn(xp, om)
            _, dfm = opf_costfcn(xm, om)
            # evaluate constraints & gradients
            _, _, dHp, dGp = opf_consfcn(xp, om, Ybus, Yf, Yt, ppopt, il)
            _, _, dHm, dGm = opf_consfcn(xm, om, Ybus, Yf, Yt, ppopt, il)
            num_d2f[:, i] = cost_mult * (dfp - dfm) / step
            num_d2G[:, i] = (dGp - dGm) * lmbda["eqnonlin"] / step
            num_d2H[:, i] = (dHp - dHm) * lmbda["ineqnonlin"] / step
        d2f_err = max(max(abs(d2f - num_d2f)))
        d2G_err = max(max(abs(d2G - num_d2G)))
        d2H_err = max(max(abs(d2H - num_d2H)))
        if d2f_err > 1e-6:
            print('Max difference in d2f: %g' % d2f_err)
        if d2G_err > 1e-5:
            print('Max difference in d2G: %g' % d2G_err)
        if d2H_err > 1e-6:
            print('Max difference in d2H: %g' % d2H_err)

    return d2f + d2G + d2H


def opf_consfcn(x, om, Ybus, Yf, Yt, ppopt, il=None, *args):
    """
    Evaluates nonlinear constraints and their Jacobian for OPF.

    Constraint evaluation function for AC optimal power flow, suitable
    for use with L{pips}. Computes constraint vectors and their gradients.

    @param x: optimization vector
    @param om: OPF model object
    @param Ybus: bus admittance matrix
    @param Yf: admittance matrix for "from" end of constrained branches
    @param Yt: admittance matrix for "to" end of constrained branches
    @param ppopt: PYPOWER options vector
    @param il: (optional) vector of branch indices corresponding to
    branches with flow limits (all others are assumed to be
    unconstrained). The default is C{range(nl)} (all branches).
    C{Yf} and C{Yt} contain only the rows corresponding to C{il}.

    @return: C{h} - vector of inequality constraint values (flow limits)
    limit^2 - flow^2, where the flow can be apparent power real power or
    current, depending on value of C{OPF_FLOW_LIM} in C{ppopt} (only for
    constrained lines). C{g} - vector of equality constraint values (power
    balances). C{dh} - (optional) inequality constraint gradients, column
    j is gradient of h(j). C{dg} - (optional) equality constraint gradients.

    @see: L{opf_costfcn}, L{opf_hessfcn}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    # ----- initialize -----

    # unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    vv, _, _, _ = om.get_idx()

    # problem dimensions
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of branches
    ng = gen.shape[0]  # number of dispatchable injections
    nxyz = len(x)  # total number of control vars of all types

    # set default constrained lines
    if il is None:
        il = np.arange(nl)  # all lines have limits by default
    nl2 = len(il)  # number of constrained lines

    # grab Pg & Qg
    Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]  # active generation in p.u.
    Qg = x[vv["i1"]["Qg"]:vv["iN"]["Qg"]]  # reactive generation in p.u.

    # put Pg & Qg back in gen
    gen[:, IDX.gen.PG] = Pg * baseMVA  # active generation in MW
    gen[:, IDX.gen.QG] = Qg * baseMVA  # reactive generation in MVAr

    # rebuild Sbus
    Sbus = makeSbus(baseMVA, bus, gen)  # net injected power in p.u.

    # ----- evaluate constraints -----
    # reconstruct V
    Va = x[vv["i1"]["Va"]:vv["iN"]["Va"]]
    Vm = x[vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    V = Vm * np.exp(1j * Va)

    # evaluate power flow equations
    mis = V * np.conj(Ybus * V) - Sbus

    # ----- evaluate constraint function values -----
    # first, the equality constraints (power flow)
    g = np.r_[mis.real,  # active power mismatch for all buses
              mis.imag]  # reactive power mismatch for all buses

    # then, the inequality constraints (branch flow limits)
    if nl2 > 0:
        flow_max = (branch[il, IDX.branch.RATE_A] / baseMVA)**2
        flow_max[flow_max == 0] = np.Inf
        if ppopt['OPF_FLOW_LIM'] == 2:  # current magnitude limit, |I|
            If = Yf * V
            It = Yt * V
            h = np.r_[If * np.conj(If) - flow_max,  # branch I limits (from bus)
                      It * np.conj(It) - flow_max].real  # branch I limits (to bus)
        else:
            # compute branch power flows
            # complex power injected at "from" bus (p.u.)
            Sf = V[branch[il, IDX.branch.F_BUS].astype(int)] * np.conj(Yf * V)
            # complex power injected at "to" bus (p.u.)
            St = V[branch[il, IDX.branch.F_BUS].astype(int)] * np.conj(Yt * V)
            if ppopt['OPF_FLOW_LIM'] == 1:  # active power limit, P (Pan Wei)
                h = np.r_[Sf.real**2 - flow_max,  # branch P limits (from bus)
                          St.real**2 - flow_max]  # branch P limits (to bus)
            else:  # apparent power limit, |S|
                h = np.r_[Sf * np.conj(Sf) - flow_max,  # branch S limits (from bus)
                          St * np.conj(St) - flow_max].real  # branch S limits (to bus)
    else:
        h = np.zeros((0, 1))

    # ----- evaluate partials of constraints -----
    # index ranges
    iVa = np.arange(vv["i1"]["Va"], vv["iN"]["Va"])
    iVm = np.arange(vv["i1"]["Vm"], vv["iN"]["Vm"])
    iPg = np.arange(vv["i1"]["Pg"], vv["iN"]["Pg"])
    iQg = np.arange(vv["i1"]["Qg"], vv["iN"]["Qg"])
    iVaVmPgQg = np.r_[iVa, iVm, iPg, iQg].T

    # compute partials of injected bus powers
    dSbus_dVm, dSbus_dVa = dSbus_dV(Ybus, V)  # w.r.t. V
    # Pbus w.r.t. Pg, Qbus w.r.t. Qg
    neg_Cg = c_sparse((-np.ones(ng), (gen[:, IDX.gen.GEN_BUS], range(ng))), (nb, ng))

    # construct Jacobian of equality constraints (power flow) and transpose it
    dg = l_sparse((2 * nb, nxyz))
    blank = c_sparse((nb, ng))
    dg[:, iVaVmPgQg] = sp.vstack([
        # P mismatch w.r.t Va, Vm, Pg, Qg
        sp.hstack([dSbus_dVa.real, dSbus_dVm.real, neg_Cg, blank]),
        # Q mismatch w.r.t Va, Vm, Pg, Qg
        sp.hstack([dSbus_dVa.imag, dSbus_dVm.imag, blank, neg_Cg])
    ], "csr")
    dg = dg.T

    if nl2 > 0:
        # compute partials of Flows w.r.t. V
        if ppopt['OPF_FLOW_LIM'] == 2:  # current
            dFf_dVa, dFf_dVm, dFt_dVa, dFt_dVm, Ff, Ft = \
                dIbr_dV(branch[il, :], Yf, Yt, V)
        else:  # power
            dFf_dVa, dFf_dVm, dFt_dVa, dFt_dVm, Ff, Ft = \
                dSbr_dV(branch[il, :], Yf, Yt, V)
        if ppopt['OPF_FLOW_LIM'] == 1:  # real part of flow (active power)
            dFf_dVa = dFf_dVa.real
            dFf_dVm = dFf_dVm.real
            dFt_dVa = dFt_dVa.real
            dFt_dVm = dFt_dVm.real
            Ff = Ff.real
            Ft = Ft.real

        # squared magnitude of flow (of complex power or current, or real power)
        df_dVa, df_dVm, dt_dVa, dt_dVm = \
            dAbr_dV(dFf_dVa, dFf_dVm, dFt_dVa, dFt_dVm, Ff, Ft)

        # construct Jacobian of inequality constraints (branch limits)
        # and transpose it.
        dh = l_sparse((2 * nl2, nxyz))
        dh[:, np.r_[iVa, iVm].T] = sp.vstack([
            sp.hstack([df_dVa, df_dVm]),  # "from" flow limit
            sp.hstack([dt_dVa, dt_dVm])  # "to" flow limit
        ], "csr")
        dh = dh.T
    else:
        dh = None

    return h, g, dh, dg


def opf_costfcn(x, om, return_hessian=False):
    """
    Evaluates objective function, gradient and Hessian for OPF.

    Objective function evaluation routine for AC optimal power flow,
    suitable for use with L{pips}. Computes objective function value,
    gradient and Hessian.

    @param x: optimization vector
    @param om: OPF model object

    @return: C{F} - value of objective function. C{df} - (optional) gradient
    of objective function (column vector). C{d2f} - (optional) Hessian of
    objective function (sparse matrix).

    @see: L{opf_consfcn}, L{opf_hessfcn}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    # ----- initialize -----
    # unpack data
    ppc = om.get_ppc()
    baseMVA, gen, gencost = ppc["baseMVA"], ppc["gen"], ppc["gencost"]
    cp = om.get_cost_params()
    N, Cw, H, dd, rh, kk, mm = \
        cp["N"], cp["Cw"], cp["H"], cp["dd"], cp["rh"], cp["kk"], cp["mm"]
    vv, _, _, _ = om.get_idx()

    # problem dimensions
    ng = gen.shape[0]  # number of dispatchable injections
    ny = om.getN('var', 'y')  # number of piece-wise linear costs
    nxyz = len(x)  # total number of control vars of all types

    # grab Pg & Qg
    Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]  # active generation in p.u.
    Qg = x[vv["i1"]["Qg"]:vv["iN"]["Qg"]]  # reactive generation in p.u.

    # ----- evaluate objective function -----
    # polynomial cost of P and Q
    # use totcost only on polynomial cost in the minimization problem
    # formulation, pwl cost is the sum of the y variables.
    ipol = find(gencost[:, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL)  # poly MW and MVAr costs
    xx = np.r_[Pg, Qg] * baseMVA
    if any(ipol):
        f = sum(totcost(gencost[ipol, :], xx[ipol]))  # cost of poly P or Q
    else:
        f = 0

    # piecewise linear cost of P and Q
    if ny > 0:
        ccost = c_sparse((np.ones(ny),
                          (np.zeros(ny), np.arange(vv["i1"]["y"], vv["iN"]["y"]))),
                         (1, nxyz)).toarray().flatten()
        f = f + np.dot(ccost, x)
    else:
        ccost = np.zeros(nxyz)

    # generalized cost term
    if sp.issparse(N) and N.nnz > 0:
        nw = N.shape[0]
        r = N * x - rh  # Nx - rhat
        iLT = find(r < -kk)  # below dead zone
        iEQ = find((r == 0) & (kk == 0))  # dead zone doesn't exist
        iGT = find(r > kk)  # above dead zone
        iND = np.r_[iLT, iEQ, iGT]  # rows that are Not in the Dead region
        iL = find(dd == 1)  # rows using linear function
        iQ = find(dd == 2)  # rows using quadratic function
        LL = c_sparse((np.ones(len(iL)), (iL, iL)), (nw, nw))
        QQ = c_sparse((np.ones(len(iQ)), (iQ, iQ)), (nw, nw))
        kbar = c_sparse((np.r_[np.ones(len(iLT)), np.zeros(len(iEQ)), -np.ones(len(iGT))],
                         (iND, iND)), (nw, nw)) * kk
        rr = r + kbar  # apply non-dead zone shift
        M = c_sparse((mm[iND], (iND, iND)), (nw, nw))  # dead zone or scale
        diagrr = c_sparse((rr, (np.arange(nw), np.arange(nw))), (nw, nw))

        # linear rows multiplied by rr(i), quadratic rows by rr(i)^2
        w = M * (LL + QQ * diagrr) * rr

        f = f + np.dot(w * H, w) / 2 + np.dot(Cw, w)

    # ----- evaluate cost gradient -----
    # index ranges
    iPg = range(vv["i1"]["Pg"], vv["iN"]["Pg"])
    iQg = range(vv["i1"]["Qg"], vv["iN"]["Qg"])

    # polynomial cost of P and Q
    df_dPgQg = np.zeros(2 * ng)  # w.r.t p.u. Pg and Qg
    df_dPgQg[ipol] = baseMVA * polycost(gencost[ipol, :], xx[ipol], 1)
    df = np.zeros(nxyz)
    df[iPg] = df_dPgQg[:ng]
    df[iQg] = df_dPgQg[ng:ng + ng]

    # piecewise linear cost of P and Q
    df = df + ccost  # The linear cost row is additive wrt any nonlinear cost.

    # generalized cost term
    if sp.issparse(N) and N.nnz > 0:
        HwC = H * w + Cw
        AA = N.T * M * (LL + 2 * QQ * diagrr)
        df = df + AA * HwC

        # numerical check
        if 0:  # 1 to check, 0 to skip check
            ddff = np.zeros(df.shape)
            step = 1e-7
            tol = 1e-3
            for k in range(len(x)):
                xx = x
                xx[k] = xx[k] + step
                ddff[k] = (opf_costfcn(xx, om) - f) / step
            if max(abs(ddff - df)) > tol:
                idx = find(abs(ddff - df) == max(abs(ddff - df)))
                print('Mismatch in gradient')
                print('idx             df(num)         df              diff')
                print('%4d%16g%16g%16g' %
                      (range(len(df)), ddff.T, df.T, abs(ddff - df).T))
                print('MAX')
                print('%4d%16g%16g%16g' %
                      (idx.T, ddff[idx].T, df[idx].T,
                       abs(ddff[idx] - df[idx]).T))

    if not return_hessian:
        return f, df

    # ---- evaluate cost Hessian -----
    pcost = gencost[range(ng), :]
    if gencost.shape[0] > ng:
        qcost = gencost[ng + 1:2 * ng, :]
    else:
        qcost = np.array([])

    # polynomial generator costs
    d2f_dPg2 = np.zeros(ng)  # w.r.t. p.u. Pg
    d2f_dQg2 = np.zeros(ng)  # w.r.t. p.u. Qg
    ipolp = find(pcost[:, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL)
    d2f_dPg2[ipolp] = \
        baseMVA**2 * polycost(pcost[ipolp, :], Pg[ipolp]*baseMVA, 2)
    if any(qcost):  # Qg is not free
        ipolq = find(qcost[:, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL)
        d2f_dQg2[ipolq] = \
            baseMVA**2 * polycost(qcost[ipolq, :], Qg[ipolq] * baseMVA, 2)
    i = np.r_[iPg, iQg].T
    d2f = c_sparse((np.r_[d2f_dPg2, d2f_dQg2], (i, i)), (nxyz, nxyz))

    # generalized cost
    if N is not None and sp.issparse(N):
        d2f = d2f + AA * H * AA.T + 2 * N.T * M * QQ * \
            c_sparse((HwC, (range(nw), range(nw))), (nw, nw)) * N

    return f, df, d2f


def run_userfcn(userfcn, stage, *args2):
    """
    Runs the userfcn callbacks for a given stage.

    Example::
        ppc = om.get_mpc()
        om = run_userfcn(ppc['userfcn'], 'formulation', om)

    @param userfcn: the 'userfcn' field of ppc, populated by L{add_userfcn}
    @param stage: the name of the callback stage begin executed
    (additional arguments) some stages require additional arguments.

    @see: L{add_userfcn}, L{remove_userfcn}, L{toggle_reserves},
          L{toggle_iflims}, L{runopf_w_res}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    rv = args2[0]
    if (len(userfcn) > 0) and (stage in userfcn):
        for k in range(len(userfcn[stage])):
            if 'args' in userfcn[stage][k]:
                args = userfcn[stage][k]['args']
            else:
                args = []

            if stage in ['ext2int', 'formulation', 'int2ext']:
                # ppc     = userfcn_*_ext2int(ppc, args)
                # om      = userfcn_*_formulation(om, args)
                # results = userfcn_*_int2ext(results, args)
                rv = userfcn[stage][k]['fcn'](rv, args)
            elif stage in ['printpf', 'savecase']:
                # results = userfcn_*_printpf(results, fd, ppopt, args)
                # ppc     = userfcn_*_savecase(mpc, fd, prefix, args)
                fdprint = args2[1]
                ppoptprint = args2[2]
                rv = userfcn[stage][k]['fcn'](rv, fdprint, ppoptprint, args)

    return rv


def add_userfcn(ppc, stage, fcn, args=None, allow_multiple=False):
    """
    Appends a userfcn to the list to be called for a case.

    A userfcn is a callback function that can be called automatically by
    PYPOWER at one of various stages in a simulation.

    Currently there are 5 different callback stages defined. Each stage has
    a name, and by convention, the name of a user-defined callback function
    ends with the name of the stage. The following is a description of each
    stage, when it is called and the input and output arguments which vary
    depending on the stage. The reserves example (see L{runopf_w_res}) is used
    to illustrate how these callback userfcns might be used.

      1. C{'ext2int'}

      Called from L{ext2int} immediately after the case is converted from
      external to internal indexing. Inputs are a PYPOWER case dict (C{ppc}),
      freshly converted to internal indexing and any (optional) C{args} value
      supplied via L{add_userfcn}. Output is the (presumably updated) C{ppc}.
      This is typically used to reorder any input arguments that may be needed
      in internal ordering by the formulation stage.

      E.g. C{ppc = userfcn_reserves_ext2int(ppc, args)}

      2. C{'formulation'}

      Called from L{opf} after the OPF Model (C{om}) object has been
      initialized with the standard OPF formulation, but before calling the
      solver. Inputs are the C{om} object and any (optional) C{args} supplied
      via L{add_userfcn}. Output is the C{om} object. This is the ideal place
      to add any additional vars, constraints or costs to the OPF formulation.

      E.g. C{om = userfcn_reserves_formulation(om, args)}

      3. C{'int2ext'}

      Called from L{int2ext} immediately before the resulting case is converted
      from internal back to external indexing. Inputs are the C{results} dict
      and any (optional) C{args} supplied via C{add_userfcn}. Output is the
      C{results} dict. This is typically used to convert any results to
      external indexing and populate any corresponding fields in the
      C{results} dict.

      E.g. C{results = userfcn_reserves_int2ext(results, args)}

      4. C{'printpf'}

      Called from L{printpf} after the pretty-printing of the standard OPF
      output. Inputs are the C{results} dict, the file descriptor to write to,
      a PYPOWER options dict, and any (optional) C{args} supplied via
      L{add_userfcn}. Output is the C{results} dict. This is typically used for
      any additional pretty-printing of results.

      E.g. C{results = userfcn_reserves_printpf(results, fd, ppopt, args)}

      5. C{'savecase'}

      Called from L{savecase} when saving a case dict to a Python file after
      printing all of the other data to the file. Inputs are the case dict,
      the file descriptor to write to, the variable prefix (typically 'ppc')
      and any (optional) C{args} supplied via L{add_userfcn}. Output is the
      case dict. This is typically used to write any non-standard case dict
      fields to the case file.

      E.g. C{ppc = userfcn_reserves_printpf(ppc, fd, prefix, args)}

    @param ppc: the case dict
    @param stage: the name of the stage at which this function should be
        called: ext2int, formulation, int2ext, printpf
    @param fcn: the name of the userfcn
    @param args: (optional) the value to be passed as an argument to the
        userfcn
    @param allow_multiple: (optional) if True, allows the same function to
        be added more than once.

    @see: L{run_userfcn}, L{remove_userfcn}, L{toggle_reserves},
          L{toggle_iflims}, L{runopf_w_res}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if args is None:
        args = []

    if stage not in ['ext2int', 'formulation', 'int2ext', 'printpf', 'savecase']:
        logger.debug('add_userfcn : \'%s\' is not the name of a valid callback stage\n' % stage)

    n = 0
    if 'userfcn' in ppc:
        if stage in ppc['userfcn']:
            n = len(ppc['userfcn'][stage])  # + 1
            if not allow_multiple:
                for k in range(n):
                    if ppc['userfcn'][stage][k]['fcn'] == fcn:
                        logger.debug('add_userfcn: the function \'%s\' has already been added\n' % fcn.__name__)
        else:
            ppc['userfcn'][stage] = []
    else:
        ppc['userfcn'] = {stage: []}

    ppc['userfcn'][stage].append({'fcn': fcn})
    if len(args) > 0:
        ppc['userfcn'][stage][n]['args'] = args

    return ppc


def remove_userfcn(ppc, stage, fcn):
    """
    Removes a userfcn from the list to be called for a case.

    A userfcn is a callback function that can be called automatically by
    PYPOWER at one of various stages in a simulation. This function removes
    the last instance of the userfcn for the given C{stage} with the function
    handle specified by C{fcn}.

    @see: L{add_userfcn}, L{run_userfcn}, L{toggle_reserves},
          L{toggle_iflims}, L{runopf_w_res}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    n = len(ppc['userfcn'][stage])

    for k in range(n - 1, -1, -1):
        if ppc['userfcn'][stage][k]['fcn'] == fcn:
            del ppc['userfcn'][stage][k]
            break

    return ppc


def totcost(gencost, Pg):
    """
    Computes total cost for generators at given output level.

    Computes total cost for generators given a matrix in gencost format and
    a column vector or matrix of generation levels. The return value has the
    same dimensions as PG. Each row of C{gencost} is used to evaluate the
    cost at the points specified in the corresponding row of C{Pg}.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    ng, m = gencost.shape
    totalcost = np.zeros(ng)

    if len(gencost) > 0:
        ipwl = find(gencost[:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR)
        ipol = find(gencost[:, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL)
        if len(ipwl) > 0:
            p = gencost[:, IDX.cost.COST:(m-1):2]
            c = gencost[:, (IDX.cost.COST+1):m:2]

            for i in ipwl:
                ncost = gencost[i, IDX.cost.NCOST]
                for k in np.arange(ncost - 1, dtype=int):
                    p1, p2 = p[i, k], p[i, k+1]
                    c1, c2 = c[i, k], c[i, k+1]
                    m = (c2 - c1) / (p2 - p1)
                    b = c1 - m * p1
                    Pgen = Pg[i]
                    if Pgen < p2:
                        totalcost[i] = m * Pgen + b
                        break
                    totalcost[i] = m * Pgen + b

        if len(ipol) > 0:
            totalcost[ipol] = polycost(gencost[ipol, :], Pg[ipol])

    return totalcost


def modcost(gencost, alpha, modtype='SCALE_F'):
    """Modifies generator costs by shifting or scaling (F or X).

    For each generator cost F(X) (for real or reactive power) in
    C{gencost}, this function modifies the cost by scaling or shifting
    the function by C{alpha}, depending on the value of C{modtype}, and
    and returns the modified C{gencost}. Rows of C{gencost} can be a mix
    of polynomial or piecewise linear costs.

    C{modtype} takes one of the 4 possible values (let F_alpha(X) denote the
    the modified function)::
        SCALE_F (default) : F_alpha(X)         == F(X) * ALPHA
        SCALE_X           : F_alpha(X * ALPHA) == F(X)
        SHIFT_F           : F_alpha(X)         == F(X) + ALPHA
        SHIFT_X           : F_alpha(X + ALPHA) == F(X)

    @author: Ray Zimmerman (PSERC Cornell)
    """
    gencost = gencost.copy()

    ng, m = gencost.shape
    if ng != 0:
        ipwl = find(gencost[:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR)
        ipol = find(gencost[:, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL)
        c = gencost[ipol, IDX.cost.COST:m]

        if modtype == 'SCALE_F':
            gencost[ipol, IDX.cost.COST:m] = alpha * c
            gencost[ipwl, IDX.cost.COST+1:m:2] = alpha * gencost[ipwl, IDX.cost.COST + 1:m:2]
        elif modtype == 'SCALE_X':
            for k in range(len(ipol)):
                n = gencost[ipol[k], IDX.cost.NCOST].astype(int)
                for i in range(n):
                    gencost[ipol[k], IDX.cost.COST + i] = c[k, i] / alpha**(n - i - 1)
            gencost[ipwl, IDX.cost.COST:m - 1:2] = alpha * gencost[ipwl, IDX.cost.COST:m - 1:2]
        elif modtype == 'SHIFT_F':
            for k in range(len(ipol)):
                n = gencost[ipol[k], IDX.cost.NCOST].astype(int)
                gencost[ipol[k], IDX.cost.COST + n - 1] = alpha + c[k, n - 1]
            gencost[ipwl, IDX.cost.COST+1:m:2] = alpha + gencost[ipwl, IDX.cost.COST + 1:m:2]
        elif modtype == 'SHIFT_X':
            for k in range(len(ipol)):
                n = gencost[ipol[k], IDX.cost.NCOST].astype(int)
                gencost[ipol[k], IDX.cost.COST:IDX.cost.COST + n] = \
                    polyshift(c[k, :n].T, alpha).T
            gencost[ipwl, IDX.cost.COST:m - 1:2] = alpha + gencost[ipwl, IDX.cost.COST:m - 1:2]
        else:
            logger.debug('modcost: "%s" is not a valid modtype\n' % modtype)

    return gencost


def polyshift(c, a):
    """
    Returns the coefficients of a horizontally shifted polynomial.

    C{d = polyshift(c, a)} shifts to the right by C{a}, the polynomial whose
    coefficients are given in the column vector C{c}.

    Example: For any polynomial with C{n} coefficients in C{c}, and any values
    for C{x} and shift C{a}, the C{f - f0} should be zero::
        x = rand
        a = rand
        c = rand(n, 1);
        f0 = polyval(c, x)
        f  = polyval(polyshift(c, a), x+a)
    """
    n = len(c)
    d = np.zeros(c.shape)
    A = pow(-a * np.ones(n), np.arange(n))
    b = np.ones(n)
    for k in range(n):
        d[n - k - 1] = np.dot(b, c[n - k - 1::-1] * A[:n - k])
        b = np.cumsum(b[:n - k - 1])

    return d


def polycost(gencost, Pg, der=0):
    """
    Evaluates polynomial generator cost & derivatives.

    C{f = polycost(gencost, Pg)} returns the vector of costs evaluated at C{Pg}

    C{df = polycost(gencost, Pg, 1)} returns the vector of first derivatives
    of costs evaluated at C{Pg}

    C{d2f = polycost(gencost, Pg, 2)} returns the vector of second derivatives
    of costs evaluated at C{Pg}

    C{gencost} must contain only polynomial costs
    C{Pg} is in MW, not p.u. (works for C{Qg} too)

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if gencost.size == 0:
        # User has a purely linear piecewise problem, exit early with empty array
        return []

    if any(gencost[:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR):
        logger.debug('polycost: all costs must be polynomial\n')

    ng = len(Pg)
    maxN = max(gencost[:, IDX.cost.NCOST].astype(int))
    minN = min(gencost[:, IDX.cost.NCOST].astype(int))

    # form coefficient matrix where 1st column is constant term, 2nd linear, etc.
    c = np.zeros((ng, maxN))
    for n in np.arange(minN, maxN + 1):
        k = find(gencost[:, IDX.cost.NCOST] == n)  # cost with n coefficients
        c[k, :n] = gencost[k, (IDX.cost.COST + n - 1):IDX.cost.COST - 1:-1]

    # do derivatives
    for d in range(1, der + 1):
        if c.shape[1] >= 2:
            c = c[:, 1:maxN - d + 1]
        else:
            c = np.zeros((ng, 1))
            break

        for k in range(2, maxN - d + 1):
            c[:, k-1] = c[:, k-1] * k

    # evaluate polynomial
    if len(c) == 0:
        f = np.zeros(Pg.shape)
    else:
        f = c[:, :1].flatten()  # constant term
        for k in range(1, c.shape[1]):
            f = f + c[:, k] * Pg**k

    return f


def pqcost(gencost, ng, on=None):
    """
    Splits the gencost variable into two pieces if costs are given for Qg.

    Checks whether C{gencost} has cost information for reactive power
    generation (rows C{ng+1} to C{2*ng}). If so, it returns the first C{ng}
    rows in C{pcost} and the last C{ng} rows in C{qcost}. Otherwise, leaves
    C{qcost} empty. Also does some error checking.
    If C{on} is specified (list of indices of generators which are on line)
    it only returns the rows corresponding to these generators.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if on is None:
        on = np.arange(ng)

    if gencost.shape[0] == ng:
        pcost = gencost[on, :]
        qcost = np.array([])
    elif gencost.shape[0] == 2 * ng:
        pcost = gencost[on, :]
        qcost = gencost[on + ng, :]
    else:
        logger.info('pqcost: gencost has wrong number of rows')

    return pcost, qcost


def poly2pwl(polycost, Pmin, Pmax, npts):
    """
    Converts polynomial cost variable to piecewise linear.

    Converts the polynomial cost variable C{polycost} into a piece-wise linear
    cost by evaluating at zero and then at C{npts} evenly spaced points between
    C{Pmin} and C{Pmax}. If C{Pmin <= 0} (such as for reactive power, where
    C{P} really means C{Q}) it just uses C{npts} evenly spaced points between
    C{Pmin} and C{Pmax}.
    """
    pwlcost = polycost
    # size of piece being changed
    m, n = polycost.shape
    # change cost model
    pwlcost[:, IDX.cost.MODEL] = IDX.cost.PW_LINEAR * np.ones(m)
    # zero out old data
    pwlcost[:, IDX.cost.COST:IDX.cost.COST + n] = np.zeros(pwlcost[:, IDX.cost.COST:IDX.cost.COST + n].shape)
    # change number of data points
    pwlcost[:, IDX.cost.NCOST] = npts * IDX.cost.ones(m)

    for i in range(m):
        if Pmin[i] == 0:
            step = (Pmax[i] - Pmin[i]) / (npts - 1)
            xx = range(Pmin[i], step, Pmax[i])
        elif Pmin[i] > 0:
            step = (Pmax[i] - Pmin[i]) / (npts - 2)
            xx = r_[0, range(Pmin[i], step, Pmax[i])]
        elif Pmin[i] < 0 & Pmax[i] > 0:  # for when P really means Q
            step = (Pmax[i] - Pmin[i]) / (npts - 1)
            xx = range(Pmin[i], step, Pmax[i])
        yy = totcost(polycost[i, :], xx)
        pwlcost[i,      IDX.cost.COST:2:(IDX.cost.COST + 2*(npts-1))] = xx
        pwlcost[i,  (IDX.cost.COST+1):2:(IDX.cost.COST + 2*(npts-1) + 1)] = yy

    return pwlcost


def scale_load(load, bus, gen=None, load_zone=None, opt=None):
    """
    Scales fixed and/or dispatchable loads.

    Assumes consecutive bus numbering when dealing with dispatchable loads.

    @param load: Each element specifies the amount of scaling for the
        corresponding load zone, either as a direct scale factor
        or as a target quantity. If there are C{nz} load zones this
        vector has C{nz} elements.
    @param bus: Standard C{bus} matrix with C{nb} rows, where the fixed active
        and reactive loads available for scaling are specified in
        columns C{PD} and C{QD}
    @param gen: (optional) standard C{gen} matrix with C{ng} rows, where the
        dispatchable loads available for scaling are specified by
        columns C{PG}, C{QG}, C{PMIN}, C{QMIN} and C{QMAX} (in rows for which
        C{isload(gen)} returns C{true}). If C{gen} is empty, it assumes
        there are no dispatchable loads.
    @param load_zone: (optional) C{nb} element vector where the value of
        each element is either zero or the index of the load zone
        to which the corresponding bus belongs. If C{load_zone[b] = k}
        then the loads at bus C{b} will be scaled according to the
        value of C{load[k]}. If C{load_zone[b] = 0}, the loads at bus C{b}
        will not be modified. If C{load_zone} is empty, the default is
        determined by the dimensions of the C{load} vector. If C{load} is
        a scalar, a single system-wide zone including all buses is
        used, i.e. C{load_zone = ones(nb)}. If C{load} is a vector, the
        default C{load_zone} is defined as the areas specified in the
        C{bus} matrix, i.e. C{load_zone = bus[:, BUS_AREA]}, and C{load}
        should have dimension C{= max(bus[:, BUS_AREA])}.
    @param opt: (optional) dict with three possible fields, 'scale',
        'pq' and 'which' that determine the behavior as follows:
            - C{scale} (default is 'FACTOR')
                - 'FACTOR'   : C{load} consists of direct scale factors, where
                C{load[k] =} scale factor C{R[k]} for zone C{k}
                - 'QUANTITY' : C{load} consists of target quantities, where
                C{load[k] =} desired total active load in MW for
                zone C{k} after scaling by an appropriate C{R(k)}
            - C{pq}    (default is 'PQ')
                - 'PQ' : scale both active and reactive loads
                - 'P'  : scale only active loads
            - C{which} (default is 'BOTH' if GEN is provided, else 'FIXED')
                - 'FIXED'        : scale only fixed loads
                - 'DISPATCHABLE' : scale only dispatchable loads
                - 'BOTH'         : scale both fixed and dispatchable loads

    @see: L{total_load}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    nb = bus.shape[0]  # number of buses

    # -----  process inputs  -----
    bus = bus.copy()
    if gen is None:
        gen = np.array([])
    else:
        gen = gen.copy()
    if load_zone is None:
        load_zone = np.array([], int)
    if opt is None:
        opt = {}

    # fill out and check opt
    if len(gen) == 0:
        opt["which"] = 'FIXED'
    if 'pq' not in opt:
        opt["pq"] = 'PQ'  # 'PQ' or 'P'
    if 'which' not in opt:
        opt["which"] = 'BOTH'  # 'FIXED', 'DISPATCHABLE' or 'BOTH'
    if 'scale' not in opt:
        opt["scale"] = 'FACTOR'  # 'FACTOR' or 'QUANTITY'
    if (opt["pq"] != 'P') and (opt["pq"] != 'PQ'):
        logger.debug("scale_load: opt['pq'] must equal 'PQ' or 'P'\n")
    if (opt["which"][0] != 'F') and (opt["which"][0] != 'D') and (opt["which"][0] != 'B'):
        logger.debug("scale_load: opt.which should be 'FIXED, 'DISPATCHABLE or 'BOTH'\n")
    if (opt["scale"][0] != 'F') and (opt["scale"][0] != 'Q'):
        logger.debug("scale_load: opt.scale should be 'FACTOR or 'QUANTITY'\n")
    if (len(gen) == 0) and (opt["which"][0] != 'F'):
        logger.debug('scale_load: need gen matrix to scale dispatchable loads\n')

    # create dispatchable load connection matrix
    if len(gen) > 0:
        ng = gen.shape[0]
        is_ld = isload(gen) & (gen[:, IDX.gen.GEN_STATUS] > 0)
        ld = find(is_ld)

        # create map of external bus numbers to bus indices
        i2e = bus[:, IDX.bus.BUS_I].astype(int)
        e2i = np.zeros(max(i2e) + 1, int)
        e2i[i2e] = np.arange(nb)

        gbus = gen[:, IDX.gen.GEN_BUS].astype(int)
        Cld = c_sparse((is_ld, (e2i[gbus], np.arange(ng))), (nb, ng))
    else:
        ng = 0
        ld = np.array([], int)

    if len(load_zone) == 0:
        if len(load) == 1:  # make a single zone of all load buses
            load_zone = np.zeros(nb, int)  # initialize
            load_zone[bus[:, IDX.bus.PD] != 0 or bus[:, IDX.bus.QD] != 0] = 1  # FIXED loads
            if len(gen) > 0:
                gbus = gen[ld, IDX.gen.GEN_BUS].astype(int)
                load_zone[e2i[gbus]] = 1  # DISPATCHABLE loads
        else:  # use areas defined in bus data as zones
            load_zone = bus[:, IDX.bus.BUS_AREA]

    # check load_zone to make sure it's consistent with size of load vector
    if max(load_zone) > len(load):
        logger.debug('scale_load: load vector must have a value for each load zone specified\n')

    # -----  compute scale factors for each zone  -----
    scale = load.copy()
    Pdd = np.zeros(nb)  # dispatchable P at each bus
    if opt["scale"][0] == 'Q':  # 'QUANTITY'
        # find load capacity from dispatchable loads
        if len(gen) > 0:
            Pdd = -Cld * gen[:, IDX.gen.PMIN]

        # compute scale factors
        for k in range(len(load)):
            idx = find(load_zone == k + 1)
            fixed = sum(bus[idx, IDX.bus.PD])
            dispatchable = sum(Pdd[idx])
            total = fixed + dispatchable
            if opt["which"][0] == 'B':  # 'BOTH'
                if total != 0:
                    scale[k] = load[k] / total
                elif load[k] == total:
                    scale[k] = 1
                else:
                    raise ValueError(
                        'scale_load: impossible to make zone %d load equal %g by scaling non-existent loads' %
                        (k, load[k]))
            elif opt["which"][0] == 'F':  # 'FIXED'
                if fixed != 0:
                    scale[k] = (load[k] - dispatchable) / fixed
                elif load[k] == dispatchable:
                    scale[k] = 1
                else:
                    raise ValueError(
                        'scale_load: impossible to make zone %d load equal %g by scaling non-existent fixed load' %
                        (k, load[k]))
            elif opt["which"][0] == 'D':  # 'DISPATCHABLE'
                if dispatchable != 0:
                    scale[k] = (load[k] - fixed) / dispatchable
                elif load[k] == fixed:
                    scale[k] = 1
                else:
                    raise ValueError(
                        'scale_load: impossible to make zone %d load equal %g by scaling non-existent dispatchable load' % (k, load[k]))

    # -----  do the scaling  -----
    # fixed loads
    if opt["which"][0] != 'D':  # includes 'FIXED', not 'DISPATCHABLE' only
        for k in range(len(scale)):
            idx = find(load_zone == k + 1)
            bus[idx, IDX.bus.PD] = bus[idx, IDX.bus.PD] * scale[k]
            if opt["pq"] == 'PQ':
                bus[idx, IDX.bus.QD] = bus[idx, IDX.bus.QD] * scale[k]

    # dispatchable loads
    if opt["which"][0] != 'F':  # includes 'DISPATCHABLE', not 'FIXED' only
        for k in range(len(scale)):
            idx = find(load_zone == k + 1)
            gbus = gen[ld, IDX.gen.GEN_BUS].astype(int)
            i = find(np.in1d(e2i[gbus], idx))
            ig = ld[i]

            gen[np.ix_(ig, [IDX.gen.PG, IDX.gen.PMIN])] = gen[np.ix_(ig, [IDX.gen.PG, IDX.gen.PMIN])] * scale[k]
            if opt["pq"] == 'PQ':
                gen[np.ix_(ig, [IDX.gen.QG, IDX.gen.QMIN, IDX.gen.QMAX])] = gen[np.ix_(
                    ig, [IDX.gen.QG, IDX.gen.QMIN, IDX.gen.QMAX])] * scale[k]

    return bus, gen


def update_mupq(baseMVA, gen, mu_PQh, mu_PQl, data):
    """
    Updates values of generator limit shadow prices.

    Updates the values of C{MU_PMIN}, C{MU_PMAX}, C{MU_QMIN}, C{MU_QMAX} based
    on any shadow prices on the sloped portions of the generator
    capability curve constraints.

    @param mu_PQh: shadow prices on upper sloped portion of capability curves
    @param mu_PQl: shadow prices on lower sloped portion of capability curves
    @param data: "data" dict returned by L{makeApq}

    @see: C{makeApq}.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    # extract the constraint parameters
    ipqh, ipql, Apqhdata, Apqldata = \
        data['ipqh'], data['ipql'], data['h'], data['l']

    # combine original limit multipliers into single value
    muP = gen[:, IDX.gen.MU_PMAX] - gen[:, IDX.gen.MU_PMIN]
    muQ = gen[:, IDX.gen.MU_QMAX] - gen[:, IDX.gen.MU_QMIN]

    # add P and Q components of multipliers on upper sloped constraint
    muP[ipqh] = muP[ipqh] - mu_PQh * Apqhdata[:, 0] / baseMVA
    muQ[ipqh] = muQ[ipqh] - mu_PQh * Apqhdata[:, 1] / baseMVA

    # add P and Q components of multipliers on lower sloped constraint
    muP[ipql] = muP[ipql] - mu_PQl * Apqldata[:, 0] / baseMVA
    muQ[ipql] = muQ[ipql] - mu_PQl * Apqldata[:, 1] / baseMVA

    # split back into upper and lower multipliers based on sign
    gen[:, IDX.gen.MU_PMAX] = (muP > 0) * muP
    gen[:, IDX.gen.MU_PMIN] = (muP < 0) * -muP
    gen[:, IDX.gen.MU_QMAX] = (muQ > 0) * muQ
    gen[:, IDX.gen.MU_QMIN] = (muQ < 0) * -muQ

    return gen


def int2ext(ppc, val_or_field=None, oldval=None, ordering=None, dim=0):
    """
    Converts internal to external bus numbering.

    C{ppc = int2ext(ppc)}

    If the input is a single PYPOWER case dict, then it restores all
    buses, generators and branches that were removed because of being
    isolated or off-line, and reverts to the original generator ordering
    and original bus numbering. This requires that the 'order' key
    created by L{ext2int} be in place.

    Example::
        ppc = int2ext(ppc)

    @see: L{ext2int}, L{i2e_field}, L{i2e_data}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ppc = deepcopy(ppc)
    if val_or_field is None:  # nargin == 1
        if 'order' not in ppc:
            logger.debug('int2ext: ppc does not have the "order" field '
                         'required for conversion back to external numbering.\n')
        o = ppc["order"]

        if o["state"] == 'i':
            # execute userfcn callbacks for 'int2ext' stage
            if 'userfcn' in ppc:
                ppc = run_userfcn(ppc["userfcn"], 'int2ext', ppc)

            # save data matrices with internal ordering & restore originals
            o["int"] = {}
            o["int"]["bus"] = ppc["bus"].copy()
            o["int"]["branch"] = ppc["branch"].copy()
            o["int"]["gen"] = ppc["gen"].copy()
            ppc["bus"] = o["ext"]["bus"].copy()
            ppc["branch"] = o["ext"]["branch"].copy()
            ppc["gen"] = o["ext"]["gen"].copy()
            if 'gencost' in ppc:
                o["int"]["gencost"] = ppc["gencost"].copy()
                ppc["gencost"] = o["ext"]["gencost"].copy()
            if 'areas' in ppc:
                o["int"]["areas"] = ppc["areas"].copy()
                ppc["areas"] = o["ext"]["areas"].copy()
            if 'A' in ppc:
                o["int"]["A"] = ppc["A"].copy()
                ppc["A"] = o["ext"]["A"].copy()
            if 'N' in ppc:
                o["int"]["N"] = ppc["N"].copy()
                ppc["N"] = o["ext"]["N"].copy()

            # update data (in bus, branch and gen only)
            ppc["bus"][o["bus"]["status"]["on"], :] = \
                o["int"]["bus"]
            ppc["branch"][o["branch"]["status"]["on"], :] = \
                o["int"]["branch"]
            ppc["gen"][o["gen"]["status"]["on"], :] = \
                o["int"]["gen"][o["gen"]["i2e"], :]
            if 'areas' in ppc:
                ppc["areas"][o["areas"]["status"]["on"], :] = \
                    o["int"]["areas"]

            # revert to original bus numbers
            ppc["bus"][o["bus"]["status"]["on"], IDX.bus.BUS_I] = \
                o["bus"]["i2e"][ppc["bus"][o["bus"]["status"]["on"], IDX.bus.BUS_I].astype(int)]
            ppc["branch"][o["branch"]["status"]["on"], IDX.branch.F_BUS] = \
                o["bus"]["i2e"][ppc["branch"]
                                [o["branch"]["status"]["on"], IDX.branch.F_BUS].astype(int)]
            ppc["branch"][o["branch"]["status"]["on"], IDX.branch.T_BUS] = \
                o["bus"]["i2e"][ppc["branch"]
                                [o["branch"]["status"]["on"], IDX.branch.T_BUS].astype(int)]
            ppc["gen"][o["gen"]["status"]["on"], IDX.gen.GEN_BUS] = \
                o["bus"]["i2e"][ppc["gen"]
                                [o["gen"]["status"]["on"], IDX.gen.GEN_BUS].astype(int)]
            if 'areas' in ppc:
                ppc["areas"][o["areas"]["status"]["on"], IDX.area.PRICE_REF_BUS] = \
                    o["bus"]["i2e"][ppc["areas"]
                                    [o["areas"]["status"]["on"], IDX.area.PRICE_REF_BUS].astype(int)]

            if 'ext' in o:
                del o['ext']
            o["state"] = 'e'
            ppc["order"] = o
        else:
            logger.debug('int2ext: ppc claims it is already using '
                         'external numbering.\n')
    else:  # convert extra data
        if isinstance(val_or_field, str) or isinstance(val_or_field, list):
            # field (key)
            logger.warning(
                'Calls of the form MPC = INT2EXT(MPC, '
                'FIELD_NAME'
                ', ...) have been deprecated. Please replace INT2EXT with I2E_FIELD.')
            bus, gen = val_or_field, oldval
            if ordering is not None:
                dim = ordering
            ppc = i2e_field(ppc, bus, gen, dim)
        else:
            # value
            logger.warning(
                'Calls of the form VAL = INT2EXT(MPC, VAL, ...) have been deprecated. Please replace INT2EXT with I2E_DATA.')
            bus, gen, branch = val_or_field, oldval, ordering
            ppc = i2e_data(ppc, bus, gen, branch, dim)

    return ppc


def int2ext1(i2e, bus, gen, branch, areas):
    """
    Converts from the consecutive internal bus numbers back to the originals
    using the mapping provided by the I2E vector returned from C{ext2int}.

    @see: L{ext2int}
    @see: U{http://www.pserc.cornell.edu/matpower/}
    """
    bus[:, IDX.bus.BUS_I] = i2e[bus[:, IDX.bus.BUS_I].astype(int)]
    gen[:, IDX.gen.GEN_BUS] = i2e[gen[:, IDX.gen.GEN_BUS].astype(int)]
    branch[:, IDX.branch.F_BUS] = i2e[branch[:, IDX.branch.F_BUS].astype(int)]
    branch[:, IDX.branch.T_BUS] = i2e[branch[:, IDX.branch.T_BUS].astype(int)]

    if areas != None and len(areas) > 0:
        areas[:, IDX.area.PRICE_REF_BUS] = i2e[areas[:, IDX.area.PRICE_REF_BUS].astype(int)]
        return bus, gen, branch, areas

    return bus, gen, branch


def e2i_data(ppc, val, ordering, dim=0):
    """
    Converts data from external to internal indexing.

    When given a case dict that has already been converted to
    internal indexing, this function can be used to convert other data
    structures as well by passing in 2 or 3 extra parameters in
    addition to the case dict. If the value passed in the 2nd
    argument is a column vector, it will be converted according to the
    C{ordering} specified by the 3rd argument (described below). If C{val}
    is an n-dimensional matrix, then the optional 4th argument (C{dim},
    default = 0) can be used to specify which dimension to reorder.
    The return value in this case is the value passed in, converted
    to internal indexing.

    The 3rd argument, C{ordering}, is used to indicate whether the data
    corresponds to bus-, gen- or branch-ordered data. It can be one
    of the following three strings: 'bus', 'gen' or 'branch'. For
    data structures with multiple blocks of data, ordered by bus,
    gen or branch, they can be converted with a single call by
    specifying C{ordering} as a list of strings.

    Any extra elements, rows, columns, etc. beyond those indicated
    in C{ordering}, are not disturbed.

    Examples:
        A_int = e2i_data(ppc, A_ext, ['bus','bus','gen','gen'], 1)

        Converts an A matrix for user-supplied OPF constraints from
        external to internal ordering, where the columns of the A
        matrix correspond to bus voltage angles, then voltage
        magnitudes, then generator real power injections and finally
        generator reactive power injections.

        gencost_int = e2i_data(ppc, gencost_ext, ['gen','gen'], 0)

        Converts a GENCOST matrix that has both real and reactive power
        costs (in rows 1--ng and ng+1--2*ng, respectively).
    """
    if 'order' not in ppc:
        logger.debug('e2i_data: ppc does not have the \'order\' field '
                     'required to convert from external to internal numbering.\n')
        return

    o = ppc['order']
    if o['state'] != 'i':
        logger.debug('e2i_data: ppc does not have internal ordering '
                     'data available, call ext2int first\n')
        return

    if isinstance(ordering, str):  # single set
        if ordering == 'gen':
            idx = o[ordering]["status"]["on"][o[ordering]["e2i"]]
        else:
            idx = o[ordering]["status"]["on"]
        val = get_reorder(val, idx, dim)
    else:  # multiple: sets
        b = 0  # base
        new_v = []
        for ordr in ordering:
            n = o["ext"][ordr].shape[0]
            v = get_reorder(val, b + np.arange(n), dim)
            new_v.append(e2i_data(ppc, v, ordr, dim))
            b = b + n
        n = val.shape[dim]
        if n > b:  # the rest
            v = get_reorder(val, np.arange(b, n), dim)
            new_v.append(v)

        if sp.issparse(new_v[0]):
            if dim == 0:
                sp.vstack(new_v, 'csr')
            elif dim == 1:
                sp.hstack(new_v, 'csr')
            else:
                raise ValueError('dim (%d) may be 0 or 1' % dim)
        else:
            val = np.concatenate(new_v, dim)
    return val


def e2i_field(ppc, field, ordering, dim=0):
    """
    Converts fields of C{ppc} from external to internal indexing.

    This function performs several different tasks, depending on the
    arguments passed.

    When given a case dict that has already been converted to
    internal indexing, this function can be used to convert other data
    structures as well by passing in 2 or 3 extra parameters in
    addition to the case dict.

    The 2nd argument is a string or list of strings, specifying
    a field in the case dict whose value should be converted by
    a corresponding call to L{e2i_data}. In this case, the converted value
    is stored back in the specified field, the original value is
    saved for later use and the updated case dict is returned.
    If C{field} is a list of strings, they specify nested fields.

    The 3rd and optional 4th arguments are simply passed along to
    the call to L{e2i_data}.

    Examples:
        ppc = e2i_field(ppc, ['reserves', 'cost'], 'gen')

        Reorders rows of ppc['reserves']['cost'] to match internal generator
        ordering.

        ppc = e2i_field(ppc, ['reserves', 'zones'], 'gen', 1)

        Reorders columns of ppc['reserves']['zones'] to match internal
        generator ordering.

    @see: L{i2e_field}, L{e2i_data}, L{ext2int}
    """
    if isinstance(field, str):
        key = '["%s"]' % field
    else:
        key = '["%s"]' % '"]["'.join(field)

        v_ext = ppc["order"]["ext"]
        for fld in field:
            if fld not in v_ext:
                v_ext[fld] = {}
                v_ext = v_ext[fld]

    exec('ppc["order"]["ext"]%s = ppc%s.copy()' % (key, key))
    exec('ppc%s = e2i_data(ppc, ppc%s, ordering, dim)' % (key, key))

    return ppc


def ext2int(ppc, val_or_field=None, ordering=None, dim=0):
    """
    Converts external to internal indexing.

    This function has two forms, the old form that operates on
    and returns individual matrices and the new form that operates
    on and returns an entire PYPOWER case dict.

    1.  C{ppc = ext2int(ppc)}

    If the input is a single PYPOWER case dict, then all isolated
    buses, off-line generators and branches are removed along with any
    generators, branches or areas connected to isolated buses. Then the
    buses are renumbered consecutively, beginning at 0, and the
    generators are sorted by increasing bus number. Any 'ext2int'
    callback routines registered in the case are also invoked
    automatically. All of the related
    indexing information and the original data matrices are stored under
    the 'order' key of the dict to be used by C{int2ext} to perform
    the reverse conversions. If the case is already using internal
    numbering it is returned unchanged.

    Example::
        ppc = ext2int(ppc)

    @see: L{int2ext}, L{e2i_field}, L{e2i_data}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ppc = deepcopy(ppc)
    if val_or_field is None:  # nargin == 1
        first = 'order' not in ppc
        if first or ppc["order"]["state"] == 'e':
            # initialize order
            if first:
                o = {
                    'ext':      {
                        'bus':      None,
                        'branch':   None,
                        'gen':      None
                    },
                    'bus':      {'e2i':      None,
                                 'i2e':      None,
                                 'status':   {}},
                    'gen':      {'e2i':      None,
                                 'i2e':      None,
                                 'status':   {}},
                    'branch':   {'status': {}}
                }
            else:
                o = ppc["order"]

            # sizes
            nb = ppc["bus"].shape[0]
            ng = ppc["gen"].shape[0]
            ng0 = ng
            if 'A' in ppc:
                dc = True if ppc["A"].shape[1] < (2 * nb + 2 * ng) else False
            elif 'N' in ppc:
                dc = True if ppc["N"].shape[1] < (2 * nb + 2 * ng) else False
            else:
                dc = False

            # save data matrices with external ordering
            if 'ext' not in o:
                o['ext'] = {}
            # Note: these dictionaries contain mixed float/int data,
            # so don't cast them all astype(int) for numpy/scipy indexing
            o["ext"]["bus"] = ppc["bus"].copy()
            o["ext"]["branch"] = ppc["branch"].copy()
            o["ext"]["gen"] = ppc["gen"].copy()
            if 'areas' in ppc:
                if len(ppc["areas"]) == 0:  # if areas field is empty
                    del ppc['areas']  # delete it (so it's ignored)
                else:  # otherwise
                    o["ext"]["areas"] = ppc["areas"].copy()  # save it

            # check that all buses have a valid BUS_TYPE
            bt = ppc["bus"][:, IDX.bus.BUS_TYPE]
            err = find(~((bt == IDX.bus.PQ) | (bt == IDX.bus.PV) |
                       (bt == IDX.bus.REF) | (bt == IDX.bus.NONE)))
            if len(err) > 0:
                logger.debug('ext2int: bus %d has an invalid BUS_TYPE\n' % err)

            # determine which buses, branches, gens are connected and
            # in-service
            n2i = c_sparse((range(nb), (ppc["bus"][:, IDX.bus.BUS_I], np.zeros(nb))),
                           shape=(max(ppc["bus"][:, IDX.bus.BUS_I].astype(int)) + 1, 1))
            n2i = (np.array(n2i.todense().flatten())[0, :]).astype(int)  # as 1D array
            bs = (bt != IDX.bus.NONE)  # bus status
            o["bus"]["status"]["on"] = find(bs)  # connected
            o["bus"]["status"]["off"] = find(~bs)  # isolated
            gs = ((ppc["gen"][:, IDX.gen.GEN_STATUS] > 0) &  # gen status
                  bs[n2i[ppc["gen"][:, IDX.gen.GEN_BUS].astype(int)]])
            o["gen"]["status"]["on"] = find(gs)  # on and connected
            o["gen"]["status"]["off"] = find(~gs)  # off or isolated
            brs = (ppc["branch"][:, IDX.branch.BR_STATUS].astype(int) &  # branch status
                   bs[n2i[ppc["branch"][:, IDX.branch.F_BUS].astype(int)]] &
                   bs[n2i[ppc["branch"][:, IDX.branch.T_BUS].astype(int)]]).astype(bool)
            o["branch"]["status"]["on"] = find(brs)  # on and conn
            o["branch"]["status"]["off"] = find(~brs)
            if 'areas' in ppc:
                ar = bs[n2i[ppc["areas"][:, IDX.area.PRICE_REF_BUS].astype(int)]]
                o["areas"] = {"status": {}}
                o["areas"]["status"]["on"] = find(ar)
                o["areas"]["status"]["off"] = find(~ar)

            # delete stuff that is "out"
            if len(o["bus"]["status"]["off"]) > 0:
                #                ppc["bus"][o["bus"]["status"]["off"], :] = array([])
                ppc["bus"] = ppc["bus"][o["bus"]["status"]["on"], :]
            if len(o["branch"]["status"]["off"]) > 0:
                #                ppc["branch"][o["branch"]["status"]["off"], :] = array([])
                ppc["branch"] = ppc["branch"][o["branch"]["status"]["on"], :]
            if len(o["gen"]["status"]["off"]) > 0:
                #                ppc["gen"][o["gen"]["status"]["off"], :] = array([])
                ppc["gen"] = ppc["gen"][o["gen"]["status"]["on"], :]
            if 'areas' in ppc and (len(o["areas"]["status"]["off"]) > 0):
                #                ppc["areas"][o["areas"]["status"]["off"], :] = array([])
                ppc["areas"] = ppc["areas"][o["areas"]["status"]["on"], :]

            # update size
            nb = ppc["bus"].shape[0]

            # apply consecutive bus numbering
            o["bus"]["i2e"] = ppc["bus"][:, IDX.bus.BUS_I].copy()
            o["bus"]["e2i"] = np.zeros(max(o["bus"]["i2e"]).astype(int) + 1)
            o["bus"]["e2i"][o["bus"]["i2e"].astype(int)] = np.arange(nb)
            ppc["bus"][:, IDX.bus.BUS_I] = \
                o["bus"]["e2i"][ppc["bus"][:, IDX.bus.BUS_I].astype(int)].copy()
            ppc["gen"][:, IDX.gen.GEN_BUS] = \
                o["bus"]["e2i"][ppc["gen"][:, IDX.gen.GEN_BUS].astype(int)].copy()
            ppc["branch"][:, IDX.branch.F_BUS] = \
                o["bus"]["e2i"][ppc["branch"][:, IDX.branch.F_BUS].astype(int)].copy()
            ppc["branch"][:, IDX.branch.T_BUS] = \
                o["bus"]["e2i"][ppc["branch"][:, IDX.branch.T_BUS].astype(int)].copy()
            if 'areas' in ppc:
                ppc["areas"][:, IDX.area.PRICE_REF_BUS] = \
                    o["bus"]["e2i"][ppc["areas"][:,
                                                 IDX.area.PRICE_REF_BUS].astype(int)].copy()

            # reorder gens in order of increasing bus number
            o["gen"]["e2i"] = np.argsort(ppc["gen"][:, IDX.gen.GEN_BUS])
            o["gen"]["i2e"] = np.argsort(o["gen"]["e2i"])

            ppc["gen"] = ppc["gen"][o["gen"]["e2i"].astype(int), :]

            if 'int' in o:
                del o['int']
            o["state"] = 'i'
            ppc["order"] = o

            # update gencost, A and N
            if 'gencost' in ppc:
                ordering = ['gen']  # Pg cost only
                if ppc["gencost"].shape[0] == (2 * ng0):
                    ordering.append('gen')  # include Qg cost
                ppc = e2i_field(ppc, 'gencost', ordering)
            if 'A' in ppc or 'N' in ppc:
                if dc:
                    ordering = ['bus', 'gen']
                else:
                    ordering = ['bus', 'bus', 'gen', 'gen']
            if 'A' in ppc:
                ppc = e2i_field(ppc, 'A', ordering, 1)
            if 'N' in ppc:
                ppc = e2i_field(ppc, 'N', ordering, 1)

            # execute userfcn callbacks for 'ext2int' stage
            if 'userfcn' in ppc:
                ppc = run_userfcn(ppc['userfcn'], 'ext2int', ppc)
    else:  # convert extra data
        if isinstance(val_or_field, str) or isinstance(val_or_field, list):
            # field
            logger.warning('Calls of the form ppc = ext2int(ppc, '
                           '\'field_name\', ...) have been deprecated. Please '
                           'replace ext2int with e2i_field.', DeprecationWarning)
            gen, branch = val_or_field, ordering
            ppc = e2i_field(ppc, gen, branch, dim)

        else:
            # value
            logger.warning('Calls of the form val = ext2int(ppc, val, ...) have been '
                           'deprecated. Please replace ext2int with e2i_data.',
                           DeprecationWarning)
            gen, branch = val_or_field, ordering
            ppc = e2i_data(ppc, gen, branch, dim)

    return ppc


def ext2int1(bus, gen, branch, areas=None):
    """
    Converts from (possibly non-consecutive) external bus numbers to
    consecutive internal bus numbers which start at 1. Changes are made
    to BUS, GEN, BRANCH and optionally AREAS matrices, which are returned
    along with a vector of indices I2E that can be passed to INT2EXT to
    perform the reverse conversion.

    @see: L{int2ext}
    @see: U{http://www.pserc.cornell.edu/matpower/}
    """
    i2e = bus[:, IDX.bus.BUS_I].astype(int)
    e2i = np.zeros(max(i2e) + 1)
    e2i[i2e] = np.arange(bus.shape[0])

    bus[:, IDX.bus.BUS_I] = e2i[bus[:, IDX.bus.BUS_I].astype(int)]
    gen[:, IDX.gen.GEN_BUS] = e2i[gen[:, IDX.gen.GEN_BUS].astype(int)]
    branch[:, IDX.branch.F_BUS] = e2i[branch[:, IDX.branch.F_BUS].astype(int)]
    branch[:, IDX.branch.T_BUS] = e2i[branch[:, IDX.branch.T_BUS].astype(int)]
    if areas is not None and len(areas) > 0:
        areas[:, IDX.area.PRICE_REF_BUS] = e2i[areas[:, IDX.area.PRICE_REF_BUS].astype(int)]

        return i2e, bus, gen, branch, areas

    return i2e, bus, gen, branch


def i2e_data(ppc, val, oldval, ordering, dim=0):
    """
    Converts data from internal to external bus numbering.

    Parameters
    ----------
    ppc : dict
        The case dict.
    val : Numpy.array
        The data to be converted.
    oldval : Numpy.array
        The data to be used for off-line gens, branches, isolated buses,
        connected gens and branches.
    ordering : str or list of str
        The ordering of the data. Can be one of the following three
        strings: 'bus', 'gen' or 'branch'. For data structures with
        multiple blocks of data, ordered by bus, gen or branch, they
        can be converted with a single call by specifying C[ordering}
        as a list of strings.
    dim : int, optional
        The dimension to reorder. Default is 0.

    Returns
    -------
    val : Numpy.array
        The converted data.

    Examples
    --------
    Converts an A matrix for user-supplied OPF constraints from
    internal to external ordering, where the columns of the A
    matrix correspond to bus voltage angles, then voltage
    magnitudes, then generator real power injections and finally
    generator reactive power injections.
    >>> A_ext = i2e_data(ppc, A_int, A_orig, ['bus','bus','gen','gen'], 1)

    Converts a C{gencost} matrix that has both real and reactive power
    costs (in rows 1--ng and ng+1--2*ng, respectively).   

    >>> gencost_ext = i2e_data(ppc, gencost_int, gencost_orig, ['gen','gen'], 0)

    For a case dict using internal indexing, this function can be 
    used to convert other data structures as well by passing in 3 or 4
    extra parameters in addition to the case dict. If the value passed
    in the 2nd argument C{val} is a column vector, it will be converted
    according to the ordering specified by the 4th argument (C{ordering},
    described below). If C{val} is an n-dimensional matrix, then the
    optional 5th argument (C{dim}, default = 0) can be used to specify
    which dimension to reorder. The 3rd argument (C{oldval}) is used to
    initialize the return value before converting C{val} to external
    indexing. In particular, any data corresponding to off-line gens
    or branches or isolated buses or any connected gens or branches
    will be taken from C{oldval}, with C[val} supplying the rest of the
    returned data.

    The C{ordering} argument is used to indicate whether the data
    corresponds to bus-, gen- or branch-ordered data. It can be one
    of the following three strings: 'bus', 'gen' or 'branch'. For
    data structures with multiple blocks of data, ordered by bus,
    gen or branch, they can be converted with a single call by
    specifying C[ordering} as a list of strings.

    Any extra elements, rows, columns, etc. beyond those indicated
    in C{ordering}, are not disturbed.

    @see: L{e2i_data}, L{i2e_field}, L{int2ext}.
    """
    if 'order' not in ppc:
        logger.debug('i2e_data: ppc does not have the \'order\' field '
                     'required for conversion back to external numbering.\n')
        return

    o = ppc["order"]
    if o['state'] != 'i':
        logger.debug('i2e_data: ppc does not appear to be in internal '
                     'order\n')
        return

    if isinstance(ordering, str):  # single set
        if ordering == 'gen':
            v = get_reorder(val, o[ordering]["i2e"], dim)
        else:
            v = val
        val = set_reorder(oldval, v, o[ordering]["status"]["on"], dim)
    else:  # multiple sets
        be = 0  # base, external indexing
        bi = 0  # base, internal indexing
        new_v = []
        for ordr in ordering:
            ne = o["ext"][ordr].shape[0]
            ni = ppc[ordr].shape[0]
            v = get_reorder(val, bi + np.arange(ni), dim)
            oldv = get_reorder(oldval, be + np.arange(ne), dim)
            new_v.append(int2ext(ppc, v, oldv, ordr, dim))
            be = be + ne
            bi = bi + ni
        ni = val.shape[dim]
        if ni > bi:  # the rest
            v = get_reorder(val, np.arange(bi, ni), dim)
            new_v.append(v)
        val = np.concatenate(new_v, dim)

    return val


def i2e_field(ppc, field, ordering, dim=0):
    """
    Converts fields of MPC from internal to external bus numbering.

    Parameters
    ----------
    ppc : dict
        The case dict.
    field : str or list of str
        The field to be converted. If C{field} is a list of strings,
        they specify nested fields.
    ordering : str or list of str
        The ordering of the data. Can be one of the following three
        strings: 'bus', 'gen' or 'branch'. For data structures with
        multiple blocks of data, ordered by bus, gen or branch, they
        can be converted with a single call by specifying C[ordering}
        as a list of strings.
    dim : int, optional
        The dimension to reorder. Default is 0.

    Returns
    -------
    ppc : dict
        The updated case dict.

    For a case dict using internal indexing, this function can be
    used to convert other data structures as well by passing in 2 or 3
    extra parameters in addition to the case dict.

    If the 2nd argument is a string or list of strings, it
    specifies a field in the case dict whose value should be
    converted by L{i2e_data}. In this case, the corresponding
    C{oldval} is taken from where it was stored by L{ext2int} in
    ppc['order']['ext'] and the updated case dict is returned.
    If C{field} is a list of strings, they specify nested fields.

    The 3rd and optional 4th arguments are simply passed along to
    the call to L{i2e_data}.

    Examples:
        ppc = i2e_field(ppc, ['reserves', 'cost'], 'gen')

        Reorders rows of ppc['reserves']['cost'] to match external generator
        ordering.

        ppc = i2e_field(ppc, ['reserves', 'zones'], 'gen', 1)

        Reorders columns of ppc.reserves.zones to match external
        generator ordering.

    @see: L{e2i_field}, L{i2e_data}, L{int2ext}.
    """
    if 'int' not in ppc['order']:
        ppc['order']['int'] = {}

    if isinstance(field, str):
        key = '["%s"]' % field
    else:  # nested dicts
        key = '["%s"]' % '"]["'.join(field)

        v_int = ppc["order"]["int"]
        for fld in field:
            if fld not in v_int:
                v_int[fld] = {}
                v_int = v_int[fld]

    exec('ppc["order"]["int"]%s = ppc%s.copy()' % (key, key))
    exec('ppc%s = i2e_data(ppc, ppc%s, ppc["order"]["ext"]%s, ordering, dim)' %
         (key, key, key))

    return ppc


def total_load(bus, gen=None, load_zone=None, which_type=None):
    """
    Returns vector of total load in each load zone.

    @param bus: standard C{bus} matrix with C{nb} rows, where the fixed active
    and reactive loads are specified in columns C{PD} and C{QD}

    @param gen: (optional) standard C{gen} matrix with C{ng} rows, where the
    dispatchable loads are specified by columns C{PG}, C{QG}, C{PMIN},
    C{QMIN} and C{QMAX} (in rows for which C{isload(GEN)} returns C{True}).
    If C{gen} is empty, it assumes there are no dispatchable loads.

    @param load_zone: (optional) C{nb} element vector where the value of
    each element is either zero or the index of the load zone
    to which the corresponding bus belongs. If C{load_zone(b) = k}
    then the loads at bus C{b} will added to the values of C{Pd[k]} and
    C{Qd[k]}. If C{load_zone} is empty, the default is defined as the areas
    specified in the C{bus} matrix, i.e. C{load_zone =  bus[:, BUS_AREA]}
    and load will have dimension C{= max(bus[:, BUS_AREA])}. If
    C{load_zone = 'all'}, the result is a scalar with the total system
    load.

    @param which_type: (default is 'BOTH' if C{gen} is provided, else 'FIXED')
        - 'FIXED'        : sum only fixed loads
        - 'DISPATCHABLE' : sum only dispatchable loads
        - 'BOTH'         : sum both fixed and dispatchable loads

    @see: L{scale_load}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    nb = bus.shape[0]  # number of buses

    if gen is None:
        gen = np.array([])
    if load_zone is None:
        load_zone = np.array([], int)

    # fill out and check which_type
    if len(gen) == 0:
        which_type = 'FIXED'

    if (which_type == None) and (len(gen) > 0):
        which_type = 'BOTH'  # 'FIXED', 'DISPATCHABLE' or 'BOTH'

    if (which_type[0] != 'F') and (which_type[0] != 'D') and (which_type[0] != 'B'):
        logger.debug("total_load: which_type should be 'FIXED, 'DISPATCHABLE or 'BOTH'\n")

    want_Q = True
    want_fixed = (which_type[0] == 'B') | (which_type[0] == 'F')
    want_disp = (which_type[0] == 'B') | (which_type[0] == 'D')

    # initialize load_zone
    if isinstance(load_zone, str) and (load_zone == 'all'):
        load_zone = np.ones(nb, int)  # make a single zone of all buses
    elif len(load_zone) == 0:
        load_zone = bus[:, IDX.bus.BUS_AREA].astype(int)  # use areas defined in bus data as zones

    nz = max(load_zone)  # number of load zones

    # fixed load at each bus, & initialize dispatchable
    if want_fixed:
        Pdf = bus[:, IDX.bus.PD]  # real power
        if want_Q:
            Qdf = bus[:, IDX.bus.QD]  # reactive power
    else:
        Pdf = np.zeros(nb)  # real power
        if want_Q:
            Qdf = np.zeros(nb)  # reactive power

    # dispatchable load at each bus
    if want_disp:  # need dispatchable
        ng = gen.shape[0]
        is_ld = isload(gen) & (gen[:, IDX.gen.GEN_STATUS] > 0)
        ld = find(is_ld)

        # create map of external bus numbers to bus indices
        i2e = bus[:, IDX.bus.BUS_I].astype(int)
        e2i = zeros(max(i2e) + 1)
        e2i[i2e] = arange(nb)

        gbus = gen[:, IDX.gen.GEN_BUS].astype(int)
        Cld = c_sparse((is_ld, (e2i[gbus], np.arange(ng))), (nb, ng))
        Pdd = -Cld * gen[:, IDX.gen.PMIN]  # real power
        if want_Q:
            Q = np.zeros(ng)
            Q[ld] = (gen[ld, IDX.gen.QMIN] == 0) * gen[ld, IDX.gen.QMAX] + \
                    (gen[ld, IDX.gen.QMAX] == 0) * gen[ld, IDX.gen.QMIN]
            Qdd = -Cld * Q  # reactive power
    else:
        Pdd = np.zeros(nb)
        if want_Q:
            Qdd = np.zeros(nb)

    # compute load sums
    Pd = np.zeros(nz)
    if want_Q:
        Qd = np.zeros(nz)

    for k in range(1, nz + 1):
        idx = find(load_zone == k)
        Pd[k - 1] = sum(Pdf[idx]) + sum(Pdd[idx])
        if want_Q:
            Qd[k - 1] = sum(Qdf[idx]) + sum(Qdd[idx])

    return Pd, Qd
