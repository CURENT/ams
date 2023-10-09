"""
PYPOWER solver interface.
"""
import logging  # NOQA
from functools import wraps  # NOQA
import re  # NOQA

import numpy as np  # NOQA
from numpy import flatnonzero as find  # NOQA

import scipy.sparse as sp  # NOQA
from scipy.sparse import csr_matrix as c_sparse

from andes.shared import rad2deg, deg2rad  # NOQA

from ams.pypower.core.pips import pips  # NOQA
from ams.pypower.make import makeYbus  # NOQA
import ams.pypower.utils as putils  # NOQA
from ams.pypower.utils import IDX  # NOQA
from ams.pypower.routines.opffcns import opf_costfcn, opf_consfcn, opf_hessfcn  # NOQA


logger = logging.getLogger(__name__)


def require_mosek(f):
    """
    Decorator for functions that require mosek.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            from pymosek import mosekopt  # NOQA
        except ImportError:
            raise ModuleNotFoundError("Package `pymosek` needs to be manually installed.")

        return f(*args, **kwds)

    return wrapper


def require_cplex(f):
    """
    Decorator for functions that require cplex.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            from cplex import Cplex, cplexlp, cplexqp, cplexoptimset  # NOQA
        except ImportError:
            raise ModuleNotFoundError("Package `cplex` needs to be manually installed.")

        return f(*args, **kwds)

    return wrapper


def require_gurobi(f):
    """
    Decorator for functions that require gurobi.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            from gurobipy import Model, GRB  # NOQA
        except ImportError:
            raise ModuleNotFoundError("Package `gurobipy` needs to be manually installed.")

        return f(*args, **kwds)

    return wrapper


def require_ipopt(f):
    """
    Decorator for functions that require ipopt.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            import pyipopt  # NOQA
        except ImportError:
            raise ModuleNotFoundError("Package `pyipopt` needs to be manually installed.")

        return f(*args, **kwds)

    return wrapper


@require_ipopt
def ipoptopf_solver(om, ppopt):
    """
    Solves AC optimal power flow using IPOPT.

    Inputs are an OPF model object and a PYPOWER options vector.

    Outputs are a C{results} dict, C{success} flag and C{raw} output dict.

    C{results} is a PYPOWER case dict (ppc) with the usual C{baseMVA}, C{bus}
    C{branch}, C{gen}, C{gencost} fields, along with the following additional
    fields:
        - C{order}      see 'help ext2int' for details of this field
        - C{x}          final value of optimization variables (internal order)
        - C{f}          final objective function value
        - C{mu}         shadow prices on ...
            - C{var}
                - C{l}  lower bounds on variables
                - C{u}  upper bounds on variables
            - C{nln}
                - C{l}  lower bounds on nonlinear constraints
                - C{u}  upper bounds on nonlinear constraints
            - C{lin}
                - C{l}  lower bounds on linear constraints
                - C{u}  upper bounds on linear constraints

    C{success} is C{True} if solver converged successfully, C{False} otherwise

    C{raw} is a raw output dict in form returned by MINOS
        - C{xr}     final value of optimization variables
        - C{pimul}  constraint multipliers
        - C{info}   solver specific termination code
        - C{output} solver specific output information

    @see: L{opf}, L{pips}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    # unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = \
        ppc['baseMVA'], ppc['bus'], ppc['gen'], ppc['branch'], ppc['gencost']
    vv, _, nn, _ = om.get_idx()

    # problem dimensions
    nb = np.shape(bus)[0]  # number of buses
    ng = np.shape(gen)[0]  # number of gens
    nl = np.shape(branch)[0]  # number of branches
    ny = om.getN('var', 'y')  # number of piece-wise linear costs

    # linear constraints
    A, l, u = om.linear_constraints()

    # bounds on optimization vars
    _, xmin, xmax = om.getv()

    # build admittance matrices
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    # try to select an interior initial point
    ll = xmin.copy()
    uu = xmax.copy()
    ll[xmin == -np.Inf] = -2e19  # replace Inf with numerical proxies
    uu[xmax == np.Inf] = 2e19
    x0 = (ll + uu) / 2
    Varefs = bus[bus[:, IDX.bus.BUS_TYPE] == IDX.bus.REF, IDX.bus.VA] * deg2rad
    x0[vv['i1']['Va']:vv['iN']['Va']] = Varefs[0]  # angles set to first reference angle
    if ny > 0:
        ipwl = find(gencost[:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR)
#        PQ = np.r_[gen[:, PMAX], gen[:, QMAX]]
#        c = totcost(gencost[ipwl, :], PQ[ipwl])
        # largest y-value in CCV data
        c = gencost.flatten('F')[putils.sub2ind(np.shape(gencost), ipwl,
                                                IDX.cost.NCOST + 2 * gencost[ipwl, IDX.cost.NCOST])]
        x0[vv['i1']['y']:vv['iN']['y']] = max(c) + 0.1 * np.abs(max(c))
#        x0[vv['i1']['y']:vv['iN']['y']) = c + 0.1 * np.abs(c)

    # find branches with flow limits
    il = find((branch[:, IDX.branch.RATE_A] != 0) & (branch[:, IDX.branch.RATE_A] < 1e10))
    nl2 = len(il)  # number of constrained lines

    # -----  run opf  -----
    # build Jacobian and Hessian structure
    if A is not None and sp.issparse(A):
        nA = A.shape[0]  # number of original linear constraints
    else:
        nA = 0
    nx = len(x0)
    f = branch[:, IDX.branch.F_BUS]  # list of "from" buses
    t = branch[:, IDX.branch.T_BUS]  # list of "to" buses
    Cf = c_sparse((np.ones(nl), (np.arange(nl), f)), (nl, nb))  # connection matrix for line & from buses
    Ct = c_sparse((np.ones(nl), (np.arange(nl), t)), (nl, nb))  # connection matrix for line & to buses
    Cl = Cf + Ct
    Cb = Cl.T * Cl + sp.eye(nb, nb)
    Cl2 = Cl[il, :]
    Cg = c_sparse((np.ones(ng), (gen[:, IDX.gen.GEN_BUS], np.arange(ng))), (nb, ng))
    nz = nx - 2 * (nb + ng)
    nxtra = nx - 2 * nb
    if nz > 0:
        Js = sp.vstack([
            sp.hstack([Cb, Cb, Cg, c_sparse((nb, ng)), c_sparse((nb,  nz))]),
            sp.hstack([Cb, Cb, c_sparse((nb, ng)), Cg, c_sparse((nb,  nz))]),
            sp.hstack([Cl2, Cl2, c_sparse((nl2, 2 * ng)), c_sparse((nl2, nz))]),
            sp.hstack([Cl2, Cl2, c_sparse((nl2, 2 * ng)), c_sparse((nl2, nz))])
        ], 'coo')
    else:
        Js = sp.vstack([
            sp.hstack([Cb, Cb, Cg, c_sparse((nb, ng))]),
            sp.hstack([Cb, Cb, c_sparse((nb, ng)), Cg,]),
            sp.hstack([Cl2, Cl2, c_sparse((nl2, 2 * ng)),]),
            sp.hstack([Cl2, Cl2, c_sparse((nl2, 2 * ng)),])
        ], 'coo')

    if A is not None and sp.issparse(A):
        Js = sp.vstack([Js, A], 'coo')

    f, _, d2f = opf_costfcn(x0, om, True)
    Hs = sp.tril(d2f + sp.vstack([
        sp.hstack([Cb,  Cb,  c_sparse((nb, nxtra))]),
        sp.hstack([Cb,  Cb,  c_sparse((nb, nxtra))]),
        c_sparse((nxtra, nx))
    ]), format='coo')

    # set options struct for IPOPT
#    options = {}
#    options['ipopt'] = ipopt_options([], ppopt)

    # extra data to pass to functions
    userdata = dict(om=om, Ybus=Ybus, Yf=Yf, Yt=Yt, ppopt=ppopt,
                    il=il, A=A, nA=nA,
                    neqnln=2 * nb, niqnln=2 * nl2, Js=Js, Hs=Hs)

    # check Jacobian and Hessian structure
    # xr                  = rand(x0.shape)
    # lmbda               = rand( 2 * nb + 2 * nl2)
    # Js1 = eval_jac_g(x, flag, userdata) #(xr, options.auxdata)
    # Hs1  = eval_h(xr, 1, lmbda, userdata)
    # i1, j1, s = find(Js)
    # i2, j2, s = find(Js1)
    # if (len(i1) != len(i2)) | (norm(i1 - i2) != 0) | (norm(j1 - j2) != 0):
    #    raise ValueError, 'something''s wrong with the Jacobian structure'
    #
    # i1, j1, s = find(Hs)
    # i2, j2, s = find(Hs1)
    # if (len(i1) != len(i2)) | (norm(i1 - i2) != 0) | (norm(j1 - j2) != 0):
    #    raise ValueError, 'something''s wrong with the Hessian structure'

    # define variable and constraint bounds
    # n is the number of variables
    n = x0.shape[0]
    # xl is the lower bound of x as bounded constraints
    xl = xmin
    # xu is the upper bound of x as bounded constraints
    xu = xmax

    neqnln = 2 * nb
    niqnln = 2 * nl2

    # number of constraints
    m = neqnln + niqnln + nA
    # lower bound of constraint
    gl = np.r_[np.zeros(neqnln), -np.Inf * np.ones(niqnln), l]
    # upper bound of constraints
    gu = np.r_[np.zeros(neqnln), np.zeros(niqnln),          u]

    # number of nonzeros in Jacobi matrix
    nnzj = Js.nnz
    # number of non-zeros in Hessian matrix, you can set it to 0
    nnzh = Hs.nnz

    eval_hessian = True
    if eval_hessian:
        def hessian(x, lagrange, obj_factor, flag, user_data=None): return \
            eval_h(x, lagrange, obj_factor, flag, userdata)

        nlp = pyipopt.create(n, xl, xu, m, gl, gu, nnzj, nnzh,
                             eval_f, eval_grad_f, eval_g, eval_jac_g, hessian)
    else:
        nnzh = 0
        nlp = pyipopt.create(n, xl, xu, m, gl, gu, nnzj, nnzh,
                             eval_f, eval_grad_f, eval_g, eval_jac_g)

    nlp.int_option('print_level', 5)
    nlp.num_option('tol', 1.0000e-12)
    nlp.int_option('max_iter', 250)
    nlp.num_option('dual_inf_tol', 0.10000)
    nlp.num_option('constr_viol_tol', 1.0000e-06)
    nlp.num_option('compl_inf_tol', 1.0000e-05)
    nlp.num_option('acceptable_tol', 1.0000e-08)
    nlp.num_option('acceptable_constr_viol_tol', 1.0000e-04)
    nlp.num_option('acceptable_compl_inf_tol', 0.0010000)
    nlp.str_option('mu_strategy', 'adaptive')

    iter = 0

    def intermediate_callback(algmod, iter_count, obj_value, inf_pr, inf_du,
                              mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials,
                              user_data=None):
        iter = iter_count
        return True

    nlp.set_intermediate_callback(intermediate_callback)

    # run the optimization
    # returns final solution x, upper and lower bound for multiplier, final
    # objective function obj and the return status of ipopt
    x, zl, zu, obj, status, zg = nlp.solve(x0, m, userdata)

    info = dict(x=x, zl=zl, zu=zu, obj=obj, status=status, lmbda=zg)

    nlp.close()

    success = (status == 0) | (status == 1)

    output = {'iterations': iter}

    f, _ = opf_costfcn(x, om)

    # update solution data
    Va = x[vv['i1']['Va']:vv['iN']['Va']]
    Vm = x[vv['i1']['Vm']:vv['iN']['Vm']]
    Pg = x[vv['i1']['Pg']:vv['iN']['Pg']]
    Qg = x[vv['i1']['Qg']:vv['iN']['Qg']]
    V = Vm * np.exp(1j * Va)

    # -----  calculate return values  -----
    # update voltages & generator outputs
    bus[:, IDX.bus.VA] = Va * rad2deg
    bus[:, IDX.bus.VM] = Vm
    gen[:, IDX.gen.PG] = Pg * baseMVA
    gen[:, IDX.gen.QG] = Qg * baseMVA
    gen[:, IDX.gen.VG] = Vm[gen[:, IDX.gen.GEN_BUS].astype(int)]

    # compute branch flows
    f_br = branch[:, IDX.branch.F_BUS].astype(int)
    t_br = branch[:, IDX.branch.T_BUS].astype(int)
    Sf = V[f_br] * np.conj(Yf * V)  # cplx pwr at "from" bus, p.u.
    St = V[t_br] * np.conj(Yt * V)  # cplx pwr at "to" bus, p.u.
    branch[:, IDX.branch.PF] = Sf.real * baseMVA
    branch[:, IDX.branch.QF] = Sf.imag * baseMVA
    branch[:, IDX.branch.PT] = St.real * baseMVA
    branch[:, IDX.branch.QT] = St.imag * baseMVA

    # line constraint is actually on square of limit
    # so we must fix multipliers
    muSf = np.zeros(nl)
    muSt = np.zeros(nl)
    if len(il) > 0:
        muSf[il] = 2 * info['lmbda'][2 * nb + np.arange(nl2)] * branch[il, IDX.branch.RATE_A] / baseMVA
        muSt[il] = 2 * info['lmbda'][2 * nb + nl2 + np.arange(nl2)] * branch[il, IDX.branch.RATE_A] / baseMVA

    # update Lagrange multipliers
    bus[:, IDX.bus.MU_VMAX] = info['zu'][vv['i1']['Vm']:vv['iN']['Vm']]
    bus[:, IDX.bus.MU_VMIN] = info['zl'][vv['i1']['Vm']:vv['iN']['Vm']]
    gen[:, IDX.gen.MU_PMAX] = info['zu'][vv['i1']['Pg']:vv['iN']['Pg']] / baseMVA
    gen[:, IDX.gen.MU_PMIN] = info['zl'][vv['i1']['Pg']:vv['iN']['Pg']] / baseMVA
    gen[:, IDX.gen.MU_QMAX] = info['zu'][vv['i1']['Qg']:vv['iN']['Qg']] / baseMVA
    gen[:, IDX.gen.MU_QMIN] = info['zl'][vv['i1']['Qg']:vv['iN']['Qg']] / baseMVA
    bus[:, IDX.bus.LAM_P] = info['lmbda'][nn['i1']['Pmis']:nn['iN']['Pmis']] / baseMVA
    bus[:, IDX.bus.LAM_Q] = info['lmbda'][nn['i1']['Qmis']:nn['iN']['Qmis']] / baseMVA
    branch[:, IDX.branch.MU_SF] = muSf / baseMVA
    branch[:, IDX.branch.MU_ST] = muSt / baseMVA

    # package up results
    nlnN = om.getN('nln')

    # extract multipliers for nonlinear constraints
    kl = find(info['lmbda'][:2 * nb] < 0)
    ku = find(info['lmbda'][:2 * nb] > 0)
    nl_mu_l = np.zeros(nlnN)
    nl_mu_u = np.r_[np.zeros(2 * nb), muSf, muSt]
    nl_mu_l[kl] = -info['lmbda'][kl]
    nl_mu_u[ku] = info['lmbda'][ku]

    # extract multipliers for linear constraints
    lam_lin = info['lmbda'][2 * nb + 2 * nl2 + np.arange(nA)]  # lmbda for linear constraints
    kl = find(lam_lin < 0)  # lower bound binding
    ku = find(lam_lin > 0)  # upper bound binding
    mu_l = np.zeros(nA)
    mu_l[kl] = -lam_lin[kl]
    mu_u = np.zeros(nA)
    mu_u[ku] = lam_lin[ku]

    mu = {
        'var': {'l': info['zl'], 'u': info['zu']},
        'nln': {'l': nl_mu_l, 'u': nl_mu_u},
        'lin': {'l': mu_l, 'u': mu_u}
    }

    results = ppc
    results['bus'], results['branch'], results['gen'], \
        results['om'], results['x'], results['mu'], results['f'] = \
        bus, branch, gen, om, x, mu, f

    pimul = np.r_[
        results['mu']['nln']['l'] - results['mu']['nln']['u'],
        results['mu']['lin']['l'] - results['mu']['lin']['u'],
        -np.ones(ny > 0),
        results['mu']['var']['l'] - results['mu']['var']['u']
    ]
    raw = {'xr': x, 'pimul': pimul, 'info': info['status'], 'output': output}

    return results, success, raw


def eval_f(x, user_data=None):
    """
    Calculates the objective value.

    @param x: input vector
    """
    om = user_data['om']
    f,  _ = opf_costfcn(x, om)
    return f


def eval_grad_f(x, user_data=None):
    """
    Calculates gradient for objective function.
    """
    om = user_data['om']
    _, df = opf_costfcn(x, om)
    return df


def eval_g(x, user_data=None):
    """
    Calculates the constraint values and returns an array.
    """
    om = user_data['om']
    Ybus = user_data['Ybus']
    Yf = user_data['Yf']
    Yt = user_data['Yt']
    ppopt = user_data['ppopt']
    il = user_data['il']
    A = user_data['A']

    hn, gn, _, _ = opf_consfcn(x, om, Ybus, Yf, Yt, ppopt, il)

    if A is not None and sp.issparse(A):
        c = np.r_[gn, hn, A * x]
    else:
        c = np.r_[gn, hn]
    return c


def eval_jac_g(x, flag, user_data=None):
    """
    Calculates the Jacobi matrix.

    If the flag is true, returns a tuple (row, col) to indicate the
    sparse Jacobi matrix's structure.
    If the flag is false, returns the values of the Jacobi matrix
    with length nnzj.
    """
    Js = user_data['Js']
    if flag:
        return (Js.row, Js.col)
    else:
        om = user_data['om']
        Ybus = user_data['Ybus']
        Yf = user_data['Yf']
        Yt = user_data['Yt']
        ppopt = user_data['ppopt']
        il = user_data['il']
        A = user_data['A']

        _, _, dhn, dgn = opf_consfcn(x, om, Ybus, Yf, Yt, ppopt, il)

        if A is not None and sp.issparse(A):
            J = sp.vstack([dgn.T, dhn.T, A], 'coo')
        else:
            J = sp.vstack([dgn.T, dhn.T], 'coo')

        # FIXME: Extend PyIPOPT to handle changes in sparsity structure
        nnzj = Js.nnz
        Jd = np.zeros(nnzj)
        Jc = J.tocsc()
        for i in range(nnzj):
            Jd[i] = Jc[Js.row[i], Js.col[i]]

        return Jd


def eval_h(x, lagrange, obj_factor, flag, user_data=None):
    """
    Calculates the Hessian matrix (optional).

    If omitted, set nnzh to 0 and Ipopt will use approximated Hessian
    which will make the convergence slower.
    """
    Hs = user_data['Hs']
    if flag:
        return (Hs.row, Hs.col)
    else:
        neqnln = user_data['neqnln']
        niqnln = user_data['niqnln']
        om = user_data['om']
        Ybus = user_data['Ybus']
        Yf = user_data['Yf']
        Yt = user_data['Yt']
        ppopt = user_data['ppopt']
        il = user_data['il']

        lam = {}
        lam['eqnonlin'] = lagrange[:neqnln]
        lam['ineqnonlin'] = lagrange[np.arange(niqnln) + neqnln]

        H = opf_hessfcn(x, lam, om, Ybus, Yf, Yt, ppopt, il, obj_factor)

        Hl = sp.tril(H, format='csc')

        # FIXME: Extend PyIPOPT to handle changes in sparsity structure
        nnzh = Hs.nnz
        Hd = np.zeros(nnzh)
        for i in range(nnzh):
            Hd[i] = Hl[Hs.row[i], Hs.col[i]]

        return Hd


def qps_cplex(H, c, A, l, u, xmin, xmax, x0, opt):
    """Quadratic Program Solver based on CPLEX.

    A wrapper function providing a PYPOWER standardized interface for using
    C{cplexqp} or C{cplexlp} to solve the following QP (quadratic programming)
    problem::

        min 1/2 X'*H*x + c'*x
         x

    subject to::

        l <= A*x <= u       (linear constraints)
        xmin <= x <= xmax   (variable bounds)

    Inputs (all optional except C{H}, C{c}, C{A} and C{l}):
        - C{H} : matrix (possibly sparse) of quadratic cost coefficients
        - C{c} : vector of linear cost coefficients
        - C{A, l, u} : define the optional linear constraints. Default
        values for the elements of L and U are -Inf and Inf, respectively.
        - C{xmin, xmax} : optional lower and upper bounds on the
        C{x} variables, defaults are -Inf and Inf, respectively.
        - C{x0} : optional starting value of optimization vector C{x}
        - C{opt} : optional options structure with the following fields,
        all of which are also optional (default values shown in parentheses)
            - C{verbose} (0) - controls level of progress output displayed
                - 0 = no progress output
                - 1 = some progress output
                - 2 = verbose progress output
            - C{cplex_opt} - options dict for CPLEX, value in
            verbose overrides these options
        - C{problem} : The inputs can alternatively be supplied in a single
        C{problem} dict with fields corresponding to the input arguments
        described above: C{H, c, A, l, u, xmin, xmax, x0, opt}

    Outputs:
        - C{x} : solution vector
        - C{f} : final objective function value
        - C{exitflag} : CPLEXQP/CPLEXLP exit flag
        (see C{cplexqp} and C{cplexlp} documentation for details)
        - C{output} : CPLEXQP/CPLEXLP output dict
        (see C{cplexqp} and C{cplexlp} documentation for details)
        - C{lmbda} : dict containing the Langrange and Kuhn-Tucker
        multipliers on the constraints, with fields:
            - mu_l - lower (left-hand) limit on linear constraints
            - mu_u - upper (right-hand) limit on linear constraints
            - lower - lower bound on optimization variables
            - upper - upper bound on optimization variables

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # ----- input argument handling  -----
    # gather inputs
    if isinstance(H, dict):  # problem struct
        p = H
        if 'opt' in p:
            opt = p['opt']
        if 'x0' in p:
            x0 = p['x0']
        if 'xmax' in p:
            xmax = p['xmax']
        if 'xmin' in p:
            xmin = p['xmin']
        if 'u' in p:
            u = p['u']
        if 'l' in p:
            l = p['l']
        if 'A' in p:
            A = p['A']
        if 'c' in p:
            c = p['c']
        if 'H' in p:
            H = p['H']
    else:  # individual args
        assert H is not None
        assert c is not None
        assert A is not None
        assert l is not None

    if opt is None:
        opt = {}
#    if x0 is None:
#        x0 = np.array([])
#    if xmax is None:
#        xmax = np.array([])
#    if xmin is None:
#        xmin = np.array([])

    # define nx, set default values for missing optional inputs
    if len(H) == 0 or not np.any(np.any(H)):
        if len(A) == 0 and len(xmin) == 0 and len(xmax) == 0:
            logger.debug('qps_cplex: LP problem must include constraints or variable bounds\n')
        else:
            if len(A) > 0:
                nx = np.shape(A)[1]
            elif len(xmin) > 0:
                nx = len(xmin)
            else:    # if len(xmax) > 0
                nx = len(xmax)
    else:
        nx = np.shape(H)[0]

    if len(c) == 0:
        c = np.zeros(nx)

    if len(A) > 0 and (len(l) == 0 or all(l == -np.Inf)) and \
            (len(u) == 0 or all(u == np.Inf)):
        A = None  # no limits => no linear constraints

    nA = np.shape(A)[0]  # number of original linear constraints
    if len(u) == 0:  # By default, linear inequalities are ...
        u = np.Inf * np.ones(nA)  # ... unbounded above and ...

    if len(l) == 0:
        l = -np.Inf * np.ones(nA)  # ... unbounded below.

    if len(xmin) == 0:  # By default, optimization variables are ...
        xmin = -np.Inf * np.ones(nx)  # ... unbounded below and ...

    if len(xmax) == 0:
        xmax = np.Inf * np.ones(nx)  # ... unbounded above.

    if len(x0) == 0:
        x0 = np.zeros(nx)

    # default options
    if 'verbose' in opt:
        verbose = opt['verbose']
    else:
        verbose = 0

    # if 'max_it' in opt:
    #    max_it = opt['max_it']
    # else:
    #    max_it = 0

    # split up linear constraints
    ieq = find(np.abs(u-l) <= putils.EPS)  # equality
    igt = find(u >= 1e10 & l > -1e10)  # greater than, unbounded above
    ilt = find(l <= -1e10 & u < 1e10)  # less than, unbounded below
    ibx = find((np.abs(u-l) > putils.EPS) & (u < 1e10) & (l > -1e10))
    Ae = A[ieq, :]
    be = u[ieq]
    Ai = np.r_[A[ilt, :], -A[igt, :], A[ibx, :] - A[ibx, :]]
    bi = np.r_[u[ilt],    -l[igt],    u[ibx],   -l[ibx]]

    # grab some dimensions
    nlt = len(ilt)  # number of upper bounded linear inequalities
    ngt = len(igt)  # number of lower bounded linear inequalities
    nbx = len(ibx)  # number of doubly bounded linear inequalities

    # set up options struct for CPLEX
    if 'cplex_opt' in opt:
        cplex_opt = cplex_options(opt['cplex_opt'])
    else:
        cplex_opt = cplex_options

    cplex = Cplex('null')
    vstr = cplex.getVersion
    s, e, tE, m, t = re.compile(vstr, '(\d+\.\d+)\.')
    vnum = int(t[0][0])
    vrb = max([0, verbose - 1])
    cplex_opt['barrier']['display'] = vrb
    cplex_opt['conflict']['display'] = vrb
    cplex_opt['mip']['display'] = vrb
    cplex_opt['sifting']['display'] = vrb
    cplex_opt['simplex']['display'] = vrb
    cplex_opt['tune']['display'] = vrb
    if vrb and (vnum > 12.2):
        cplex_opt['diagnostics'] = 'on'
    # if max_it:
    #    cplex_opt.    ## not sure what to set here

    if len(Ai) == 0 and len(Ae) == 0:
        unconstrained = 1
        Ae = c_sparse((1, nx))
        be = 0
    else:
        unconstrained = 0

    # call the solver
    if verbose:
        methods = [
            'default',
            'primal simplex',
            'dual simplex',
            'network simplex',
            'barrier',
            'sifting',
            'concurrent'
        ]

    if len(H) == 0 or not np.any(np.any(H)):
        logger.info('CPLEX Version %s -- %s LP solver\n' %
                    (vstr, methods[cplex_opt['lpmethod'] + 1]))

        x, f, eflag, output, lam = \
            cplexlp(c, Ai, bi, Ae, be, xmin, xmax, x0, cplex_opt)
    else:
        logger.info('CPLEX Version %s --  %s QP solver\n' %
                    (vstr, methods[cplex_opt['qpmethod'] + 1]))
        # ensure H is numerically symmetric
        if H != H.T:
            H = (H + H.T) / 2

        x, f, eflag, output, lam = \
            cplexqp(H, c, Ai, bi, Ae, be, xmin, xmax, x0, cplex_opt)

    # check for empty results (in case optimization failed)
    if len(x) == 0:
        x = np.NaN * np.zeros(nx)

    if len(f) == 0:
        f = np.NaN

    if len(lam) == 0:
        lam['ineqlin'] = np.NaN * np.zeros(len(bi))
        lam['eqlin'] = np.NaN * np.zeros(len(be))
        lam['lower'] = np.NaN * np.zeros(nx)
        lam['upper'] = np.NaN * np.zeros(nx)
        mu_l = np.NaN * np.zeros(nA)
        mu_u = np.NaN * np.zeros(nA)
    else:
        mu_l = np.zeros(nA)
        mu_u = np.zeros(nA)

    if unconstrained:
        lam['eqlin'] = np.array([])

    # negate prices depending on version
    if vnum < 12.3:
        lam['eqlin'] = -lam['eqlin']
        lam['ineqlin'] = -lam['ineqlin']

    # repackage lambdas
    kl = find(lam.eqlin < 0)  # lower bound binding
    ku = find(lam.eqlin > 0)  # upper bound binding

    mu_l[ieq[kl]] = -lam['eqlin'][kl]
    mu_l[igt] = lam['ineqlin'][nlt + np.ones(ngt)]
    mu_l[ibx] = lam['ineqlin'][nlt + ngt + nbx + np.ones(nbx)]

    mu_u[ieq[ku]] = lam['eqlin'][ku]
    mu_u[ilt] = lam['ineqlin'][:nlt]
    mu_u[ibx] = lam['ineqlin'][nlt + ngt + np.ones(nbx)]

    lmbda = dict(mu_l=mu_l, mu_u=mu_u, lower=lam.lower, upper=lam.upper)

    return x, f, eflag, output, lmbda


@require_gurobi
def qps_gurobi(H, c, A, l, u, xmin, xmax, x0, opt):
    """
    Quadratic Program Solver based on GUROBI.

    A wrapper function providing a PYPOWER standardized interface for using
    gurobipy to solve the following QP (quadratic programming)
    problem:

        min 1/2 x'*H*x + c'*x
         x

    subject to

        l <= A*x <= u       (linear constraints)
        xmin <= x <= xmax   (variable bounds)

    Inputs (all optional except H, c, A and l):
        H : matrix (possibly sparse) of quadratic cost coefficients
        c : vector of linear cost coefficients
        A, l, u : define the optional linear constraints. Default
            values for the elements of l and u are -Inf and Inf,
            respectively.
        xmin, xmax : optional lower and upper bounds on the
            C{x} variables, defaults are -Inf and Inf, respectively.
        x0 : optional starting value of optimization vector C{x}
        opt : optional options structure with the following fields,
            all of which are also optional (default values shown in
            parentheses)
            verbose (0) - controls level of progress output displayed
                0 = no progress output
                1 = some progress output
                2 = verbose progress output
            grb_opt - options dict for Gurobi, value in
                verbose overrides these options
        problem : The inputs can alternatively be supplied in a single
            PROBLEM dict with fields corresponding to the input arguments
            described above: H, c, A, l, u, xmin, xmax, x0, opt

    Outputs:
        x : solution vector
        f : final objective function value
        exitflag : gurobipy exit flag
            1 = converged
            0 or negative values = negative of GUROBI_MEX exit flag
            (see gurobipy documentation for details)
        output : gurobipy output dict
            (see gurobipy documentation for details)
        lmbda : dict containing the Langrange and Kuhn-Tucker
            multipliers on the constraints, with fields:
            mu_l - lower (left-hand) limit on linear constraints
            mu_u - upper (right-hand) limit on linear constraints
            lower - lower bound on optimization variables
            upper - upper bound on optimization variables

    Note the calling syntax is almost identical to that of QUADPROG
    from MathWorks' Optimization Toolbox. The main difference is that
    the linear constraints are specified with A, l, u instead of
    A, b, Aeq, beq.

    Calling syntax options:
        x, f, exitflag, output, lmbda = ...
            qps_gurobi(H, c, A, l, u, xmin, xmax, x0, opt)

        r = qps_gurobi(H, c, A, l, u)
        r = qps_gurobi(H, c, A, l, u, xmin, xmax)
        r = qps_gurobi(H, c, A, l, u, xmin, xmax, x0)
        r = qps_gurobi(H, c, A, l, u, xmin, xmax, x0, opt)
        r = qps_gurobi(problem), where problem is a dict with fields:
                        H, c, A, l, u, xmin, xmax, x0, opt
                        all fields except 'c', 'A' and 'l' or 'u' are optional

    Example: (problem from from http://www.jmu.edu/docs/sasdoc/sashtml/iml/chap8/sect12.htm)
        H = [   1003.1  4.3     6.3     5.9;
                4.3     2.2     2.1     3.9;
                6.3     2.1     3.5     4.8;
                5.9     3.9     4.8     10  ]
        c = np.zeros((4, 1))
        A = [   [1       1       1       1]
                [0.17    0.11    0.10    0.18]    ]
        l = [1; 0.10]
        u = [1; Inf]
        xmin = np.zeros((4, 1))
        x0 = [1; 0; 0; 1]
        opt = {'verbose': 2}
        x, f, s, out, lmbda = qps_gurobi(H, c, A, l, u, xmin, [], x0, opt)

    @see: L{gurobipy}.
    """
    # ----- input argument handling  -----
    # gather inputs
    if isinstance(H, dict):  # problem struct
        p = H
        if 'opt' in p:
            opt = p['opt']
        if 'x0' in p:
            x0 = p['x0']
        if 'xmax' in p:
            xmax = p['xmax']
        if 'xmin' in p:
            xmin = p['xmin']
        if 'u' in p:
            u = p['u']
        if 'l' in p:
            l = p['l']
        if 'A' in p:
            A = p['A']
        if 'c' in p:
            c = p['c']
        if 'H' in p:
            H = p['H']
    else:  # individual args
        assert H is not None
        assert c is not None
        assert A is not None
        assert l is not None

    if opt is None:
        opt = {}
#    if x0 is None:
#        x0 = np.array([])
#    if xmax is None:
#        xmax = np.array([])
#    if xmin is None:
#        xmin = np.array([])

    # define nx, set default values for missing optional inputs
    if len(H) == 0 or not np.any(np.any(H)):
        if len(A) == 0 and len(xmin) == 0 and len(xmax) == 0:
            logger.debug('qps_gurobi: LP problem must include constraints or variable bounds\n')
        else:
            if len(A) > 0:
                nx = np.shape(A)[1]
            elif len(xmin) > 0:
                nx = len(xmin)
            else:    # if len(xmax) > 0
                nx = len(xmax)
        H = c_sparse((nx, nx))
    else:
        nx = np.shape(H)[0]

    if len(c) == 0:
        c = np.zeros(nx)

    if len(A) > 0 and (len(l) == 0 or all(l == -np.Inf)) and \
            (len(u) == 0 or all(u == np.Inf)):
        A = None  # no limits => no linear constraints

    nA = np.shape(A)[0]  # number of original linear constraints
    if nA:
        if len(u) == 0:  # By default, linear inequalities are ...
            u = np.Inf * np.ones(nA)  # ... unbounded above and ...

        if len(l) == 0:
            l = -np.Inf * np.ones(nA)  # ... unbounded below.

    if len(x0) == 0:
        x0 = np.zeros(nx)

    # default options
    if 'verbose' in opt:
        verbose = opt['verbose']
    else:
        verbose = 0

#    if 'max_it' in opt:
#        max_it = opt['max_it']
#    else:
#        max_it = 0

    # set up options struct for Gurobi
    if 'grb_opt' in opt:
        g_opt = gurobi_options(opt['grb_opt'])
    else:
        g_opt = gurobi_options()

    g_opt['Display'] = min(verbose, 3)
    if verbose:
        g_opt['DisplayInterval'] = 1
    else:
        g_opt['DisplayInterval'] = np.Inf

    if not sp.issparse(A):
        A = c_sparse(A)

    # split up linear constraints
    ieq = find(np.abs(u-l) <= putils.EPS)  # equality
    igt = find(u >= 1e10 & l > -1e10)  # greater than, unbounded above
    ilt = find(l <= -1e10 & u < 1e10)  # less than, unbounded below
    ibx = find((np.abs(u-l) > putils.EPS) & (u < 1e10) & (l > -1e10))

    # grab some dimensions
    nlt = len(ilt)  # number of upper bounded linear inequalities
    ngt = len(igt)  # number of lower bounded linear inequalities
    nbx = len(ibx)  # number of doubly bounded linear inequalities
    neq = len(ieq)  # number of equalities
    niq = nlt + ngt + 2 * nbx  # number of inequalities

    AA = [A[ieq, :], A[ilt, :], -A[igt, :], A[ibx, :], -A[ibx, :]]
    bb = [u[ieq],    u[ilt],    -l[igt],    u[ibx],    -l[ibx]]
    contypes = '=' * neq + '<' * niq

    # call the solver
    if len(H) == 0 or not np.any(np.any(H)):
        lpqp = 'LP'
    else:
        lpqp = 'QP'
        rr, cc, vv = find(H)
        g_opt['QP']['qrow'] = int(rr.T - 1)
        g_opt['QP']['qcol'] = int(cc.T - 1)
        g_opt['QP']['qval'] = 0.5 * vv.T

    if verbose:
        methods = [
            'primal simplex',
            'dual simplex',
            'interior point',
            'concurrent',
            'deterministic concurrent'
        ]
        logger.info('Gurobi Version %s -- %s %s solver\n'
                    '<unknown>' % (methods[g_opt['Method'] + 1], lpqp))

    x, f, eflag, output, lmbda = \
        gurobipy(c.T, 1, AA, bb, contypes, xmin, xmax, 'C', g_opt)
    pi = lmbda['Pi']
    rc = lmbda['RC']
    output['flag'] = eflag
    if eflag == 2:
        eflag = 1  # optimal solution found
    else:
        eflag = -eflag  # failed somehow

    # check for empty results (in case optimization failed)
    lam = {}
    if len(x) == 0:
        x = np.NaN(nx, 1)
        lam['lower'] = np.NaN(nx)
        lam['upper'] = np.NaN(nx)
    else:
        lam['lower'] = np.zeros(nx)
        lam['upper'] = np.zeros(nx)

    if len(f) == 0:
        f = np.NaN

    if len(pi) == 0:
        pi = np.NaN(len(bb))

    kl = find(rc > 0)  # lower bound binding
    ku = find(rc < 0)  # upper bound binding
    lam['lower'][kl] = rc[kl]
    lam['upper'][ku] = -rc[ku]
    lam['eqlin'] = pi[:neq + 1]
    lam['ineqlin'] = pi[neq + range(niq + 1)]
    mu_l = np.zeros(nA)
    mu_u = np.zeros(nA)

    # repackage lmbdas
    kl = find(lam['eqlin'] > 0)  # lower bound binding
    ku = find(lam['eqlin'] < 0)  # upper bound binding

    mu_l[ieq[kl]] = lam['eqlin'][kl]
    mu_l[igt] = -lam['ineqlin'][nlt + range(ngt + 1)]
    mu_l[ibx] = -lam['ineqlin'][nlt + ngt + nbx + range(nbx)]

    mu_u[ieq[ku]] = -lam['eqlin'][ku]
    mu_u[ilt] = -lam['ineqlin'][:nlt + 1]
    mu_u[ibx] = -lam['ineqlin'][nlt + ngt + range(nbx + 1)]

    lmbda = dict(mu_l=mu_l, mu_u=mu_u, lower=lam['lower'], upper=lam['upper'])

    return x, f, eflag, output, lmbda


def qps_ipopt(H, c, A, l, u, xmin, xmax, x0, opt):
    """
    Quadratic Program Solver based on IPOPT.

    Uses IPOPT to solve the following QP (quadratic programming) problem::

        min 1/2 x'*H*x + c'*x
         x

    subject to::

        l <= A*x <= u       (linear constraints)
        xmin <= x <= xmax   (variable bounds)

    Inputs (all optional except C{H}, C{C}, C{A} and C{L}):
        - C{H} : matrix (possibly sparse) of quadratic cost coefficients
        - C{C} : vector of linear cost coefficients
        - C{A, l, u} : define the optional linear constraints. Default
        values for the elements of C{l} and C{u} are -Inf and Inf,
        respectively.
        - C{xmin, xmax} : optional lower and upper bounds on the
        C{x} variables, defaults are -Inf and Inf, respectively.
        - C{x0} : optional starting value of optimization vector C{x}
        - C{opt} : optional options structure with the following fields,
        all of which are also optional (default values shown in parentheses)
            - C{verbose} (0) - controls level of progress output displayed
                - 0 = no progress output
                - 1 = some progress output
                - 2 = verbose progress output
            - C{max_it} (0) - maximum number of iterations allowed
                - 0 = use algorithm default
            - C{ipopt_opt} - options struct for IPOPT, values in
            C{verbose} and C{max_it} override these options
        - C{problem} : The inputs can alternatively be supplied in a single
        C{problem} dict with fields corresponding to the input arguments
        described above: C{H, c, A, l, u, xmin, xmax, x0, opt}

    Outputs:
        - C{x} : solution vector
        - C{f} : final objective function value
        - C{exitflag} : exit flag
            - 1 = first order optimality conditions satisfied
            - 0 = maximum number of iterations reached
            - -1 = numerically failed
        - C{output} : output struct with the following fields:
            - C{iterations} - number of iterations performed
            - C{hist} - dict list with trajectories of the following:
            C{feascond}, C{gradcond}, C{compcond}, C{costcond}, C{gamma},
            C{stepsize}, C{obj}, C{alphap}, C{alphad}
            - message - exit message
        - C{lmbda} : dict containing the Langrange and Kuhn-Tucker
        multipliers on the constraints, with fields:
            - C{mu_l} - lower (left-hand) limit on linear constraints
            - C{mu_u} - upper (right-hand) limit on linear constraints
            - C{lower} - lower bound on optimization variables
            - C{upper} - upper bound on optimization variables

    Calling syntax options::
        x, f, exitflag, output, lmbda = \
            qps_ipopt(H, c, A, l, u, xmin, xmax, x0, opt)

        x = qps_ipopt(H, c, A, l, u)
        x = qps_ipopt(H, c, A, l, u, xmin, xmax)
        x = qps_ipopt(H, c, A, l, u, xmin, xmax, x0)
        x = qps_ipopt(H, c, A, l, u, xmin, xmax, x0, opt)
        x = qps_ipopt(problem), where problem is a struct with fields:
                        H, c, A, l, u, xmin, xmax, x0, opt
                        all fields except 'c', 'A' and 'l' or 'u' are optional
        x = qps_ipopt(...)
        x, f = qps_ipopt(...)
        x, f, exitflag = qps_ipopt(...)
        x, f, exitflag, output = qps_ipopt(...)
        x, f, exitflag, output, lmbda = qps_ipopt(...)

    Example::
        H = [   1003.1  4.3     6.3     5.9;
                4.3     2.2     2.1     3.9;
                6.3     2.1     3.5     4.8;
                5.9     3.9     4.8     10  ]
        c = np.zeros((4, 1))
        A = [   1       1       1       1
                0.17    0.11    0.10    0.18    ]
        l = [1, 0.10]
        u = [1, Inf]
        xmin = np.zeros((4, 1))
        x0 = [1, 0, 0, 1]
        opt = {'verbose': 2)
        x, f, s, out, lambda = qps_ipopt(H, c, A, l, u, xmin, [], x0, opt)

    Problem from U{http://www.jmu.edu/docs/sasdoc/sashtml/iml/chap8/sect12.htm}

    @see: C{pyipopt}, L{ipopt_options}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # ----- input argument handling  -----
    # gather inputs
    if isinstance(H, dict):  # problem struct
        p = H
        if 'opt' in p:
            opt = p['opt']
        if 'x0' in p:
            x0 = p['x0']
        if 'xmax' in p:
            xmax = p['xmax']
        if 'xmin' in p:
            xmin = p['xmin']
        if 'u' in p:
            u = p['u']
        if 'l' in p:
            l = p['l']
        if 'A' in p:
            A = p['A']
        if 'c' in p:
            c = p['c']
        if 'H' in p:
            H = p['H']
    else:  # individual args
        assert H is not None
        assert c is not None
        assert A is not None
        assert l is not None

    if opt is None:
        opt = {}
#    if x0 is None:
#        x0 = np.array([])
#    if xmax is None:
#        xmax = np.array([])
#    if xmin is None:
#        xmin = np.array([])

    # define nx, set default values for missing optional inputs
    if len(H) == 0 or not np.any(np.any(H)):
        if len(A) == 0 and len(xmin) == 0 and len(xmax) == 0:
            logger.info('qps_ipopt: LP problem must include constraints or variable bounds')
        else:
            if len(A) > 0:
                nx = np.shape(A)[1]
            elif len(xmin) > 0:
                nx = len(xmin)
            else:    # if len(xmax) > 0
                nx = len(xmax)
        H = c_sparse((nx, nx))
    else:
        nx = np.shape(H)[0]

    if len(c) == 0:
        c = np.zeros(nx)

    if len(A) > 0 and (len(l) == 0 or all(l == -np.Inf)) and \
            (len(u) == 0 or all(u == np.Inf)):
        A = None  # no limits => no linear constraints

    nA = np.shape(A)[0]  # number of original linear constraints
    if nA:
        if len(u) == 0:  # By default, linear inequalities are ...
            u = np.Inf * np.ones(nA)  # ... unbounded above and ...

        if len(l) == 0:
            l = -np.Inf * np.ones(nA)  # ... unbounded below.

    if len(x0) == 0:
        x0 = np.zeros(nx)

    # default options
    if 'verbose' in opt:
        verbose = opt['verbose']
    else:
        verbose = 0

    if 'max_it' in opt:
        max_it = opt['max_it']
    else:
        max_it = 0

    # make sure args are sparse/full as expected by IPOPT
    if len(H) > 0:
        if not sp.issparse(H):
            H = c_sparse(H)

    if not sp.issparse(A):
        A = c_sparse(A)

    # -----  run optimization  -----
    # set options dict for IPOPT
    options = {}
    if 'ipopt_opt' in opt:
        options['ipopt'] = ipopt_options(opt['ipopt_opt'])
    else:
        options['ipopt'] = ipopt_options()

    options['ipopt']['jac_c_constant'] = 'yes'
    options['ipopt']['jac_d_constant'] = 'yes'
    options['ipopt']['hessian_constant'] = 'yes'
    options['ipopt']['least_square_init_primal'] = 'yes'
    options['ipopt']['least_square_init_duals'] = 'yes'
    # options['ipopt']['mehrotra_algorithm']        = 'yes'     ## default 'no'
    if verbose:
        options['ipopt']['print_level'] = min(12, verbose * 2 + 1)
    else:
        options['ipopt']['print_level = 0']

    if max_it:
        options['ipopt']['max_iter'] = max_it

    # define variable and constraint bounds, if given
    if nA:
        options['cu'] = u
        options['cl'] = l

    if len(xmin) > 0:
        options['lb'] = xmin

    if len(xmax) > 0:
        options['ub'] = xmax

    # assign function handles
    funcs = {}
    funcs['objective'] = lambda x: 0.5 * x.T * H * x + c.T * x
    funcs['gradient'] = lambda x: H * x + c
    funcs['constraints'] = lambda x: A * x
    funcs['jacobian'] = lambda x: A
    funcs['jacobianstructure'] = lambda: A
    funcs['hessian'] = lambda x, sigma, lmbda: np.tril(H)
    funcs['hessianstructure'] = lambda: np.tril(H)

    # run the optimization
    x, info = pyipopt(x0, funcs, options)

    if info['status'] == 0 | info['status'] == 1:
        eflag = 1
    else:
        eflag = 0

    output = {}
    if 'iter' in info:
        output['iterations'] = info['iter']

    output['info'] = info['status']
    f = funcs['objective'](x)

    # repackage lmbdas
    kl = find(info['lmbda'] < 0)  # lower bound binding
    ku = find(info['lmbda'] > 0)  # upper bound binding
    mu_l = np.zeros(nA)
    mu_l[kl] = -info['lmbda'][kl]
    mu_u = np.zeros(nA)
    mu_u[ku] = info['lmbda'][ku]

    lmbda = dict(mu_l=mu_l, mu_u=mu_u, lower=info['zl'], upper=info['zu'])

    return x, f, eflag, output, lmbda


@require_mosek
def qps_mosek(H, c=None, A=None, l=None, u=None, xmin=None, xmax=None,
              x0=None, opt=None):
    """
    Quadratic Program Solver based on MOSEK.

    A wrapper function providing a PYPOWER standardized interface for using
    MOSEKOPT to solve the following QP (quadratic programming) problem::

        min 1/2 x'*H*x + c'*x
         x

    subject to::

        l <= A*x <= u       (linear constraints)
        xmin <= x <= xmax   (variable bounds)

    Inputs (all optional except C{H}, C{C}, C{A} and C{L}):
        - C{H} : matrix (possibly sparse) of quadratic cost coefficients
        - C{C} : vector of linear cost coefficients
        - C{A, l, u} : define the optional linear constraints. Default
        values for the elements of L and U are -Inf and Inf, respectively.
        - xmin, xmax : optional lower and upper bounds on the
        C{x} variables, defaults are -Inf and Inf, respectively.
        - C{x0} : optional starting value of optimization vector C{x}
        - C{opt} : optional options structure with the following fields,
        all of which are also optional (default values shown in parentheses)
            - C{verbose} (0) - controls level of progress output displayed
                - 0 = no progress output
                - 1 = some progress output
                - 2 = verbose progress output
            - C{max_it} (0) - maximum number of iterations allowed
                - 0 = use algorithm default
            - C{mosek_opt} - options struct for MOSEK, values in
            C{verbose} and C{max_it} override these options
        - C{problem} : The inputs can alternatively be supplied in a single
        C{problem} struct with fields corresponding to the input arguments
        described above: C{H, c, A, l, u, xmin, xmax, x0, opt}

    Outputs:
        - C{x} : solution vector
        - C{f} : final objective function value
        - C{exitflag} : exit flag
              - 1 = success
              - 0 = terminated at maximum number of iterations
              - -1 = primal or dual infeasible
              < 0 = the negative of the MOSEK return code
        - C{output} : output dict with the following fields:
            - C{r} - MOSEK return code
            - C{res} - MOSEK result dict
        - C{lmbda} : dict containing the Langrange and Kuhn-Tucker
        multipliers on the constraints, with fields:
            - C{mu_l} - lower (left-hand) limit on linear constraints
            - C{mu_u} - upper (right-hand) limit on linear constraints
            - C{lower} - lower bound on optimization variables
            - C{upper} - upper bound on optimization variables

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # ----- input argument handling  -----
    # gather inputs
    if isinstance(H, dict):  # problem struct
        p = H
    else:  # individual args
        p = {'H': H, 'c': c, 'A': A, 'l': l, 'u': u}
        if xmin is not None:
            p['xmin'] = xmin
        if xmax is not None:
            p['xmax'] = xmax
        if x0 is not None:
            p['x0'] = x0
        if opt is not None:
            p['opt'] = opt

    # define nx, set default values for H and c
    if 'H' not in p or len(p['H']) or not np.any(np.any(p['H'])):
        if ('A' not in p) | len(p['A']) == 0 & \
                ('xmin' not in p) | len(p['xmin']) == 0 & \
                ('xmax' not in p) | len(p['xmax']) == 0:
            logger.debug('qps_mosek: LP problem must include constraints or variable bounds\n')
        else:
            if 'A' in p & len(p['A']) > 0:
                nx = np.shape(p['A'])[1]
            elif 'xmin' in p & len(p['xmin']) > 0:
                nx = len(p['xmin'])
            else:    # if isfield(p, 'xmax') && ~isempty(p.xmax)
                nx = len(p['xmax'])
        p['H'] = c_sparse((nx, nx))
        qp = 0
    else:
        nx = np.shape(p['H'])[0]
        qp = 1

    if 'c' not in p | len(p['c']) == 0:
        p['c'] = np.zeros(nx)

    if 'x0' not in p | len(p['x0']) == 0:
        p['x0'] = np.zeros(nx)

    # default options
    if 'opt' not in p:
        p['opt'] = []

    if 'verbose' in p['opt']:
        verbose = p['opt']['verbose']
    else:
        verbose = 0

    if 'max_it' in p['opt']:
        max_it = p['opt']['max_it']
    else:
        max_it = 0

    if 'mosek_opt' in p['opt']:
        mosek_opt = mosek_options(p['opt']['mosek_opt'])
    else:
        mosek_opt = mosek_options()

    if max_it:
        mosek_opt['MSK_IPAR_INTPNT_MAX_ITERATIONS'] = max_it

    if qp:
        mosek_opt['MSK_IPAR_OPTIMIZER'] = 0  # default solver only for QP

    # set up problem struct for MOSEK
    prob = {}
    prob['c'] = p['c']
    if qp:
        prob['qosubi'], prob['qosubj'], prob['qoval'] = find(sp.tril(c_sparse(p['H'])))

    if 'A' in p & len(p['A']) > 0:
        prob['a'] = c_sparse(p['A'])

    if 'l' in p & len(p['A']) > 0:
        prob['blc'] = p['l']

    if 'u' in p & len(p['A']) > 0:
        prob['buc'] = p['u']

    if 'xmin' in p & len(p['xmin']) > 0:
        prob['blx'] = p['xmin']

    if 'xmax' in p & len(p['xmax']) > 0:
        prob['bux'] = p['xmax']

    # A is not allowed to be empty
    if 'a' not in prob | len(prob['a']) == 0:
        unconstrained = True
        prob['a'] = c_sparse((1, (1, 1)), (1, nx))
        prob.blc = -np.Inf
        prob.buc = np.Inf
    else:
        unconstrained = False

    # -----  run optimization  -----
    if verbose:
        methods = [
            'default',
            'interior point',
            '<default>',
            '<default>',
            'primal simplex',
            'dual simplex',
            'primal dual simplex',
            'automatic simplex',
            '<default>',
            '<default>',
            'concurrent'
        ]
        if len(H) == 0 or not np.any(np.any(H)):
            lpqp = 'LP'
        else:
            lpqp = 'QP'

        # (this code is also in mpver.m)
        # MOSEK Version 6.0.0.93 (Build date: 2010-10-26 13:03:27)
        # MOSEK Version 6.0.0.106 (Build date: 2011-3-17 10:46:54)
#        pat = 'Version (\.*\d)+.*Build date: (\d\d\d\d-\d\d-\d\d)';
        pat = 'Version (\.*\d)+.*Build date: (\d+-\d+-\d+)'
        s, e, tE, m, t = re.compile(eval('mosekopt'), pat)
        if len(t) == 0:
            vn = '<unknown>'
        else:
            vn = t[0][0]

        logger.info('MOSEK Version %s -- %s %s solver\n' %
                    (vn, methods[mosek_opt['MSK_IPAR_OPTIMIZER'] + 1], lpqp))

    cmd = 'minimize echo(%d)' % verbose
    r, res = mosekopt(cmd, prob, mosek_opt)

    # -----  repackage results  -----
    if 'sol' in res:
        if 'bas' in res['sol']:
            sol = res['sol.bas']
        else:
            sol = res['sol.itr']
        x = sol['xx']
    else:
        sol = np.array([])
        x = np.array([])

    # -----  process return codes  -----
    if 'symbcon' in res:
        sc = res['symbcon']
    else:
        r2, res2 = mosekopt('symbcon echo(0)')
        sc = res2['symbcon']

    eflag = -r
    msg = ''
    if r == sc.MSK_RES_OK:
        if len(sol) > 0:
            #            if sol['solsta'] == sc.MSK_SOL_STA_OPTIMAL:
            if sol['solsta'] == 'OPTIMAL':
                msg = 'The solution is optimal.'
                eflag = 1
            else:
                eflag = -1
#                if sol['prosta'] == sc['MSK_PRO_STA_PRIM_INFEAS']:
                if sol['prosta'] == 'PRIMAL_INFEASIBLE':
                    msg = 'The problem is primal infeasible.'
#                elif sol['prosta'] == sc['MSK_PRO_STA_DUAL_INFEAS']:
                elif sol['prosta'] == 'DUAL_INFEASIBLE':
                    msg = 'The problem is dual infeasible.'
                else:
                    msg = sol['solsta']

    elif r == sc['MSK_RES_TRM_MAX_ITERATIONS']:
        eflag = 0
        msg = 'The optimizer terminated at the maximum number of iterations.'
    else:
        if 'rmsg' in res and 'rcodestr' in res:
            msg = '%s : %s' % (res['rcodestr'], res['rmsg'])
        else:
            msg = 'MOSEK return code = %d' % r

    # always alert user if license is expired
    if (verbose or r == 1001) and len(msg) < 0:
        logger.info('%s\n' % msg)

    # -----  repackage results  -----
    if r == 0:
        f = p['c'].T * x
        if len(p['H']) > 0:
            f = 0.5 * x.T * p['H'] * x + f
    else:
        f = np.array([])

    output = {}
    output['r'] = r
    output['res'] = res

    if 'sol' in res:
        lmbda = {}
        lmbda['lower'] = sol['slx']
        lmbda['upper'] = sol['sux']
        lmbda['mu_l'] = sol['slc']
        lmbda['mu_u'] = sol['suc']
        if unconstrained:
            lmbda['mu_l'] = np.array([])
            lmbda['mu_u'] = np.array([])
    else:
        lmbda = np.array([])

    return x, f, eflag, output, lmbda


def qps_pips(H, c, A, l, u, xmin=None, xmax=None, x0=None, opt=None):
    """
    Uses the Python Interior Point Solver (PIPS) to solve the following
    QP (quadratic programming) problem::

            min 1/2 x'*H*x + C'*x
             x

    subject to::

            l <= A*x <= u       (linear constraints)
            xmin <= x <= xmax   (variable bounds)

    Note the calling syntax is almost identical to that of QUADPROG from
    MathWorks' Optimization Toolbox. The main difference is that the linear
    constraints are specified with C{A}, C{L}, C{U} instead of C{A}, C{B},
    C{Aeq}, C{Beq}.

    Example from U{http://www.uc.edu/sashtml/iml/chap8/sect12.htm}:

        >>> from numpy import array, zeros, Inf
        >>> from scipy.sparse import csr_matrix
        >>> H = csr_matrix(np.array([[1003.1,  4.3,     6.3,     5.9],
        ...                       [4.3,     2.2,     2.1,     3.9],
        ...                       [6.3,     2.1,     3.5,     4.8],
        ...                       [5.9,     3.9,     4.8,     10 ]]))
        >>> c = np.zeros(4)
        >>> A = csr_matrix(np.array([[1,       1,       1,       1   ],
        ...                       [0.17,    0.11,    0.10,    0.18]]))
        >>> l = np.array([1, 0.10])
        >>> u = np.array([1, Inf])
        >>> xmin = np.zeros(4)
        >>> xmax = None
        >>> x0 = np.array([1, 0, 0, 1])
        >>> solution = qps_pips(H, c, A, l, u, xmin, xmax, x0)
        >>> round(solution["f"], 11) == 1.09666678128
        True
        >>> solution["converged"]
        True
        >>> solution["output"]["iterations"]
        10

    All parameters are optional except C{H}, C{c}, C{A} and C{l} or C{u}.
    @param H: Quadratic cost coefficients.
    @type H: csr_matrix
    @param c: vector of linear cost coefficients
    @type c: array
    @param A: Optional linear constraints.
    @type A: csr_matrix
    @param l: Optional linear constraints. Default values are M{-Inf}.
    @type l: array
    @param u: Optional linear constraints. Default values are M{Inf}.
    @type u: array
    @param xmin: Optional lower bounds on the M{x} variables, defaults are
                 M{-Inf}.
    @type xmin: array
    @param xmax: Optional upper bounds on the M{x} variables, defaults are
                 M{Inf}.
    @type xmax: array
    @param x0: Starting value of optimization vector M{x}.
    @type x0: array
    @param opt: optional options dictionary with the following keys, all of
                which are also optional (default values shown in parentheses)
                  - C{verbose} (False) - Controls level of progress output
                    displayed
                  - C{feastol} (1e-6) - termination tolerance for feasibility
                    condition
                  - C{gradtol} (1e-6) - termination tolerance for gradient
                    condition
                  - C{comptol} (1e-6) - termination tolerance for
                    complementarity condition
                  - C{costtol} (1e-6) - termination tolerance for cost
                    condition
                  - C{max_it} (150) - maximum number of iterations
                  - C{step_control} (False) - set to True to enable step-size
                    control
                  - C{max_red} (20) - maximum number of step-size reductions if
                    step-control is on
                  - C{cost_mult} (1.0) - cost multiplier used to scale the
                    objective function for improved conditioning. Note: The
                    same value must also be passed to the Hessian evaluation
                    function so that it can appropriately scale the objective
                    function term in the Hessian of the Lagrangian.
    @type opt: dict

    @rtype: dict
    @return: The solution dictionary has the following keys:
               - C{x} - solution vector
               - C{f} - final objective function value
               - C{converged} - exit status
                   - True = first order optimality conditions satisfied
                   - False = maximum number of iterations reached
                   - None = numerically failed
               - C{output} - output dictionary with keys:
                   - C{iterations} - number of iterations performed
                   - C{hist} - dictionary of arrays with trajectories of the
                     following: feascond, gradcond, coppcond, costcond, gamma,
                     stepsize, obj, alphap, alphad
                   - C{message} - exit message
               - C{lmbda} - dictionary containing the Langrange and Kuhn-Tucker
                 multipliers on the constraints, with keys:
                   - C{eqnonlin} - nonlinear equality constraints
                   - C{ineqnonlin} - nonlinear inequality constraints
                   - C{mu_l} - lower (left-hand) limit on linear constraints
                   - C{mu_u} - upper (right-hand) limit on linear constraints
                   - C{lower} - lower bound on optimization variables
                   - C{upper} - upper bound on optimization variables

    @see: L{pips}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if isinstance(H, dict):
        p = H
    else:
        p = {'H': H, 'c': c, 'A': A, 'l': l, 'u': u}
        if xmin is not None:
            p['xmin'] = xmin
        if xmax is not None:
            p['xmax'] = xmax
        if x0 is not None:
            p['x0'] = x0
        if opt is not None:
            p['opt'] = opt

    if 'H' not in p or p['H'] == None:  # p['H'].nnz == 0:
        if p['A'] is None or p['A'].nnz == 0 and \
           'xmin' not in p and \
           'xmax' not in p:
            #           'xmin' not in p or len(p['xmin']) == 0 and \
            #           'xmax' not in p or len(p['xmax']) == 0:
            print('qps_pips: LP problem must include constraints or variable bounds')
            return
        else:
            if p['A'] is not None and p['A'].nnz >= 0:
                nx = p['A'].shape[1]
            elif 'xmin' in p and len(p['xmin']) > 0:
                nx = p['xmin'].shape[0]
            elif 'xmax' in p and len(p['xmax']) > 0:
                nx = p['xmax'].shape[0]
        p['H'] = c_sparse((nx, nx))
    else:
        nx = p['H'].shape[0]

    p['xmin'] = -np.Inf * np.ones(nx) if 'xmin' not in p else p['xmin']
    p['xmax'] = np.Inf * np.ones(nx) if 'xmax' not in p else p['xmax']

    p['c'] = np.zeros(nx) if p['c'] is None else p['c']

    p['x0'] = np.zeros(nx) if 'x0' not in p else p['x0']

    def qp_f(x, return_hessian=False):
        f = 0.5 * np.dot(x * p['H'], x) + np.dot(p['c'], x)
        df = p['H'] * x + p['c']
        if not return_hessian:
            return f, df
        d2f = p['H']
        return f, df, d2f

    p['f_fcn'] = qp_f

    sol = pips(p)

    return sol["x"], sol["f"], sol["eflag"], sol["output"], sol["lmbda"]


def qps_pypower(H, c=None, A=None, l=None, u=None, xmin=None, xmax=None,
                x0=None, opt=None):
    """
    Quadratic Program Solver for PYPOWER.

    A common wrapper function for various QP solvers.
    Solves the following QP (quadratic programming) problem::

        min 1/2 x'*H*x + c'*x
         x

    subject to::

        l <= A*x <= u       (linear constraints)
        xmin <= x <= xmax   (variable bounds)

    Inputs (all optional except C{H}, C{c}, C{A} and C{l}):
        - C{H} : matrix (possibly sparse) of quadratic cost coefficients
        - C{c} : vector of linear cost coefficients
        - C{A, l, u} : define the optional linear constraints. Default
        values for the elements of C{l} and C{u} are -Inf and Inf,
        respectively.
        - C{xmin}, C{xmax} : optional lower and upper bounds on the
        C{x} variables, defaults are -Inf and Inf, respectively.
        - C{x0} : optional starting value of optimization vector C{x}
        - C{opt} : optional options structure with the following fields,
        all of which are also optional (default values shown in parentheses)
            - C{alg} (0) - determines which solver to use
                -   0 = automatic, first available of BPMPD_MEX, CPLEX,
                        Gurobi, PIPS
                - 100 = BPMPD_MEX
                - 200 = PIPS, Python Interior Point Solver
                pure Python implementation of a primal-dual
                interior point method
                - 250 = PIPS-sc, a step controlled variant of PIPS
                - 300 = Optimization Toolbox, QUADPROG or LINPROG
                - 400 = IPOPT
                - 500 = CPLEX
                - 600 = MOSEK
                - 700 = Gurobi
            - C{verbose} (0) - controls level of progress output displayed
                - 0 = no progress output
                - 1 = some progress output
                - 2 = verbose progress output
            - C{max_it} (0) - maximum number of iterations allowed
                - 0 = use algorithm default
            - C{bp_opt} - options vector for BP
            - C{cplex_opt} - options dict for CPLEX
            - C{grb_opt}   - options dict for gurobipy
            - C{ipopt_opt} - options dict for IPOPT
            - C{pips_opt}  - options dict for L{qps_pips}
            - C{mosek_opt} - options dict for MOSEK
            - C{ot_opt}    - options dict for QUADPROG/LINPROG
        - C{problem} : The inputs can alternatively be supplied in a single
        C{problem} dict with fields corresponding to the input arguments
        described above: C{H, c, A, l, u, xmin, xmax, x0, opt}

    Outputs:
        - C{x} : solution vector
        - C{f} : final objective function value
        - C{exitflag} : exit flag
            - 1 = converged
            - 0 or negative values = algorithm specific failure codes
        - C{output} : output struct with the following fields:
            - C{alg} - algorithm code of solver used
            - (others) - algorithm specific fields
        - C{lmbda} : dict containing the Langrange and Kuhn-Tucker
        multipliers on the constraints, with fields:
            - C{mu_l} - lower (left-hand) limit on linear constraints
            - C{mu_u} - upper (right-hand) limit on linear constraints
            - C{lower} - lower bound on optimization variables
            - C{upper} - upper bound on optimization variables


    Example from U{http://www.uc.edu/sashtml/iml/chap8/sect12.htm}:

        >>> from numpy import array, zeros, Inf
        >>> from scipy.sparse import csr_matrix
        >>> H = csr_matrix(np.array([[1003.1,  4.3,     6.3,     5.9],
        ...                       [4.3,     2.2,     2.1,     3.9],
        ...                       [6.3,     2.1,     3.5,     4.8],
        ...                       [5.9,     3.9,     4.8,     10 ]]))
        >>> c = np.zeros(4)
        >>> A = csr_matrix(np.array([[1,       1,       1,       1   ],
        ...                       [0.17,    0.11,    0.10,    0.18]]))
        >>> l = np.array([1, 0.10])
        >>> u = np.array([1, Inf])
        >>> xmin = np.zeros(4)
        >>> xmax = None
        >>> x0 = np.array([1, 0, 0, 1])
        >>> solution = qps_pips(H, c, A, l, u, xmin, xmax, x0)
        >>> round(solution["f"], 11) == 1.09666678128
        True
        >>> solution["converged"]
        True
        >>> solution["output"]["iterations"]
        10

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # ----- input argument handling  -----
    # gather inputs
    if isinstance(H, dict):  # problem struct
        p = H
        if 'opt' in p:
            opt = p['opt']
        if 'x0' in p:
            x0 = p['x0']
        if 'xmax' in p:
            xmax = p['xmax']
        if 'xmin' in p:
            xmin = p['xmin']
        if 'u' in p:
            u = p['u']
        if 'l' in p:
            l = p['l']
        if 'A' in p:
            A = p['A']
        if 'c' in p:
            c = p['c']
        if 'H' in p:
            H = p['H']
    else:  # individual args
        #        assert H is not None  zero dimensional sparse matrices not supported
        assert c is not None
#        assert A is not None  zero dimensional sparse matrices not supported
#        assert l is not None  no lower bounds indicated by None

    if opt is None:
        opt = {}
#    if x0 is None:
#        x0 = np.array([])
#    if xmax is None:
#        xmax = np.array([])
#    if xmin is None:
#        xmin = np.array([])

    # default options
    if 'alg' in opt:
        alg = opt['alg']
    else:
        alg = 0

    if 'verbose' in opt:
        verbose = opt['verbose']
    else:
        verbose = 0

    if alg == 0:
        if putils.have_fcn('cplex'):  # use CPLEX by default, if available
            alg = 500
        elif putils.have_fcn('mosek'):  # if not, then MOSEK, if available
            alg = 600
        elif putils.have_fcn('gurobipy'):  # if not, then Gurobi, if available
            alg = 700
        else:  # otherwise PIPS
            alg = 200

    # ----- call the appropriate solver  -----
    if alg == 200 or alg == 250:  # use MIPS or sc-MIPS
        # set up options
        if 'pips_opt' in opt:
            pips_opt = opt['pips_opt']
        else:
            pips_opt = {}

        if 'max_it' in opt:
            pips_opt['max_it'] = opt['max_it']

        if alg == 200:
            pips_opt['step_control'] = False
        else:
            pips_opt['step_control'] = True

        pips_opt['verbose'] = verbose

        # call solver
        x, f, eflag, output, lmbda = \
            qps_pips(H, c, A, l, u, xmin, xmax, x0, pips_opt)
    elif alg == 400:  # use IPOPT
        x, f, eflag, output, lmbda = \
            qps_ipopt(H, c, A, l, u, xmin, xmax, x0, opt)
    elif alg == 500:  # use CPLEX
        x, f, eflag, output, lmbda = \
            qps_cplex(H, c, A, l, u, xmin, xmax, x0, opt)
    elif alg == 600:  # use MOSEK
        x, f, eflag, output, lmbda = \
            qps_mosek(H, c, A, l, u, xmin, xmax, x0, opt)
    elif 700:  # use Gurobi
        x, f, eflag, output, lmbda = \
            qps_gurobi(H, c, A, l, u, xmin, xmax, x0, opt)
    else:
        logger.info('qps_pypower: %d is not a valid algorithm code', alg)

    if 'alg' not in output:
        output['alg'] = alg

    return x, f, eflag, output, lmbda


def cplex_options(overrides=None, ppopt=None):
    """
    Sets options for CPLEX.

    Sets the values for the options dict normally passed to
    C{cplexoptimset}.

    Inputs are all optional, second argument must be either a string
    (C{fname}) or a dict (C{ppopt}):

    Output is an options dict to pass to C{cplexoptimset}.

    Example:

    If C{ppopt['CPLEX_OPT'] = 3}, then after setting the default CPLEX options,
    CPLEX_OPTIONS will execute the following user-defined function
    to allow option overrides::

        opt = cplex_user_options_3(opt, ppopt)

    The contents of cplex_user_options_3.py, could be something like::

        def cplex_user_options_3(opt, ppopt):
            opt = {}
            opt['threads']          = 2
            opt['simplex']['refactor'] = 1
            opt['timelimit']        = 10000
            return opt

    For details on the available options, see the I{"Parameters Reference
    Manual"} section of the CPLEX documentation at:
    U{http://publib.boulder.ibm.com/infocenter/cosinfoc/v12r2/}

    @param overrides:
      - dict containing values to override the defaults
      - fname: name of user-supplied function called after default
        options are set to modify them. Calling syntax is::

            modified_opt = fname(default_opt)

    @param ppopt: PYPOWER options vector, uses the following entries:
      - OPF_VIOLATION - used to set opt.simplex.tolerances.feasibility
      - VERBOSE - used to set opt.barrier.display,
        opt.conflict.display, opt.mip.display, opt.sifting.display,
        opt.simplex.display, opt.tune.display
      - CPLEX_LPMETHOD - used to set opt.lpmethod
      - CPLEX_QPMETHOD - used to set opt.qpmethod
      - CPLEX_OPT      - user option file, if ppopt['CPLEX_OPT'] is
        non-zero it is appended to 'cplex_user_options_' to form
        the name of a user-supplied function used as C{fname}
        described above, except with calling syntax::

            modified_opt = fname(default_opt, ppopt)

    @see: C{cplexlp}, C{cplexqp}, L{ppoption}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # -----  initialization and arg handling  -----
    # defaults
    verbose = 1
    feastol = 1e-6
    fname = ''

    # second argument
    if ppopt != None:
        if isinstance(ppopt, str):  # 2nd arg is FNAME (string)
            fname = ppopt
            have_ppopt = False
        else:  # 2nd arg is ppopt (MATPOWER options vector)
            have_ppopt = True
            # (make default OPF_VIOLATION correspond to default CPLEX feastol)
            feastol = ppopt['OPF_VIOLATION'] / 5
            verbose = ppopt['VERBOSE']
            lpmethod = ppopt['CPLEX_LPMETHOD']
            qpmethod = ppopt['CPLEX_QPMETHOD']
            if ppopt['CPLEX_OPT']:
                fname = 'cplex_user_options_#d' % ppopt['CPLEX_OPT']
    else:
        have_ppopt = False

    # -----  set default options for CPLEX  -----
    opt = cplexoptimset('cplex')
    opt['simplex']['tolerances']['feasibility'] = feastol

    # printing
    vrb = max([0, verbose - 1])
    opt['barrier']['display'] = vrb
    opt['conflict']['display'] = vrb
    opt['mip']['display'] = vrb
    opt['sifting']['display'] = vrb
    opt['simplex']['display'] = vrb
    opt['tune']['display'] = vrb

    # solution algorithm
    if have_ppopt:
        opt['lpmethod'] = lpmethod
        opt['qpmethod'] = qpmethod
    # else:
    #    opt['lpmethod'] = 2
    #    opt['qpmethod'] = 2

    # -----  call user function to modify defaults  -----
    if len(fname) > 0:
        if have_ppopt:
            opt = putils.feval(fname, opt, ppopt)
        else:
            opt = putils.feval(fname, opt)

    # -----  apply overrides  -----
    if overrides is not None:
        names = overrides.keys()
        for k in range(len(names)):
            if isinstance(overrides[names[k]], dict):
                names2 = overrides[names[k]].keys()
                for k2 in range(len(names2)):
                    if isinstance(overrides[names[k]][names2[k2]], dict):
                        names3 = overrides[names[k]][names2[k2]].keys()
                        for k3 in range(len(names3)):
                            opt[names[k]][names2[k2]][names3[k3]] = overrides[names[k]][names2[k2]][names3[k3]]
                    else:
                        opt[names[k]][names2[k2]] = overrides[names[k]][names2[k2]]
            else:
                opt[names[k]] = overrides[names[k]]

    return opt


def gurobi_options(overrides=None, ppopt=None):
    """
    Sets options for GUROBI.

    Sets the values for the options dict normally passed to GUROBI_MEX.

    Inputs are all optional, second argument must be either a string
    (fname) or a vector (ppopt):

        overrides - dict containing values to override the defaults
        fname - name of user-supplied function called after default
            options are set to modify them. Calling syntax is:
                modified_opt = fname(default_opt)
        ppopt - PYPOWER options vector, uses the following entries:
            OPF_VIOLATION (16)  - used to set opt.FeasibilityTol
            VERBOSE (31)        - used to set opt.DisplayInterval, opt.Display
            GRB_METHOD (121)    - used to set opt.Method
            GRB_TIMELIMIT (122) - used to set opt.TimeLimit (seconds)
            GRB_THREADS (123)   - used to set opt.Threads
            GRB_OPT (124)       - user option file, if PPOPT(124) is non-zero
                it is appended to 'gurobi_user_options_' to form the name of a
                user-supplied function used as C{fname} described above, except
                with calling syntax:
                    modified_opt = fname(default_opt, mpopt)

    Output is an options struct to pass to GUROBI_MEX.

    Example:

    If ppopt['GRB_OPT'] = 3, then after setting the default GUROBI options,
    GUROBI_OPTIONS will execute the following user-defined function
    to allow option overrides:

        opt = gurobi_user_options_3(opt, ppopt)

    The contents of gurobi_user_options_3.py, could be something like:

        def gurobi_user_options_3(opt, ppopt):
            opt = {}
            opt['OptimalityTol']   = 1e-9
            opt['IterationLimit']  = 3000
            opt['BarIterLimit']    = 200
            opt['Crossover']       = 0
            opt['Presolve']        = 0
            return opt

    For details on the available options, see the "Parameters" section
    of the "Gurobi Optimizer Reference Manual" at:

        http://www.gurobi.com/doc/45/refman/

    @see: L{gurobi_mex}, L{ppoption}.
    """
    # -----  initialization and arg handling  -----
    # defaults
    verbose = True
    fname = ''

    # second argument
    if ppopt != None:
        if isinstance(ppopt, str):  # 2nd arg is FNAME (string)
            fname = ppopt
            have_ppopt = False
        else:  # 2nd arg is MPOPT (MATPOWER options vector)
            have_ppopt = True
            verbose = ppopt['VERBOSE']
            if ppopt['GRB_OPT']:
                fname = 'gurobi_user_options_%d', ppopt['GRB_OPT']
    else:
        have_ppopt = False

    # -----  set default options for CPLEX  -----
    opt = {}
    # opt['OptimalityTol'] = 1e-6
    # -1 - auto, 0 - no, 1 - conserv, 2 - aggressive=
    # opt['Presolve'] = -1
    # opt['LogFile'] = 'qps_gurobi.log'
    if have_ppopt:
        # (make default OPF_VIOLATION correspond to default FeasibilityTol)
        opt['FeasibilityTol'] = ppopt['OPF_VIOLATION'] / 5
        opt['Method'] = ppopt['GRB_METHOD']
        opt['TimeLimit'] = ppopt['GRB_TIMELIMIT']
        opt['Threads'] = ppopt['GRB_THREADS']
    else:
        opt['Method'] = 1  # dual simplex

    opt['Display'] = min(verbose, 3)
    if verbose:
        opt['DisplayInterval'] = 1
    else:
        opt['DisplayInterval'] = np.Inf

    # -----  call user function to modify defaults  -----
    if len(fname) > 0:
        if have_ppopt:
            opt = putils.feval(fname, opt, ppopt)
        else:
            opt = putils.feval(fname, opt)

    # -----  apply overrides  -----
    if overrides is not None:
        names = overrides.keys()
        for k in range(len(names)):
            opt[names[k]] = overrides[names[k]]

    return opt


def ipopt_options(overrides=None, ppopt=None):
    """
    Sets options for IPOPT.

    Sets the values for the options.ipopt dict normally passed to
    IPOPT.

    Inputs are all optional, second argument must be either a string
    (C{fname}) or a dict (C{ppopt}):

        - C{overrides}
            - dict containing values to override the defaults
            - C{fname} name of user-supplied function called after default
            options are set to modify them. Calling syntax is::
                modified_opt = fname(default_opt)
        - C{ppopt} PYPOWER options vector, uses the following entries:
            - C{OPF_VIOLATION} used to set opt['constr_viol_tol']
            - C{VERBOSE}       used to opt['print_level']
            - C{IPOPT_OPT}     user option file, if ppopt['IPOPT_OPT'] is
            non-zero it is appended to 'ipopt_user_options_' to form
            the name of a user-supplied function used as C{fname}
            described above, except with calling syntax::
                modified_opt = fname(default_opt ppopt)

    Output is an options.ipopt dict to pass to IPOPT.

    Example: If ppopt['IPOPT_OPT'] = 3, then after setting the default IPOPT
    options, L{ipopt_options} will execute the following user-defined function
    to allow option overrides::

        opt = ipopt_user_options_3(opt, ppopt);

    The contents of ipopt_user_options_3.py, could be something like::

        def ipopt_user_options_3(opt, ppopt):
            opt = {}
            opt['nlp_scaling_method'] = 'none'
            opt['max_iter']           = 500
            opt['derivative_test']    = 'first-order'
            return opt

    See the options reference section in the IPOPT documentation for
    details on the available options.

    U{http://www.coin-or.org/Ipopt/documentation/}

    @see: C{pyipopt}, L{ppoption}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # -----  initialization and arg handling  -----
    # defaults
    verbose = 2
    fname = ''

    # second argument
    if ppopt != None:
        if isinstance(ppopt, str):  # 2nd arg is FNAME (string)
            fname = ppopt
            have_ppopt = False
        else:  # 2nd arg is ppopt (MATPOWER options vector)
            have_ppopt = True
            verbose = ppopt['VERBOSE']
            if ppopt['IPOPT_OPT']:
                fname = 'ipopt_user_options_#d' % ppopt['IPOPT_OPT']
    else:
        have_ppopt = False

    opt = {}
    # -----  set default options for IPOPT  -----
    # printing
    if verbose:
        opt['print_level'] = min([12, verbose * 2 + 1])
    else:
        opt['print_level'] = 0

    # convergence
    opt['tol'] = 1e-8  # default 1e-8
    opt['max_iter'] = 250  # default 3000
    opt['dual_inf_tol'] = 0.1  # default 1
    if have_ppopt:
        opt['constr_viol_tol'] = ppopt[16]  # default 1e-4
        opt['acceptable_constr_viol_tol'] = ppopt[16] * 100  # default 1e-2
    opt['compl_inf_tol'] = 1e-5  # default 1e-4
    opt['acceptable_tol'] = 1e-8  # default 1e-6
    # opt['acceptable_iter'] = 15                   ## default 15
    # opt['acceptable_dual_inf_tol']     = 1e+10    ## default 1e+10
    opt['acceptable_compl_inf_tol'] = 1e-3  # default 1e-2
    # opt['acceptable_obj_change_tol']   = 1e+20    ## default 1e+20
    # opt['diverging_iterates_tol']      = 1e+20    ## default 1e+20

    # NLP scaling
    # opt['nlp_scaling_method']  = 'none'           ## default 'gradient-based'

    # NLP
    # opt['fixed_variable_treatment']    = 'make_constraint'    ## default 'make_parameter'
    # opt['honor_original_bounds']       = 'no'                 ## default 'yes'
    # opt['check_derivatives_for_naninf'] = 'yes'               ## default 'no'

    # initialization
    # opt['least_square_init_primal']    = 'yes'        ## default 'no'
    # opt['least_square_init_duals']     = 'yes'        ## default 'no'

    # barrier parameter update
    opt['mu_strategy'] = 'adaptive'  # default 'monotone'

    # linear solver
    # opt['linear_solver']   = 'ma27'
    # opt['linear_solver']   = 'ma57'
    # opt['linear_solver']   = 'pardiso'
    # opt['linear_solver']   = 'wsmp'
    # opt['linear_solver']   = 'mumps'          ## default 'mumps'
    # opt['linear_solver']   = 'custom'
    # opt['linear_scaling_on_demand']    = 'no' ## default 'yes'

    # step calculation
    # opt['mehrotra_algorithm']      = 'yes'    ## default 'no'
    # opt['fast_step_computation']   = 'yes'    ## default 'no'

    # restoration phase
    # opt['expect_infeasible_problem']   = 'yes'    ## default 'no'

    # derivative checker
    # opt['derivative_test']         = 'second-order'   ## default 'none'

    # hessian approximation
    # opt['hessian_approximation']   = 'limited-memory' ## default 'exact'

    # ma57 options
    # opt['ma57_pre_alloc'] = 3
    # opt['ma57_pivot_order'] = 4

    # -----  call user function to modify defaults  -----
    if len(fname) > 0:
        if have_ppopt:
            opt = putils.feval(fname, opt, ppopt)
        else:
            opt = putils.feval(fname, opt)

    # -----  apply overrides  -----
    if overrides is not None:
        names = overrides.keys()
        for k in range(len(names)):
            opt[names[k]] = overrides[names[k]]

    return opt


def mosek_options(overrides=None, ppopt=None):
    """Sets options for MOSEK.

    Inputs are all optional, second argument must be either a string
    (C{fname}) or a dict (C{ppopt}):

        - C{overrides}
            - dict containing values to override the defaults
            - C{fname} name of user-supplied function called after default
            options are set to modify them. Calling syntax is::
                modified_opt = fname(default_opt)
        - C{ppopt} PYPOWER options vector, uses the following entries:
            - C{OPF_VIOLATION} used to set opt.MSK_DPAR_INTPNT_TOL_PFEAS
            - C{VERBOSE} not currently used here
            - C{MOSEK_LP_ALG} - used to set opt.MSK_IPAR_OPTIMIZER
            - C{MOSEK_MAX_IT} used to set opt.MSK_IPAR_INTPNT_MAX_ITERATIONS
            - C{MOSEK_GAP_TOL} used to set opt.MSK_DPAR_INTPNT_TOL_REL_GAP
            - C{MOSEK_MAX_TIME} used to set opt.MSK_DPAR_OPTIMIZER_MAX_TIME
            - C{MOSEK_NUM_THREADS} used to set opt.MSK_IPAR_INTPNT_NUM_THREADS
            - C{MOSEK_OPT} user option file, if ppopt['MOSEK_OPT'] is non-zero
            it is appended to 'mosek_user_options_' to form
            the name of a user-supplied function used as C{fname}
            described above, except with calling syntax::
                modified_opt = fname(default_opt, ppopt)

    Output is a param dict to pass to MOSEKOPT.

    Example:

    If PPOPT['MOSEK_OPT'] = 3, then after setting the default MOSEK options,
    L{mosek_options} will execute the following user-defined function
    to allow option overrides::

        opt = mosek_user_options_3(opt, ppopt)

    The contents of mosek_user_options_3.py, could be something like::

        def mosek_user_options_3(opt, ppopt):
            opt = {}
            opt.MSK_DPAR_INTPNT_TOL_DFEAS   = 1e-9
            opt.MSK_IPAR_SIM_MAX_ITERATIONS = 5000000
            return opt

    See the Parameters reference in Appix E of "The MOSEK
    optimization toolbox for MATLAB manaul" for
    details on the available options.

    U{http://www.mosek.com/documentation/}

    @see: C{mosekopt}, L{ppoption}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # -----  initialization and arg handling  -----
    # defaults
    verbose = 2
    gaptol = 0
    fname = ''

    # get symbolic constant names
    r, res = mosekopt('symbcon echo(0)')
    sc = res['symbcon']

    # second argument
    if ppopt == None:
        if isinstance(ppopt, str):  # 2nd arg is FNAME (string)
            fname = ppopt
            have_ppopt = False
        else:  # 2nd arg is ppopt (MATPOWER options vector)
            have_ppopt = True
            verbose = ppopt['VERBOSE']
            if ppopt['MOSEK_OPT']:
                fname = 'mosek_user_options_#d'  # ppopt['MOSEK_OPT']
    else:
        have_ppopt = False

    opt = {}
    # -----  set default options for MOSEK  -----
    # solution algorithm
    if have_ppopt:
        alg = ppopt['MOSEK_LP_ALG']
        if alg == sc['MSK_OPTIMIZER_FREE'] or \
                alg == sc['MSK_OPTIMIZER_INTPNT'] or \
                alg == sc['MSK_OPTIMIZER_PRIMAL_SIMPLEX'] or \
                alg == sc['MSK_OPTIMIZER_DUAL_SIMPLEX'] or \
                alg == sc['MSK_OPTIMIZER_PRIMAL_DUAL_SIMPLEX'] or \
                alg == sc['MSK_OPTIMIZER_FREE_SIMPLEX'] or \
                alg == sc['MSK_OPTIMIZER_CONCURRENT']:
            opt['MSK_IPAR_OPTIMIZER'] = alg
        else:
            opt['MSK_IPAR_OPTIMIZER'] = sc['MSK_OPTIMIZER_FREE']

        # (make default OPF_VIOLATION correspond to default MSK_DPAR_INTPNT_TOL_PFEAS)
        opt['MSK_DPAR_INTPNT_TOL_PFEAS'] = ppopt['OPF_VIOLATION'] / 500
        if ppopt['MOSEK_MAX_IT']:
            opt['MSK_IPAR_INTPNT_MAX_ITERATIONS'] = ppopt['MOSEK_MAX_IT']

        if ppopt['MOSEK_GAP_TOL']:
            opt['MSK_DPAR_INTPNT_TOL_REL_GAP'] = ppopt['MOSEK_GAP_TOL']

        if ppopt['MOSEK_MAX_TIME']:
            opt['MSK_DPAR_OPTIMIZER_MAX_TIME'] = ppopt['MOSEK_MAX_TIME']

        if ppopt['MOSEK_NUM_THREADS']:
            opt['MSK_IPAR_INTPNT_NUM_THREADS'] = ppopt['MOSEK_NUM_THREADS']
    else:
        opt['MSK_IPAR_OPTIMIZER'] = sc['MSK_OPTIMIZER_FREE']

    # opt['MSK_DPAR_INTPNT_TOL_PFEAS'] = 1e-8       ## primal feasibility tol
    # opt['MSK_DPAR_INTPNT_TOL_DFEAS'] = 1e-8       ## dual feasibility tol
    # opt['MSK_DPAR_INTPNT_TOL_MU_RED'] = 1e-16     ## relative complementarity gap tol
    # opt['MSK_DPAR_INTPNT_TOL_REL_GAP'] = 1e-8     ## relative gap termination tol
    # opt['MSK_IPAR_INTPNT_MAX_ITERATIONS'] = 400   ## max iterations for int point
    # opt['MSK_IPAR_SIM_MAX_ITERATIONS'] = 10000000 ## max iterations for simplex
    # opt['MSK_DPAR_OPTIMIZER_MAX_TIME'] = -1       ## max time allowed (< 0 --> Inf)
    # opt['MSK_IPAR_INTPNT_NUM_THREADS'] = 1        ## number of threads
    # opt['MSK_IPAR_PRESOLVE_USE'] = sc['MSK_PRESOLVE_MODE_OFF']

    # if verbose == 0:
    #     opt['MSK_IPAR_LOG'] = 0
    #

    # -----  call user function to modify defaults  -----
    if len(fname) > 0:
        if have_ppopt:
            opt = putils.feval(fname, opt, ppopt)
        else:
            opt = putils.feval(fname, opt)

    # -----  apply overrides  -----
    if overrides is not None:
        names = overrides.keys()
        for k in range(len(names)):
            opt[names[k]] = overrides[names[k]]
