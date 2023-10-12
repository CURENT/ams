"""
Python Interior Point Solver (PIPS).
"""
import logging  # NOQA

import numpy as np  # NOQA
from numpy import flatnonzero as find  # NOQA

import scipy.sparse as sp  # NOQA
from scipy.sparse import csr_matrix as c_sparse  # NOQA

from andes.shared import deg2rad, rad2deg  # NOQA

from ams.pypower.eps import EPS  # NOQA
from ams.pypower.idx import IDX  # NOQA
from ams.pypower.utils import sub2ind  # NOQA
from ams.pypower.make import makeYbus  # NOQA
from ams.pypower.routines.opffcns import (opf_costfcn, opf_consfcn,
                                          opf_hessfcn,)  # NOQA


logger = logging.getLogger(__name__)


def pips(f_fcn, x0=None, A=None, l=None, u=None, xmin=None, xmax=None,
         gh_fcn=None, hess_fcn=None, opt=None):
    """
    Primal-dual interior point method for NLP (nonlinear programming).

    Minimize a function F(X) beginning from a starting point x0, subject to
    optional linear and nonlinear constraints and variable bounds:

    min f(x)
     x

    subject to:

    g(x) = 0            (nonlinear equalities)
    h(x) <= 0           (nonlinear inequalities)
    l <= A*x <= u       (linear constraints)
    xmin <= x <= xmax   (variable bounds)

    Note: The calling syntax is almost identical to that of FMINCON from
    MathWorks' Optimization Toolbox. The main difference is that the linear
    constraints are specified with A, l, u instead of A, B,
    Aeq, Beq. The functions for evaluating the objective function,
    constraints and Hessian are identical.

    Example from http://en.wikipedia.org/wiki/Nonlinear_programming:
    >>> from numpy import array, r_, float64, dot
    >>> from scipy.sparse import csr_matrix
    >>> def f2(x):
    ...     f = -x[0] * x[1] - x[1] * x[2]
    ...     df = -r_[x[1], x[0] + x[2], x[1]]
    ...     # actually not used since 'hess_fcn' is provided
    ...     d2f = -array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float64)
    ...     return f, df, d2f
    >>> def gh2(x):
    ...     h = dot(array([[1, -1, 1],
    ...                    [1,  1, 1]]), x**2) + array([-2.0, -10.0])
    ...     dh = 2 * csr_matrix(array([[ x[0], x[0]],
    ...                                [-x[1], x[1]],
    ...                                [ x[2], x[2]]]))
    ...     g = array([])
    ...     dg = None
    ...     return h, g, dh, dg
    >>> def hess2(x, lam, cost_mult=1):
    ...     mu = lam["ineqnonlin"]
    ...     a = r_[dot(2 * array([1, 1]), mu), -1, 0]
    ...     b = r_[-1, dot(2 * array([-1, 1]), mu),-1]
    ...     c = r_[0, -1, dot(2 * array([1, 1]), mu)]
    ...     Lxx = csr_matrix(array([a, b, c]))
    ...     return Lxx
    >>> x0 = array([1, 1, 0], float64)
    >>> solution = pips(f2, x0, gh_fcn=gh2, hess_fcn=hess2)
    >>> round(solution["f"], 11) == -7.07106725919
    True
    >>> solution["output"]["iterations"]
    8

    All parameters are optional except f_fcn and x0.

    Parameters:
    ----------
    f_fcn : callable
        Function that evaluates the objective function, its gradients
        and Hessian for a given value of x. If there are
        nonlinear constraints, the Hessian information is provided
        by the 'hess_fcn' argument and is not required here.
    x0 : array, optional
        Starting value of optimization vector x.
    A : csr_matrix, optional
        Optional linear constraints.
    l : array, optional
        Optional linear constraints. Default values are -Inf.
    u : array, optional
        Optional linear constraints. Default values are Inf.
    xmin : array, optional
        Optional lower bounds on the x variables, defaults are
        -Inf.
    xmax : array, optional
        Optional upper bounds on the x variables, defaults are
        Inf.
    gh_fcn : callable, optional
        Function that evaluates the optional nonlinear constraints
        and their gradients for a given value of x.
    hess_fcn : callable, optional
        Handle to function that computes the Hessian of the
        Lagrangian for given values of x, lambda and mu,
        where lambda and mu are the multipliers on the
        equality and inequality constraints, g and h,
        respectively.
    opt : dict, optional
        Optional options dictionary with the following keys, all of
        which are also optional (default values shown in parentheses)
            - 'verbose' (False) - Controls level of progress output
              displayed
            - 'feastol' (1e-6) - termination tolerance for feasibility
              condition
            - 'gradtol' (1e-6) - termination tolerance for gradient
              condition
            - 'comptol' (1e-6) - termination tolerance for
              complementarity condition
            - 'costtol' (1e-6) - termination tolerance for cost
              condition
            - 'max_it' (150) - maximum number of iterations
            - 'step_control' (False) - set to True to enable step-size
              control
            - 'max_red' (20) - maximum number of step-size reductions if
              step-control is on
            - 'cost_mult' (1.0) - cost multiplier used to scale the
              objective function for improved conditioning. Note: This
              value is also passed as the 3rd argument to the Hessian
              evaluation function so that it can appropriately scale the
              objective function term in the Hessian of the Lagrangian.

    Returns:
    --------
    dict
        The solution dictionary has the following keys:
            - 'x' - solution vector
            - 'f' - final objective function value
            - 'converged' - exit status
                - True = first order optimality conditions satisfied
                - False = maximum number of iterations reached
                - None = numerically failed
            - 'output' - output dictionary with keys:
                - 'iterations' - number of iterations performed
                - 'hist' - list of arrays with trajectories of the
                following: feascond, gradcond, compcond, costcond, gamma,
                stepsize, obj, alphap, alphad
                - 'message' - exit message
            - 'lmbda' - dictionary containing the Langrange and Kuhn-Tucker
            multipliers on the constraints, with keys:
                - 'eqnonlin' - nonlinear equality constraints
                - 'ineqnonlin' - nonlinear inequality constraints
                - 'mu_l' - lower (left-hand) limit on linear constraints
                - 'mu_u' - upper (right-hand) limit on linear constraints
                - 'lower' - lower bound on optimization variables
                - 'upper' - upper bound on optimization variables

    Notes
    -----
    1. Ported by Richard Lincoln from the MATLAB Interior Point Solver (MIPS)
        (v1.9) by Ray Zimmerman.  MIPS is distributed as part of the MATPOWER
        project, developed at PSERC, Cornell.

    Reference:

    [1] "On the Computation and Application of Multi-period Security-Constrained
    Optimal Power Flow for Real-time Electricity Market Operations",
    Cornell University, May 2007.

    [2] H. Wang, C. E. Murillo-Sanchez, R. D. Zimmerman, R. J. Thomas, "On Computational
    Issues of Market-Based Optimal Power Flow", IEEE Transactions on Power Systems,
    Vol. 22, No. 3, Aug. 2007, pp. 1185-1193.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    if isinstance(f_fcn, dict):  # problem dict
        p = f_fcn
        f_fcn = p['f_fcn']
        x0 = p['x0']
        if 'opt' in p:
            opt = p['opt']
        if 'hess_fcn' in p:
            hess_fcn = p['hess_fcn']
        if 'gh_fcn' in p:
            gh_fcn = p['gh_fcn']
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

    nx = x0.shape[0]                        # number of variables
    nA = A.shape[0] if A is not None else 0  # number of original linear constr

    # default argument values
    if l is None or len(l) == 0:
        l = -np.Inf * np.ones(nA)
    if u is None or len(u) == 0:
        u = np.Inf * np.ones(nA)
    if xmin is None or len(xmin) == 0:
        xmin = -np.Inf * np.ones(x0.shape[0])
    if xmax is None or len(xmax) == 0:
        xmax = np.Inf * np.ones(x0.shape[0])
    if gh_fcn is None:
        nonlinear = False
        gn = np.array([])
        hn = np.array([])
    else:
        nonlinear = True

    if opt is None:
        opt = {}
    # options
    if "feastol" not in opt:
        opt["feastol"] = 1e-06
    if "gradtol" not in opt:
        opt["gradtol"] = 1e-06
    if "comptol" not in opt:
        opt["comptol"] = 1e-06
    if "costtol" not in opt:
        opt["costtol"] = 1e-06
    if "max_it" not in opt:
        opt["max_it"] = 150
    if "max_red" not in opt:
        opt["max_red"] = 20
    if "step_control" not in opt:
        opt["step_control"] = False
    if "cost_mult" not in opt:
        opt["cost_mult"] = 1
    if "verbose" not in opt:
        opt["verbose"] = 0

    # initialize history
    hist = []

    # constants
    xi = 0.99995
    sigma = 0.1
    z0 = 1
    alpha_min = 1e-8
    rho_min = 0.95
    rho_max = 1.05
    mu_threshold = 1e-5

    # initialize
    i = 0                       # iteration counter
    converged = False           # flag
    eflag = False               # exit flag

    # add var limits to linear constraints
    eyex = sp.eye(nx, nx, format="csr")
    AA = eyex if A is None else sp.vstack([eyex, A], "csr")
    ll = np.r_[xmin, l]
    uu = np.r_[xmax, u]

    # split up linear constraints
    ieq = find(np.abs(uu - ll) <= EPS)
    igt = find((uu >= 1e10) & (ll > -1e10))
    ilt = find((ll <= -1e10) & (uu < 1e10))
    ibx = find((np.abs(uu - ll) > EPS) & (uu < 1e10) & (ll > -1e10))
    # zero-sized sparse matrices unsupported
    Ae = AA[ieq, :] if len(ieq) else None
    if len(ilt) or len(igt) or len(ibx):
        idxs = [(1, ilt), (-1, igt), (1, ibx), (-1, ibx)]
        Ai = sp.vstack([sig * AA[idx, :] for sig, idx in idxs if len(idx)], 'csr')
    else:
        Ai = None
    be = uu[ieq]
    bi = np.r_[uu[ilt], -ll[igt], uu[ibx], -ll[ibx]]

    # evaluate cost f(x0) and constraints g(x0), h(x0)
    x = x0
    f, df = f_fcn(x)                 # cost
    f = f * opt["cost_mult"]
    df = df * opt["cost_mult"]
    if nonlinear:
        hn, gn, dhn, dgn = gh_fcn(x)        # nonlinear constraints
        h = hn if Ai is None else np.r_[hn, Ai * x - bi]  # inequality constraints
        g = gn if Ae is None else np.r_[gn, Ae * x - be]  # equality constraints

        if (dhn is None) and (Ai is None):
            dh = None
        elif dhn is None:
            dh = Ai.T
        elif Ai is None:
            dh = dhn
        else:
            dh = sp.hstack([dhn, Ai.T])

        if (dgn is None) and (Ae is None):
            dg = None
        elif dgn is None:
            dg = Ae.T
        elif Ae is None:
            dg = dgn
        else:
            dg = sp.hstack([dgn, Ae.T])
    else:
        h = -bi if Ai is None else Ai * x - bi        # inequality constraints
        g = -be if Ae is None else Ae * x - be        # equality constraints
        dh = None if Ai is None else Ai.T     # 1st derivative of inequalities
        dg = None if Ae is None else Ae.T     # 1st derivative of equalities

    # some dimensions
    neq = g.shape[0]           # number of equality constraints
    niq = h.shape[0]           # number of inequality constraints
    neqnln = gn.shape[0]       # number of nonlinear equality constraints
    niqnln = hn.shape[0]       # number of nonlinear inequality constraints
    nlt = len(ilt)             # number of upper bounded linear inequalities
    ngt = len(igt)             # number of lower bounded linear inequalities
    nbx = len(ibx)             # number of doubly bounded linear inequalities

    # initialize gamma, lam, mu, z, e
    gamma = 1                  # barrier coefficient
    lam = np.zeros(neq)
    z = z0 * np.ones(niq)
    mu = z0 * np.ones(niq)
    k = find(h < -z0)
    z[k] = -h[k]
    k = find((gamma / z) > z0)
    mu[k] = gamma / z[k]
    e = np.ones(niq)

    # check tolerance
    f0 = f
    if opt["step_control"]:
        L = f + np.dot(lam, g) + np.dot(mu, h + z) - gamma * sum(np.log(z))

    Lx = df.copy()
    Lx = Lx + dg * lam if dg is not None else Lx
    Lx = Lx + dh * mu if dh is not None else Lx

    maxh = np.zeros(1) if len(h) == 0 else max(h)

    gnorm = np.linalg.norm(g, np.Inf) if len(g) else 0.0
    lam_norm = np.linalg.norm(lam, np.Inf) if len(lam) else 0.0
    mu_norm = np.linalg.norm(mu, np.Inf) if len(mu) else 0.0
    znorm = np.linalg.norm(z, np.Inf) if len(z) else 0.0
    feascond = \
        max([gnorm, maxh]) / (1 + max([np.linalg.norm(x, np.Inf), znorm]))
    gradcond = \
        np.linalg.norm(Lx, np.Inf) / (1 + max([lam_norm, mu_norm]))
    compcond = np.dot(z, mu) / (1 + np.linalg.norm(x, np.Inf))
    costcond = np.abs(f - f0) / (1 + np.abs(f0))

    # save history
    hist.append({'feascond': feascond, 'gradcond': gradcond,
                 'compcond': compcond, 'costcond': costcond, 'gamma': gamma,
                 'stepsize': 0, 'obj': f / opt["cost_mult"], 'alphap': 0, 'alphad': 0})

    s = '-sc' if opt["step_control"] else ''
    headers = '  n     objective     step size'
    headers += '    feascond    gradcond'
    headers += '    compcond    costcond'
    head_line = "=== ========  ======  "
    head_line += "   =======  ======="
    head_line += "   =======  ======="
    logger.debug(headers)
    logger.debug(head_line)
    logger.debug("%3d  %12.8g %12g %12g %12g %12g %12g" %
                 (i, (f / opt["cost_mult"]), 0, feascond, gradcond,
                     compcond, costcond))

    if feascond < opt["feastol"] and gradcond < opt["gradtol"] and \
            compcond < opt["comptol"] and costcond < opt["costtol"]:
        converged = True

    # do Newton iterations
    while (not converged) and (i < opt["max_it"]):
        # update iteration counter
        i += 1

        # compute update step
        lmbda = {"eqnonlin": lam[range(neqnln)],
                 "ineqnonlin": mu[range(niqnln)]}
        if nonlinear:
            if hess_fcn is None:
                logger.error("pips: Hessian evaluation via finite differences "
                             "not yet implemented.\nPlease provide "
                             "your own hessian evaluation function.")
            Lxx = hess_fcn(x, lmbda, opt["cost_mult"])
        else:
            _, _, d2f = f_fcn(x, True)      # cost
            Lxx = d2f * opt["cost_mult"]
        rz = range(len(z))
        zinvdiag = c_sparse((1.0 / z, (rz, rz))) if len(z) else None
        rmu = range(len(mu))
        mudiag = c_sparse((mu, (rmu, rmu))) if len(mu) else None
        dh_zinv = None if dh is None else dh * zinvdiag
        M = Lxx if dh is None else Lxx + dh_zinv * mudiag * dh.T
        N = Lx if dh is None else Lx + dh_zinv * (mudiag * h + gamma * e)

        Ab = c_sparse(M) if dg is None else sp.vstack([
            sp.hstack([M, dg]),
            sp.hstack([dg.T, c_sparse((neq, neq))])
        ])
        bb = np.r_[-N, -g]

        dxdlam = sp.linalg.spsolve(Ab.tocsr(), bb)

        if np.any(np.isnan(dxdlam)):
            print('\nNumerically Failed\n')
            eflag = -1
            break

        dx = dxdlam[:nx]
        dlam = dxdlam[nx:nx + neq]
        dz = -h - z if dh is None else -h - z - dh.T * dx
        dmu = -mu if dh is None else -mu + zinvdiag * (gamma * e - mudiag * dz)

        # optional step-size control
        sc = False
        if opt["step_control"]:
            x1 = x + dx

            # evaluate cost, constraints, derivatives at x1
            f1, df1 = f_fcn(x1)          # cost
            f1 = f1 * opt["cost_mult"]
            df1 = df1 * opt["cost_mult"]
            if nonlinear:
                hn1, gn1, dhn1, dgn1 = gh_fcn(x1)  # nonlinear constraints

                h1 = hn1 if Ai is None else np.r_[hn1, Ai * x1 - bi]  # ieq constraints
                g1 = gn1 if Ae is None else np.r_[gn1, Ae * x1 - be]  # eq constraints

                # 1st der of ieq
                if (dhn1 is None) and (Ai is None):
                    dh1 = None
                elif dhn1 is None:
                    dh1 = Ai.T
                elif Ai is None:
                    dh1 = dhn1
                else:
                    dh1 = sp.hstack([dhn1, Ai.T])

                # 1st der of eqs
                if (dgn1 is None) and (Ae is None):
                    dg1 = None
                elif dgn is None:
                    dg1 = Ae.T
                elif Ae is None:
                    dg1 = dgn1
                else:
                    dg1 = sp.hstack([dgn1, Ae.T])
            else:
                h1 = -bi if Ai is None else Ai * x1 - bi    # inequality constraints
                g1 = -be if Ae is None else Ae * x1 - be    # equality constraints

                dh1 = dh  # 1st derivative of inequalities
                dg1 = dg  # 1st derivative of equalities

            # check tolerance
            Lx1 = df1
            Lx1 = Lx1 + dg1 * lam if dg1 is not None else Lx1
            Lx1 = Lx1 + dh1 * mu if dh1 is not None else Lx1

            maxh1 = np.zeros(1) if len(h1) == 0 else max(h1)

            g1norm = np.linalg.norm(g1, np.Inf) if len(g1) else 0.0
            lam1_norm = np.linalg.norm(lam, np.Inf) if len(lam) else 0.0
            mu1_norm = np.linalg.norm(mu, np.Inf) if len(mu) else 0.0
            z1norm = np.linalg.norm(z, np.Inf) if len(z) else 0.0

            feascond1 = max([g1norm, maxh1]) / \
                (1 + max([np.linalg.norm(x1, np.Inf), z1norm]))
            gradcond1 = np.linalg.norm(Lx1, np.Inf) / (1 + max([lam1_norm, mu1_norm]))

            if (feascond1 > feascond) and (gradcond1 > gradcond):
                sc = True
        if sc:
            alpha = 1.0
            for j in range(opt["max_red"]):
                dx1 = alpha * dx
                x1 = x + dx1
                f1, _ = f_fcn(x1)             # cost
                f1 = f1 * opt["cost_mult"]
                if nonlinear:
                    hn1, gn1, _, _ = gh_fcn(x1)              # nonlinear constraints
                    h1 = hn1 if Ai is None else np.r_[hn1, Ai * x1 - bi]         # inequality constraints
                    g1 = gn1 if Ae is None else np.r_[gn1, Ae * x1 - be]         # equality constraints
                else:
                    h1 = -bi if Ai is None else Ai * x1 - bi    # inequality constraints
                    g1 = -be if Ae is None else Ae * x1 - be    # equality constraints

                L1 = f1 + np.dot(lam, g1) + np.dot(mu, h1 + z) - gamma * sum(np.log(z))

                logger.info("   %3d            %10.5f" % (-j, np.linalg.norm(dx1)))

                rho = (L1 - L) / (np.dot(Lx, dx1) + 0.5 * np.dot(dx1, Lxx * dx1))

                if (rho > rho_min) and (rho < rho_max):
                    break
                else:
                    alpha = alpha / 2.0
            dx = alpha * dx
            dz = alpha * dz
            dlam = alpha * dlam
            dmu = alpha * dmu

        # do the update
        k = find(dz < 0.0)
        alphap = min([xi * min(z[k] / -dz[k]), 1]) if len(k) else 1.0
        k = find(dmu < 0.0)
        alphad = min([xi * min(mu[k] / -dmu[k]), 1]) if len(k) else 1.0
        x = x + alphap * dx
        z = z + alphap * dz
        lam = lam + alphad * dlam
        mu = mu + alphad * dmu
        if niq > 0:
            gamma = sigma * np.dot(z, mu) / niq

        # evaluate cost, constraints, derivatives
        f, df = f_fcn(x)             # cost
        f = f * opt["cost_mult"]
        df = df * opt["cost_mult"]
        if nonlinear:
            hn, gn, dhn, dgn = gh_fcn(x)                   # nln constraints
            h = hn if Ai is None else np.r_[hn, Ai * x - bi]  # ieq constr
            g = gn if Ae is None else np.r_[gn, Ae * x - be]  # eq constr

            if (dhn is None) and (Ai is None):
                dh = None
            elif dhn is None:
                dh = Ai.T
            elif Ai is None:
                dh = dhn
            else:
                dh = sp.hstack([dhn, Ai.T])

            if (dgn is None) and (Ae is None):
                dg = None
            elif dgn is None:
                dg = Ae.T
            elif Ae is None:
                dg = dgn
            else:
                dg = sp.hstack([dgn, Ae.T])
        else:
            h = -bi if Ai is None else Ai * x - bi    # inequality constraints
            g = -be if Ae is None else Ae * x - be    # equality constraints
            # 1st derivatives are constant, still dh = Ai.T, dg = Ae.T

        Lx = df
        Lx = Lx + dg * lam if dg is not None else Lx
        Lx = Lx + dh * mu if dh is not None else Lx

        if len(h) == 0:
            maxh = np.zeros(1)
        else:
            maxh = max(h)

        gnorm = np.linalg.norm(g, np.Inf) if len(g) else 0.0
        lam_norm = np.linalg.norm(lam, np.Inf) if len(lam) else 0.0
        mu_norm = np.linalg.norm(mu, np.Inf) if len(mu) else 0.0
        znorm = np.linalg.norm(z, np.Inf) if len(z) else 0.0
        feascond = \
            max([gnorm, maxh]) / (1 + max([np.linalg.norm(x, np.Inf), znorm]))
        gradcond = \
            np.linalg.norm(Lx, np.Inf) / (1 + max([lam_norm, mu_norm]))
        compcond = np.dot(z, mu) / (1 + np.linalg.norm(x, np.Inf))
        costcond = float(np.abs(f - f0) / (1 + np.abs(f0)))

        hist.append({'feascond': feascond, 'gradcond': gradcond,
                     'compcond': compcond, 'costcond': costcond, 'gamma': gamma,
                     'stepsize': np.linalg.norm(dx), 'obj': f / opt["cost_mult"],
                     'alphap': alphap, 'alphad': alphad})

        logger.debug("%3d  %12.8g %10.5g %12g %12g %12g %12g" %
                     (i, (f / opt["cost_mult"]), np.linalg.norm(dx), feascond, gradcond,
                      compcond, costcond))

        if feascond < opt["feastol"] and gradcond < opt["gradtol"] and \
                compcond < opt["comptol"] and costcond < opt["costtol"]:
            converged = True
        else:
            if np.any(np.isnan(x)) or (alphap < alpha_min) or \
                    (alphad < alpha_min) or (gamma < EPS) or (gamma > 1.0 / EPS):
                eflag = -1
                break
            f0 = f

            if opt["step_control"]:
                L = f + np.dot(lam, g) + np.dot(mu, (h + z)) - gamma * sum(np.log(z))

    if not converged:
        logger.debug("Did not converge in %d iterations." % i)

    # package results
    if eflag != -1:
        eflag = converged

    if eflag == 0:
        message = 'Did not converge'
    elif eflag == 1:
        message = 'Converged'
    elif eflag == -1:
        message = 'Numerically failed'
    else:
        raise

    output = {"iterations": i, "hist": hist, "message": message}

    # zero out multipliers on non-binding constraints
    mu[find((h < -opt["feastol"]) & (mu < mu_threshold))] = 0.0

    # un-scale cost and prices
    f = f / opt["cost_mult"]
    lam = lam / opt["cost_mult"]
    mu = mu / opt["cost_mult"]

    # re-package multipliers into struct
    lam_lin = lam[neqnln:neq]           # lambda for linear constraints
    mu_lin = mu[niqnln:niq]             # mu for linear constraints
    kl = find(lam_lin < 0.0)     # lower bound binding
    ku = find(lam_lin > 0.0)     # upper bound binding

    mu_l = np.zeros(nx + nA)
    mu_l[ieq[kl]] = -lam_lin[kl]
    mu_l[igt] = mu_lin[nlt:nlt + ngt]
    mu_l[ibx] = mu_lin[nlt + ngt + nbx:nlt + ngt + nbx + nbx]

    mu_u = np.zeros(nx + nA)
    mu_u[ieq[ku]] = lam_lin[ku]
    mu_u[ilt] = mu_lin[:nlt]
    mu_u[ibx] = mu_lin[nlt + ngt:nlt + ngt + nbx]

    lmbda = {'mu_l': mu_l[nx:], 'mu_u': mu_u[nx:],
             'lower': mu_l[:nx], 'upper': mu_u[:nx]}

    if niqnln > 0:
        lmbda['ineqnonlin'] = mu[:niqnln]
    if neqnln > 0:
        lmbda['eqnonlin'] = lam[:neqnln]

#    lmbda = {"eqnonlin": lam[:neqnln], 'ineqnonlin': mu[:niqnln],
#             "mu_l": mu_l[nx:], "mu_u": mu_u[nx:],
#             "lower": mu_l[:nx], "upper": mu_u[:nx]}

    solution = {"x": x, "f": f, "eflag": converged,
                "output": output, "lmbda": lmbda}

    return solution


def pipsver(*args):
    """
    Return PIPS version information for the current installation.

    Author: Ray Zimmerman (PSERC Cornell)

    Returns
    -------
    dict
        A dictionary containing PIPS version information with the following keys:
        - 'Name': Name of the software (PIPS)
        - 'Version': Version number (1.0)
        - 'Release': Release information (empty string)
        - 'Date': Date of release (07-Feb-2011)

    Author
    ------
    Ray Zimmerman (PSERC Cornell)
    """
    ver = {'Name': 'PIPS',
           'Version': '1.0',
           'Release':  '',
           'Date': '07-Feb-2011'}

    return ver


def pipsopf_solver(om, ppopt, out_opt=None):
    """
    Solves AC optimal power flow using PIPS.

    Inputs are an OPF model object, a PYPOWER options vector, and
    a dict containing keys (can be empty) for each of the desired
    optional output fields.

    Outputs are a `results` dict, a `success` flag, and a `raw` output dict.

    Parameters
    ----------
    om : object
        OPF model object.
    ppopt : dict
        PYPOWER options vector.
    out_opt : dict, optional
        Dictionary containing keys for optional output fields (default is None).

    Returns
    -------
    dict
        A `results` dictionary containing various fields, including optimization
        variables, objective function value, and shadow prices on constraints.
    bool
        A `success` flag indicating whether the solver converged successfully.
    dict
        A `raw` output dictionary in the form returned by MINOS, containing
        information about optimization variables, constraint multipliers, solver
        termination code, and solver-specific output information.

    Notes
    -----
    The `results` dictionary contains fields such as 'order', 'x', 'f', 'mu',
    where 'mu' includes shadow prices on variables and constraints.

    The `success` flag is `True` if the solver converged successfully, and `False`
    otherwise.

    The `raw` output dictionary includes information specific to the solver.

    Author
    ------
    Ray Zimmerman (PSERC Cornell)

    Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad Autonoma de Manizales)
    """
    # ----- initialization -----
    # optional output
    if out_opt is None:
        out_opt = {}

    # options
    verbose = ppopt['VERBOSE']
    feastol = ppopt['PDIPM_FEASTOL']
    gradtol = ppopt['PDIPM_GRADTOL']
    comptol = ppopt['PDIPM_COMPTOL']
    costtol = ppopt['PDIPM_COSTTOL']
    max_it = ppopt['PDIPM_MAX_IT']
    max_red = ppopt['SCPDIPM_RED_IT']
    step_control = (ppopt['OPF_ALG'] == 565)  # OPF_ALG == 565, PIPS-sc
    if feastol == 0:
        feastol = ppopt['OPF_VIOLATION']
    opt = {'feastol': feastol,
           'gradtol': gradtol,
           'comptol': comptol,
           'costtol': costtol,
           'max_it': max_it,
           'max_red': max_red,
           'step_control': step_control,
           'cost_mult': 1e-4,
           'verbose': verbose}

    # unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]
    vv, _, nn, _ = om.get_idx()

    # problem dimensions
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of branches
    ny = om.getN('var', 'y')  # number of piece-wise linear costs

    # linear constraints
    A, l, u = om.linear_constraints()

    # bounds on optimization vars
    _, xmin, xmax = om.getv()

    # build admittance matrices
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    # try to select an interior initial point
    ll, uu = xmin.copy(), xmax.copy()
    ll[xmin == -np.Inf] = -1e10  # replace Inf with numerical proxies
    uu[xmax == np.Inf] = 1e10
    x0 = (ll + uu) / 2
    Varefs = bus[bus[:, IDX.bus.BUS_TYPE] == IDX.bus.REF, IDX.bus.VA] * deg2rad
    # angles set to first reference angle
    x0[vv["i1"]["Va"]:vv["iN"]["Va"]] = Varefs[0]
    if ny > 0:
        ipwl = find(gencost[:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR)
#         PQ = np.r_[gen[:, PMAX], gen[:, QMAX]]
#         c = totcost(gencost[ipwl, :], PQ[ipwl])
        # largest y-value in CCV data
        c = gencost.flatten('F')[sub2ind(gencost.shape, ipwl, IDX.cost.NCOST+2*gencost[ipwl, IDX.cost.NCOST])]
        x0[vv["i1"]["y"]:vv["iN"]["y"]] = max(c) + 0.1 * abs(max(c))
#        x0[vv["i1"]["y"]:vv["iN"]["y"]] = c + 0.1 * abs(c)

    # find branches with flow limits
    il = find((branch[:, IDX.branch.RATE_A] != 0) & (branch[:, IDX.branch.RATE_A] < 1e10))
    nl2 = len(il)  # number of constrained lines

    # -----  run opf  -----
    def f_fcn(x, return_hessian=False): return opf_costfcn(x, om, return_hessian)
    def gh_fcn(x): return opf_consfcn(x, om, Ybus, Yf[il, :], Yt[il, :], ppopt, il)

    def hess_fcn(
        x, lmbda, cost_mult): return opf_hessfcn(
            x, lmbda, om, Ybus, Yf[il, :],
            Yt[il, :],
        ppopt, il, cost_mult)

    solution = pips(f_fcn, x0, A, l, u, xmin, xmax, gh_fcn, hess_fcn, opt)
    x, f, info, lmbda, output = solution["x"], solution["f"], \
        solution["eflag"], solution["lmbda"], solution["output"]

    success = (info > 0)

    # update solution data
    Va = x[vv["i1"]["Va"]:vv["iN"]["Va"]]
    Vm = x[vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]
    Qg = x[vv["i1"]["Qg"]:vv["iN"]["Qg"]]

    V = Vm * np.exp(1j * Va)

    # -----  calculate return values  -----
    # update voltages & generator outputs
    bus[:, IDX.bus.VA] = Va * rad2deg

    bus[:, IDX.bus.VM] = Vm
    gen[:, IDX.gen.PG] = Pg * baseMVA
    gen[:, IDX.gen.QG] = Qg * baseMVA
    gen[:, IDX.gen.VG] = Vm[gen[:, IDX.gen.GEN_BUS].astype(int)]

    # compute branch flows
    Sf = V[branch[:, IDX.branch.F_BUS].astype(int)] * np.conj(Yf * V)  # cplx pwr at "from" bus, p["u"].
    St = V[branch[:, IDX.branch.T_BUS].astype(int)] * np.conj(Yt * V)  # cplx pwr at "to" bus, p["u"].
    branch[:, IDX.branch.PF] = Sf.real * baseMVA
    branch[:, IDX.branch.QF] = Sf.imag * baseMVA
    branch[:, IDX.branch.PT] = St.real * baseMVA
    branch[:, IDX.branch.QT] = St.imag * baseMVA

    # line constraint is actually on square of limit
    # so we must fix multipliers
    muSf = np.zeros(nl)
    muSt = np.zeros(nl)
    if len(il) > 0:
        muSf[il] = \
            2 * lmbda["ineqnonlin"][:nl2] * branch[il, IDX.branch.RATE_A] / baseMVA
        muSt[il] = \
            2 * lmbda["ineqnonlin"][nl2:nl2+nl2] * branch[il, IDX.branch.RATE_A] / baseMVA

    # update Lagrange multipliers
    bus[:, IDX.bus.MU_VMAX] = lmbda["upper"][vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    bus[:, IDX.bus.MU_VMIN] = lmbda["lower"][vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    gen[:, IDX.gen.MU_PMAX] = lmbda["upper"][vv["i1"]["Pg"]:vv["iN"]["Pg"]] / baseMVA
    gen[:, IDX.gen.MU_PMIN] = lmbda["lower"][vv["i1"]["Pg"]:vv["iN"]["Pg"]] / baseMVA
    gen[:, IDX.gen.MU_QMAX] = lmbda["upper"][vv["i1"]["Qg"]:vv["iN"]["Qg"]] / baseMVA
    gen[:, IDX.gen.MU_QMIN] = lmbda["lower"][vv["i1"]["Qg"]:vv["iN"]["Qg"]] / baseMVA

    bus[:, IDX.bus.LAM_P] = \
        lmbda["eqnonlin"][nn["i1"]["Pmis"]:nn["iN"]["Pmis"]] / baseMVA
    bus[:, IDX.bus.LAM_Q] = \
        lmbda["eqnonlin"][nn["i1"]["Qmis"]:nn["iN"]["Qmis"]] / baseMVA
    branch[:, IDX.branch.MU_SF] = muSf / baseMVA
    branch[:, IDX.branch.MU_ST] = muSt / baseMVA

    # package up results
    nlnN = om.getN('nln')

    # extract multipliers for nonlinear constraints
    kl = find(lmbda["eqnonlin"] < 0)
    ku = find(lmbda["eqnonlin"] > 0)
    nl_mu_l = np.zeros(nlnN)
    nl_mu_u = np.r_[np.zeros(2*nb), muSf, muSt]
    nl_mu_l[kl] = -lmbda["eqnonlin"][kl]
    nl_mu_u[ku] = lmbda["eqnonlin"][ku]

    mu = {
        'var': {'l': lmbda["lower"], 'u': lmbda["upper"]},
        'nln': {'l': nl_mu_l, 'u': nl_mu_u},
        'lin': {'l': lmbda["mu_l"], 'u': lmbda["mu_u"]}}

    results = ppc
    results["bus"], results["branch"], results["gen"], \
        results["om"], results["x"], results["mu"], results["f"] = \
        bus, branch, gen, om, x, mu, f

    pimul = np.r_[
        results["mu"]["nln"]["l"] - results["mu"]["nln"]["u"],
        results["mu"]["lin"]["l"] - results["mu"]["lin"]["u"],
        -np.ones(int(ny > 0)),
        results["mu"]["var"]["l"] - results["mu"]["var"]["u"],
    ]
    raw = {'xr': x, 'pimul': pimul, 'info': info, 'output': output}

    return results, success, raw
