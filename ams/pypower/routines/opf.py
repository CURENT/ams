"""
Module to solve OPF.
"""

import logging
from copy import deepcopy

import numpy as np
from numpy import flatnonzero as find

import scipy.sparse as sp
from scipy.sparse import csr_matrix as c_sparse

from andes.shared import deg2rad
from andes.utils.misc import elapsed

from ams.pypower.core import (ppoption, pipsopf_solver, ipoptopf_solver)
from ams.pypower.utils import isload, fairmax
from ams.pypower.idx import IDX
from ams.pypower.io import loadcase
from ams.pypower.make import (makeYbus, makeAvl, makeApq,
                              makeAang, makeAy)

import ams.pypower.routines.opffcns as opfcn

from ams.pypower.toggle import toggle_reserves

logger = logging.getLogger(__name__)


def runopf(casedata, ppopt):
    """
    Runs an optimal power flow.

    @see: L{rundcopf}, L{runuopf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    sstats = dict(solver_name='PYPOWER',
                  num_iters=1)  # solver stats
    ppopt = ppoption(ppopt)
    r = fopf(casedata, ppopt)
    sstats['solver_name'] = 'PYPOWER-PIPS'
    sstats['num_iters'] = r['raw']['output']['iterations']
    return r, sstats


def runuopf(casedata, ppopt):
    """
    Runs an optimal power flow with unit-decommitment heuristic.

    @see: L{rundcopf}, L{runuopf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # default arguments
    ppopt = ppoption(ppopt)

    # -----  run the unit de-commitment / optimal power flow  -----
    r = uopf(casedata, ppopt)

    return r


def runduopf(casedata, ppopt):
    """
    Runs a DC optimal power flow with unit-decommitment heuristic.

    @see: L{rundcopf}, L{runuopf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # default arguments
    ppopt = ppoption(ppopt, PF_DC=True)

    return runuopf(casedata, ppopt)


def runopf_w_res(*args):
    """
    Runs an optimal power flow with fixed zonal reserves.

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


def fopf(*args):
    """
    Solve an optimal power flow, return a `results` dict,
    previously named ``opf``.

    The data for the problem can be specified in one of three ways:
      1. a string (ppc) containing the file name of a PYPOWER case
      which defines the data matrices baseMVA, bus, gen, branch, and
      gencost (areas is not used at all, it is only included for
      backward compatibility of the API).
      2. a dict (ppc) containing the data matrices as fields.
      3. the individual data matrices themselves.

    The optional user parameters for user constraints (A, l, u), user costs
    (N, fparm, H, Cw), user variable initializer (z0), and user variable
    limits (zl, zu) can also be specified as fields in a case dict,
    either passed in directly or defined in a case file referenced by name.

    When specified, A, l, u represent additional linear constraints on the
    optimization variables, l <= A*[x z] <= u. If the user specifies an A
    matrix that has more columns than the number of "x" (OPF) variables,
    then there are extra linearly constrained "z" variables. For an
    explanation of the formulation used and instructions for forming the
    A matrix, see the MATPOWER manual.

    A generalized cost on all variables can be applied if input arguments
    N, fparm, H, and Cw are specified. First, a linear transformation
    of the optimization variables is defined by means of r = N * [x z].
    Then, to each element of r a function is applied as encoded in the
    fparm matrix (see MATPOWER manual). If the resulting vector is named
    w, then H and Cw define a quadratic cost on w:
    (1/2)*w'*H*w + Cw * w. H and N should be sparse matrices and H
    should also be symmetric.

    The optional ppopt vector specifies PYPOWER options. If the OPF
    algorithm is not explicitly set in the options, PYPOWER will use the default
    solver, based on a primal-dual interior point method. For the AC OPF, this
    is OPF_ALG = 560. For the DC OPF, the default is OPF_ALG_DC = 200.
    See L{ppoption} for more details on the available OPF solvers and other OPF
    options and their default values.

    The solved case is returned in a single results dict (described
    below). Also returned are the final objective function value (f) and a
    flag which is True if the algorithm was successful in finding a solution
    (success). Additional optional return values are an algorithm specific
    return status (info), elapsed time in seconds (et), the constraint
    vector (g), the Jacobian matrix (jac), and the vector of variables
    (xr) as well as the constraint multipliers (pimul).

    The single results dict is a PYPOWER case struct (ppc) with the
    usual baseMVA, bus, branch, gen, gencost fields, along with the
    following additional fields:

        - order      see 'help ext2int' for details of this field
        - et         elapsed time in seconds for solving OPF
        - success    1 if solver converged successfully, 0 otherwise
        - om         OPF model object, see 'help opf_model'
        - x          final value of optimization variables (internal order)
        - f          final objective function value
        - mu         shadow prices on ...
            - var
                - l  lower bounds on variables
                - u  upper bounds on variables
            - nln
                - l  lower bounds on nonlinear constraints
                - u  upper bounds on nonlinear constraints
            - lin
                - l  lower bounds on linear constraints
                - u  upper bounds on linear constraints
        - g          (optional) constraint values
        - dg         (optional) constraint 1st derivatives
        - df         (optional) obj fun 1st derivatives (not yet implemented)
        - d2f        (optional) obj fun 2nd derivatives (not yet implemented)
        - raw        raw solver output in form returned by MINOS, and more
            - xr     final value of optimization variables
            - pimul  constraint multipliers
            - info   solver specific termination code
            - output solver specific output information
               - alg  algorithm code of solver used
        - var
            - val    optimization variable values, by named block
                - Va     voltage angles
                - Vm     voltage magnitudes (AC only)
                - Pg     real power injections
                - Qg     reactive power injections (AC only)
                - y      constrained cost variable (only if have pwl costs)
                - (other) any user-defined variable blocks
            - mu     variable bound shadow prices, by named block
                - l  lower bound shadow prices
                    - Va, Vm, Pg, Qg, y, (other)
                - u  upper bound shadow prices
                    - Va, Vm, Pg, Qg, y, (other)
        - nln    (AC only)
            - mu     shadow prices on nonlinear constraints, by named block
                - l  lower bounds
                    - Pmis   real power mismatch equations
                    - Qmis   reactive power mismatch equations
                    - Sf     flow limits at "from" end of branches
                    - St     flow limits at "to" end of branches
                - u  upper bounds
                    - Pmis, Qmis, Sf, St
        - lin
            - mu     shadow prices on linear constraints, by named block
                - l  lower bounds
                    - Pmis   real power mistmatch equations (DC only)
                    - Pf     flow limits at "from" end of branches (DC only)
                    - Pt     flow limits at "to" end of branches (DC only)
                    - PQh    upper portion of gen PQ-capability curve (AC only)
                    - PQl    lower portion of gen PQ-capability curve (AC only)
                    - vl     constant power factor constraint for loads
                    - ycon   basin constraints for CCV for pwl costs
                    - (other) any user-defined constraint blocks
                - u  upper bounds
                    - Pmis, Pf, Pt, PQh, PQl, vl, ycon, (other)
        - cost       user-defined cost values, by named block

    Author
    ------
    Ray Zimmerman (PSERC Cornell)

    Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad Autonoma de Manizales)
    """
    # ----- initialization -----
    t0, _ = elapsed()  # start timer

    # process input arguments
    ppc, ppopt = opf_args2(*args)

    # add zero columns to bus, gen, branch for multipliers, etc if needed
    nb = np.shape(ppc['bus'])[0]  # number of buses
    nl = np.shape(ppc['branch'])[0]  # number of branches
    ng = np.shape(ppc['gen'])[0]  # number of dispatchable injections
    if np.shape(ppc['bus'])[1] < IDX.bus.MU_VMIN + 1:
        ppc['bus'] = np.c_[ppc['bus'], np.zeros((nb, IDX.bus.MU_VMIN + 1 - np.shape(ppc['bus'])[1]))]

    if np.shape(ppc['gen'])[1] < IDX.gen.MU_QMIN + 1:
        ppc['gen'] = np.c_[ppc['gen'], np.zeros((ng, IDX.gen.MU_QMIN + 1 - np.shape(ppc['gen'])[1]))]

    if np.shape(ppc['branch'])[1] < IDX.branch.MU_ANGMAX + 1:
        ppc['branch'] = np.c_[ppc['branch'], np.zeros((nl, IDX.branch.MU_ANGMAX + 1 - np.shape(ppc['branch'])[1]))]

    # -----  convert to internal numbering, remove out-of-service stuff  -----
    ppc = opfcn.ext2int(ppc)

    # -----  construct OPF model object  -----
    om = opf_setup(ppc, ppopt)

    # -----  execute the OPF  -----
    results, success, raw = opf_execute(om, ppopt)

    # -----  revert to original ordering, including out-of-service stuff  -----
    results = opfcn.int2ext(results)

    # zero out result fields of out-of-service gens & branches
    if len(results['order']['gen']['status']['off']) > 0:
        results['gen'][
            np.ix_(
                results['order']['gen']['status']['off'],
                [IDX.gen.PG,
                 IDX.gen.QG,
                 IDX.gen.MU_PMAX,
                 IDX.gen.MU_PMIN])] = 0

    if len(results['order']['branch']['status']['off']) > 0:
        results['branch'][
            np.ix_(
                results['order']['branch']['status']['off'],
                [IDX.branch.PF,
                 IDX.branch.QF,
                 IDX.branch.PT,
                 IDX.branch.QT,
                 IDX.branch.MU_SF,
                 IDX.branch.MU_ST,
                 IDX.branch.MU_ANGMIN,
                 IDX.branch.MU_ANGMAX])] = 0

    # -----  finish preparing output  -----
    _, results['et'] = elapsed(t0)

    results['success'] = success
    results['raw'] = raw

    return results


def opf_setup(ppc, ppopt):
    """Constructs an OPF model object from a PYPOWER case dict.

    Assumes that ppc is a PYPOWER case dict with internal indexing,
    all equipment in-service, etc.

    @see: L{opf}, L{ext2int}, L{opf_execute}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    # options
    alg = ppopt['OPF_ALG']

    # data dimensions
    nb = ppc['bus'].shape[0]  # number of buses
    nl = ppc['branch'].shape[0]  # number of branches
    ng = ppc['gen'].shape[0]  # number of dispatchable injections
    if 'A' in ppc:
        nusr = ppc['A'].shape[0]  # number of linear user constraints
    else:
        nusr = 0

    if 'N' in ppc:
        nw = ppc['N'].shape[0]  # number of general cost vars, w
    else:
        nw = 0

    # convert single-block piecewise-linear costs into linear polynomial cost
    pwl1 = find((ppc['gencost'][:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR)
                & (ppc['gencost'][:, IDX.cost.NCOST] == 2))
    # p1 = np.array([])
    if len(pwl1) > 0:
        x0 = ppc['gencost'][pwl1, IDX.cost.COST]
        y0 = ppc['gencost'][pwl1, IDX.cost.COST + 1]
        x1 = ppc['gencost'][pwl1, IDX.cost.COST + 2]
        y1 = ppc['gencost'][pwl1, IDX.cost.COST + 3]
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
        ppc['gencost'][pwl1, IDX.cost.MODEL] = IDX.cost.POLYNOMIAL
        ppc['gencost'][pwl1, IDX.cost.NCOST] = 2
        ppc['gencost'][pwl1, IDX.cost.COST:IDX.cost.COST + 2] = np.r_[m, b]

    # create (read-only) copies of individual fields for convenience
    baseMVA, bus, gen, branch, gencost, _, lbu, ubu, ppopt, \
        _, fparm, H, Cw, z0, zl, zu, userfcn, _ = opf_args(ppc, ppopt)

    # warn if there is more than one reference bus
    refs = find(bus[:, IDX.bus.BUS_TYPE] == IDX.bus.REF)
    if len(refs) > 1:
        errstr = 'opf_setup: Warning: Multiple reference buses.\n' + \
            '           For a system with islands, a reference bus in each island\n' + \
            '           may help convergence, but in a fully connected system such\n' + \
            '           a situation is probably not reasonable.\n\n'
        logger.info(errstr)

    # set up initial variables and bounds
    gbus = gen[:, IDX.gen.GEN_BUS].astype(int)
    Va = bus[:, IDX.bus.VA] * deg2rad
    Vm = bus[:, IDX.bus.VM].copy()
    Vm[gbus] = gen[:, IDX.gen.VG]  # buses with gens, init Vm from gen data
    Pg = gen[:, IDX.gen.PG] / baseMVA
    Qg = gen[:, IDX.gen.QG] / baseMVA
    Pmin = gen[:, IDX.gen.PMIN] / baseMVA
    Pmax = gen[:, IDX.gen.PMAX] / baseMVA
    Qmin = gen[:, IDX.gen.QMIN] / baseMVA
    Qmax = gen[:, IDX.gen.QMAX] / baseMVA

    # AC model with more problem dimensions
    nv = nb  # number of voltage magnitude vars
    nq = ng  # number of Qg vars
    q1 = ng  # index of 1st Qg column in Ay

    # dispatchable load, constant power factor constraints
    Avl, lvl, uvl, _ = makeAvl(baseMVA, gen)

    # generator PQ capability curve constraints
    Apqh, ubpqh, Apql, ubpql, Apqdata = makeApq(baseMVA, gen)

    user_vars = ['Va', 'Vm', 'Pg', 'Qg']
    ycon_vars = ['Pg', 'Qg', 'y']

    # voltage angle reference constraints
    Vau = np.Inf * np.ones(nb)
    Val = -Vau
    Vau[refs] = Va[refs]
    Val[refs] = Va[refs]

    # branch voltage angle difference limits
    Aang, lang, uang, iang = makeAang(baseMVA, branch, nb, ppopt)

    # basin constraints for piece-wise linear gen cost variables
    if alg == 545 or alg == 550:  # SC-PDIPM or TRALM, no CCV cost vars
        ny = 0
        Ay = None
        by = np.array([])
    else:
        ipwl = find(gencost[:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR)  # piece-wise linear costs
        ny = ipwl.shape[0]  # number of piece-wise linear cost vars
        Ay, by = makeAy(baseMVA, ng, gencost, 1, q1, 1+ng+nq)

    if np.any((gencost[:, IDX.cost.MODEL] != IDX.cost.POLYNOMIAL) &
              (gencost[:, IDX.cost.MODEL] != IDX.cost.PW_LINEAR)):
        logger.debug('opf_setup: some generator cost rows have invalid MODEL value\n')

    # more problem dimensions
    nx = nb+nv + ng+nq  # number of standard OPF control variables
    if nusr:
        nz = ppc['A'].shape[1] - nx  # number of user z variables
        if nz < 0:
            logger.debug('opf_setup: user supplied A matrix must have at least %d columns.\n' % nx)
    else:
        nz = 0  # number of user z variables
        if nw:  # still need to check number of columns of N
            if ppc['N'].shape[1] != nx:
                logger.debug('opf_setup: user supplied N matrix must have %d columns.\n' % nx)

    # construct OPF model object
    om = opf_model(ppc)
    if len(pwl1) > 0:
        om.userdata('pwl1', pwl1)

    om.userdata('Apqdata', Apqdata)
    om.userdata('iang', iang)
    om.add_vars('Va', nb, Va, Val, Vau)
    om.add_vars('Vm', nb, Vm, bus[:, IDX.bus.VMIN], bus[:, IDX.bus.VMAX])
    om.add_vars('Pg', ng, Pg, Pmin, Pmax)
    om.add_vars('Qg', ng, Qg, Qmin, Qmax)
    om.add_constraints('Pmis', nb, 'nonlinear')
    om.add_constraints('Qmis', nb, 'nonlinear')
    om.add_constraints('Sf', nl, 'nonlinear')
    om.add_constraints('St', nl, 'nonlinear')
    om.add_constraints('PQh', Apqh, np.array([]), ubpqh, ['Pg', 'Qg'])  # npqh
    om.add_constraints('PQl', Apql, np.array([]), ubpql, ['Pg', 'Qg'])  # npql
    om.add_constraints('vl',  Avl, lvl, uvl,   ['Pg', 'Qg'])  # nvl
    om.add_constraints('ang', Aang, lang, uang, ['Va'])  # nang

    # y vars, constraints for piece-wise linear gen costs
    if ny > 0:
        om.add_vars('y', ny)
        om.add_constraints('ycon', Ay, np.array([]), by, ycon_vars)  # ncony

    # add user vars, constraints and costs (as specified via A, ..., N, ...)
    if nz > 0:
        om.add_vars('z', nz, z0, zl, zu)
        user_vars.append('z')

    if nusr:
        om.add_constraints('usr', ppc['A'], lbu, ubu, user_vars)  # nusr

    if nw:
        user_cost = {}
        user_cost['N'] = ppc['N']
        user_cost['Cw'] = Cw
        if len(fparm) > 0:
            user_cost['dd'] = fparm[:, 0]
            user_cost['rh'] = fparm[:, 1]
            user_cost['kk'] = fparm[:, 2]
            user_cost['mm'] = fparm[:, 3]

#        if len(H) > 0:
        user_cost['H'] = H

        om.add_costs('usr', user_cost, user_vars)

    # execute userfcn callbacks for 'formulation' stage
    opfcn.run_userfcn(userfcn, 'formulation', om)

    return om


class opf_model(object):
    """This class implements the OPF model object used to encapsulate
    a given OPF problem formulation. It allows for access to optimization
    variables, constraints and costs in named blocks, keeping track of the
    ordering and indexing of the blocks as variables, constraints and costs
    are added to the problem.

    @author: Ray Zimmerman (PSERC Cornell)
    """

    def __init__(self, ppc):
        #: PYPOWER case dict used to build the object.
        self.ppc = ppc

        #: data for optimization variable sets that make up the
        #  full optimization variable x
        self.var = {
            'idx': {
                'i1': {},  # starting index within x
                'iN': {},  # ending index within x
                'N': {}  # number of elements in this variable set
            },
            'N': 0,  # total number of elements in x
            'NS': 0,  # number of variable sets or named blocks
            'data': {  # bounds and initial value data
                'v0': {},  # vector of initial values
                'vl': {},  # vector of lower bounds
                'vu': {},  # vector of upper bounds
            },
            'order': []  # list of names for variable blocks in the order they appear in x
        }

        #: data for nonlinear constraints that make up the
        #  full set of nonlinear constraints ghn(x)
        self.nln = {
            'idx': {
                'i1': {},  # starting index within ghn(x)
                'iN': {},  # ending index within ghn(x)
                'N': {}  # number of elements in this constraint set
            },
            'N': 0,  # total number of elements in ghn(x)
            'NS': 0,  # number of nonlinear constraint sets or named blocks
            'order': []  # list of names for nonlinear constraint blocks in the order they appear in ghn(x)
        }

        #: data for linear constraints that make up the
        #  full set of linear constraints ghl(x)
        self.lin = {
            'idx': {
                'i1': {},  # starting index within ghl(x)
                'iN': {},  # ending index within ghl(x)
                'N': {}  # number of elements in this constraint set
            },
            'N': 0,  # total number of elements in ghl(x)
            'NS': 0,  # number of linear constraint sets or named blocks
            'data': {  # data for l <= A*xx <= u linear constraints
                'A': {},  # sparse linear constraint matrix
                'l': {},  # left hand side vector, bounding A*x below
                'u': {},  # right hand side vector, bounding A*x above
                'vs': {}  # cell array of variable sets that define the xx for this constraint block
            },
            'order': []  # list of names for linear constraint blocks in the order they appear in ghl(x)
        }

        #: data for user-defined costs
        self.cost = {
            'idx': {
                'i1': {},  # starting row index within full N matrix
                'iN': {},  # ending row index within full N matrix
                'N':  {}  # number of rows in this cost block in full N matrix
            },
            'N': 0,  # total number of rows in full N matrix
            'NS': 0,  # number of cost blocks
            'data': {  # data for each user-defined cost block
                'N': {},  # see help for add_costs() for details
                'H': {},  # "
                'Cw': {},  # "
                'dd': {},  # "
                'rh': {},  # "
                'kk': {},  # "
                'mm': {},  # "
                'vs': {}  # list of variable sets that define xx for this cost block, where the N for this block multiplies xx'
            },
            'order': []  # of names for cost blocks in the order they appear in the rows of the full N matrix
        }

        self.user_data = {}

    # def __repr__(self):
    #     """String representation of the object.
    #     """
    #     s = ''
    #     if self.var['NS']:
    #         s += '\n%-22s %5s %8s %8s %8s\n' % ('VARIABLES', 'name', 'i1', 'iN', 'N')
    #         s += '%-22s %5s %8s %8s %8s\n' % ('=========', '------', '-----', '-----', '------')
    #         for k in range(self.var['NS']):
    #             name = self.var['order'][k]
    #             idx = self.var['idx']
    #             s += '%15d:%12s %8d %8d %8d\n' % (k, name, idx['i1'][name], idx['iN'][name], idx['N'][name])

    #         s += '%15s%31s\n' % (('var[\'NS\'] = %d' % self.var['NS']), ('var[\'N\'] = %d' % self.var['N']))
    #         s += '\n'
    #     else:
    #         s += '%s  :  <none>\n', 'VARIABLES'

    #     if self.nln['NS']:
    #         s += '\n%-22s %5s %8s %8s %8s\n' % ('NON-LINEAR CONSTRAINTS', 'name', 'i1', 'iN', 'N')
    #         s += '%-22s %5s %8s %8s %8s\n' % ('======================', '------', '-----', '-----', '------')
    #         for k in range(self.nln['NS']):
    #             name = self.nln['order'][k]
    #             idx = self.nln['idx']
    #             s += '%15d:%12s %8d %8d %8d\n' % (k, name, idx['i1'][name], idx['iN'][name], idx['N'][name])

    #         s += '%15s%31s\n' % (('nln.NS = %d' % self.nln['NS']), ('nln.N = %d' % self.nln['N']))
    #         s += '\n'
    #     else:
    #         s += '%s  :  <none>\n', 'NON-LINEAR CONSTRAINTS'

    #     if self.lin['NS']:
    #         s += '\n%-22s %5s %8s %8s %8s\n' % ('LINEAR CONSTRAINTS', 'name', 'i1', 'iN', 'N')
    #         s += '%-22s %5s %8s %8s %8s\n' % ('==================', '------', '-----', '-----', '------')
    #         for k in range(self.lin['NS']):
    #             name = self.lin['order'][k]
    #             idx = self.lin['idx']
    #             s += '%15d:%12s %8d %8d %8d\n' % (k, name, idx['i1'][name], idx['iN'][name], idx['N'][name])

    #         s += '%15s%31s\n' % (('lin.NS = %d' % self.lin['NS']), ('lin.N = %d' % self.lin['N']))
    #         s += '\n'
    #     else:
    #         s += '%s  :  <none>\n', 'LINEAR CONSTRAINTS'

    #     if self.cost['NS']:
    #         s += '\n%-22s %5s %8s %8s %8s\n' % ('COSTS', 'name', 'i1', 'iN', 'N')
    #         s += '%-22s %5s %8s %8s %8s\n' % ('=====', '------', '-----', '-----', '------')
    #         for k in range(self.cost['NS']):
    #             name = self.cost['order'][k]
    #             idx = self.cost['idx']
    #             s += '%15d:%12s %8d %8d %8d\n' % (k, name, idx['i1'][name], idx['iN'][name], idx['N'][name])

    #         s += '%15s%31s\n' % (('cost.NS = %d' % self.cost['NS']), ('cost.N = %d' % self.cost['N']))
    #         s += '\n'
    #     else:
    #         s += '%s  :  <none>\n' % 'COSTS'

    #     #s += '  ppc = '
    #     #if len(self.ppc):
    #     #    s += '\n'
    #     #
    #     #s += str(self.ppc) + '\n'

    #     s += '  userdata = '
    #     if len(self.user_data):
    #         s += '\n'

    #     s += str(self.user_data)

    #     return s

    def add_constraints(self, name, AorN, l, u=None, varsets=None):
        """Adds a set of constraints to the model.

        Linear constraints are of the form C{l <= A * x <= u}, where
        C{x} is a vector made of of the vars specified in C{varsets} (in
        the order given). This allows the C{A} matrix to be defined only
        in terms of the relevant variables without the need to manually
        create a lot of zero columns. If C{varsets} is empty, C{x} is taken
        to be the full vector of all optimization variables. If C{l} or
        C{u} are empty, they are assumed to be appropriately sized vectors
        of C{-Inf} and C{Inf}, respectively.

        For nonlinear constraints, the 3rd argument, C{N}, is the number
        of constraints in the set. Currently, this is used internally
        by PYPOWER, but there is no way for the user to specify
        additional nonlinear constraints.
        """
        if u is None:  # nonlinear
            # prevent duplicate named constraint sets
            if name in self.nln["idx"]["N"]:
                logger.debug("opf_model.add_constraints: nonlinear constraint set named '%s' already exists\n" % name)

            # add info about this nonlinear constraint set
            self.nln["idx"]["i1"][name] = self.nln["N"]  # + 1    ## starting index
            self.nln["idx"]["iN"][name] = self.nln["N"] + AorN  # ing index
            self.nln["idx"]["N"][name] = AorN  # number of constraints

            # update number of nonlinear constraints and constraint sets
            self.nln["N"] = self.nln["idx"]["iN"][name]
            self.nln["NS"] = self.nln["NS"] + 1

            # put name in ordered list of constraint sets
#            self.nln["order"][self.nln["NS"]] = name
            self.nln["order"].append(name)
        else:  # linear
            # prevent duplicate named constraint sets
            if name in self.lin["idx"]["N"]:
                logger.debug('opf_model.add_constraints: linear constraint set named ''%s'' already exists\n' % name)

            if varsets is None:
                varsets = []

            N, M = AorN.shape
            if len(l) == 0:  # default l is -Inf
                l = -np.Inf * np.ones(N)

            if len(u) == 0:  # default u is Inf
                u = np.Inf * np.ones(N)

            if len(varsets) == 0:
                varsets = self.var["order"]

            # check sizes
            if (l.shape[0] != N) or (u.shape[0] != N):
                logger.debug('opf_model.add_constraints: sizes of A, l and u must match\n')

            nv = 0
            for k in range(len(varsets)):
                nv = nv + self.var["idx"]["N"][varsets[k]]

            if M != nv:
                logger.debug(
                    'opf_model.add_constraints: number of columns of A does not match\nnumber of variables, A is %d x %d, nv = %d\n' % (N, M, nv))

            # add info about this linear constraint set
            self.lin["idx"]["i1"][name] = self.lin["N"]  # + 1   ## starting index
            self.lin["idx"]["iN"][name] = self.lin["N"] + N  # ing index
            self.lin["idx"]["N"][name] = N  # number of constraints
            self.lin["data"]["A"][name] = AorN
            self.lin["data"]["l"][name] = l
            self.lin["data"]["u"][name] = u
            self.lin["data"]["vs"][name] = varsets

            # update number of vars and var sets
            self.lin["N"] = self.lin["idx"]["iN"][name]
            self.lin["NS"] = self.lin["NS"] + 1

            # put name in ordered list of var sets
#            self.lin["order"][self.lin["NS"]] = name
            self.lin["order"].append(name)

    def add_costs(self, name, cp, varsets):
        """Adds a set of user costs to the model.

        Adds a named block of user-defined costs to the model. Each set is
        defined by the C{cp} dict described below. All user-defined sets of
        costs are combined together into a single set of cost parameters in
        a single C{cp} dict by L{build_cost_params}. This full aggregate set of
        cost parameters can be retrieved from the model by L{get_cost_params}.

        Let C{x} refer to the vector formed by combining the specified
        C{varsets}, and C{f_u(x, cp)} be the cost at C{x} corresponding to the
        cost parameters contained in C{cp}, where C{cp} is a dict with the
        following fields::
            N      - nw x nx sparse matrix
            Cw     - nw x 1 vector
            H      - nw x nw sparse matrix (optional, all zeros by default)
            dd, mm - nw x 1 vectors (optional, all ones by default)
            rh, kk - nw x 1 vectors (optional, all zeros by default)

        These parameters are used as follows to compute C{f_u(x, CP)}::

            R  = N*x - rh

                    /  kk(i),  R(i) < -kk(i)
            K(i) = <   0,     -kk(i) <= R(i) <= kk(i)
                    \ -kk(i),  R(i) > kk(i)

            RR = R + K

            U(i) =  /  0, -kk(i) <= R(i) <= kk(i)
                    \  1, otherwise

            DDL(i) = /  1, dd(i) = 1
                     \  0, otherwise

            DDQ(i) = /  1, dd(i) = 2
                     \  0, otherwise

            Dl = diag(mm) * diag(U) * diag(DDL)
            Dq = diag(mm) * diag(U) * diag(DDQ)

            w = (Dl + Dq * diag(RR)) * RR

            f_u(x, CP) = 1/2 * w'*H*w + Cw'*w
        """
        # prevent duplicate named cost sets
        if name in self.cost["idx"]["N"]:
            logger.debug('opf_model.add_costs: cost set named \'%s\' already exists\n' % name)

        if varsets is None:
            varsets = []

        if len(varsets) == 0:
            varsets = self.var["order"]

        nw, nx = cp["N"].shape

        # check sizes
        nv = 0
        for k in range(len(varsets)):
            nv = nv + self.var["idx"]["N"][varsets[k]]

        if nx != nv:
            if nw == 0:
                cp["N"] = c_sparse(nw, nx)
            else:
                logger.debug(
                    'opf_model.add_costs: number of columns in N (%d x %d) does not match\nnumber of variables (%d)\n' %
                    (nw, nx, nv))

        if cp["Cw"].shape[0] != nw:
            logger.debug('opf_model.add_costs: number of rows of Cw (%d x %d) and N (%d x %d) must match\n' %
                         (cp["Cw"].shape[0], nw, nx))

        if 'H' in cp:
            if (cp["H"].shape[0] != nw) | (cp["H"].shape[1] != nw):
                logger.debug('opf_model.add_costs: both dimensions of H (%d x %d) must match the number of rows in N (%d x %d)\n' % (
                    cp["H"].shape, nw, nx))

        if 'dd' in cp:
            if cp["dd"].shape[0] != nw:
                logger.debug(
                    'opf_model.add_costs: number of rows of dd (%d x %d) and N (%d x %d) must match\n' %
                    (cp["dd"].shape, nw, nx))

        if 'rh' in cp:
            if cp["rh"].shape[0] != nw:
                logger.debug(
                    'opf_model.add_costs: number of rows of rh (%d x %d) and N (%d x %d) must match\n' %
                    (cp["rh"].shape, nw, nx))

        if 'kk' in cp:
            if cp["kk"].shape[0] != nw:
                logger.debug(
                    'opf_model.add_costs: number of rows of kk (%d x %d) and N (%d x %d) must match\n' %
                    (cp["kk"].shape, nw, nx))

        if 'mm' in cp:
            if cp["mm"].shape[0] != nw:
                logger.debug(
                    'opf_model.add_costs: number of rows of mm (%d x %d) and N (%d x %d) must match\n' %
                    (cp["mm"].shape, nw, nx))

        # add info about this user cost set
        self.cost["idx"]["i1"][name] = self.cost["N"]  # + 1     ## starting index
        self.cost["idx"]["iN"][name] = self.cost["N"] + nw  # ing index
        self.cost["idx"]["N"][name] = nw  # number of costs (nw)
        self.cost["data"]["N"][name] = cp["N"]
        self.cost["data"]["Cw"][name] = cp["Cw"]
        self.cost["data"]["vs"][name] = varsets
        if 'H' in cp:
            self.cost["data"]["H"][name] = cp["H"]

        if 'dd' in cp:
            self.cost["data"]["dd"]["name"] = cp["dd"]

        if 'rh' in cp:
            self.cost["data"]["rh"]["name"] = cp["rh"]

        if 'kk' in cp:
            self.cost["data"]["kk"]["name"] = cp["kk"]

        if 'mm' in cp:
            self.cost["data"]["mm"]["name"] = cp["mm"]

        # update number of vars and var sets
        self.cost["N"] = self.cost["idx"]["iN"][name]
        self.cost["NS"] = self.cost["NS"] + 1

        # put name in ordered list of var sets
        self.cost["order"].append(name)

    def add_vars(self, name, N, v0=None, vl=None, vu=None):
        """ Adds a set of variables to the model.

        Adds a set of variables to the model, where N is the number of
        variables in the set, C{v0} is the initial value of those variables,
        and C{vl} and C{vu} are the lower and upper bounds on the variables.
        The defaults for the last three arguments, which are optional,
        are for all values to be initialized to zero (C{v0 = 0}) and unbounded
        (C{VL = -Inf, VU = Inf}).
        """
        # prevent duplicate named var sets
        if name in self.var["idx"]["N"]:
            logger.debug('opf_model.add_vars: variable set named ''%s'' already exists\n' % name)

        if v0 is None or len(v0) == 0:
            v0 = np.zeros(N)  # init to zero by default

        if vl is None or len(vl) == 0:
            vl = -np.Inf * np.ones(N)  # unbounded below by default

        if vu is None or len(vu) == 0:
            vu = np.Inf * np.ones(N)  # unbounded above by default

        # add info about this var set
        self.var["idx"]["i1"][name] = self.var["N"]  # + 1   ## starting index
        self.var["idx"]["iN"][name] = self.var["N"] + N  # ing index
        self.var["idx"]["N"][name] = N  # number of vars
        self.var["data"]["v0"][name] = v0  # initial value
        self.var["data"]["vl"][name] = vl  # lower bound
        self.var["data"]["vu"][name] = vu  # upper bound

        # update number of vars and var sets
        self.var["N"] = self.var["idx"]["iN"][name]
        self.var["NS"] = self.var["NS"] + 1

        # put name in ordered list of var sets
#        self.var["order"][self.var["NS"]] = name
        self.var["order"].append(name)

    def build_cost_params(self):
        """Builds and saves the full generalized cost parameters.

        Builds the full set of cost parameters from the individual named
        sub-sets added via L{add_costs}. Skips the building process if it has
        already been done, unless a second input argument is present.

        These cost parameters can be retrieved by calling L{get_cost_params}
        and the user-defined costs evaluated by calling L{compute_cost}.
        """
        # initialize parameters
        nw = self.cost["N"]
#        nnzN = 0
#        nnzH = 0
#        for k in range(self.cost["NS"]):
#            name = self.cost["order"][k]
#            nnzN = nnzN + nnz(self.cost["data"]["N"][name])
#            if name in self.cost["data"]["H"]:
#                nnzH = nnzH + nnz(self.cost["data"]["H"][name])

        # FIXME Zero dimensional sparse matrices
        N = np.zeros((nw, self.var["N"]))
        H = np.zeros((nw, nw))  # default => no quadratic term

        Cw = np.zeros(nw)
        dd = np.ones(nw)  # default => linear
        rh = np.zeros(nw)  # default => no shift
        kk = np.zeros(nw)  # default => no dead zone
        mm = np.ones(nw)  # default => no scaling

        # fill in each piece
        for k in range(self.cost["NS"]):
            name = self.cost["order"][k]
            Nk = self.cost["data"]["N"][name]  # N for kth cost set
            i1 = self.cost["idx"]["i1"][name]  # starting row index
            iN = self.cost["idx"]["iN"][name]  # ing row index
            if self.cost["idx"]["N"][name]:  # non-zero number of rows to add
                vsl = self.cost["data"]["vs"][name]  # var set list
                kN = 0  # initialize last col of Nk used
                for v in vsl:
                    j1 = self.var["idx"]["i1"][v]  # starting column in N
                    jN = self.var["idx"]["iN"][v]  # ing column in N
                    k1 = kN  # starting column in Nk
                    kN = kN + self.var["idx"]["N"][v]  # ing column in Nk
                    N[i1:iN, j1:jN] = Nk[:, k1:kN].todense()

                Cw[i1:iN] = self.cost["data"]["Cw"][name]
                if name in self.cost["data"]["H"]:
                    H[i1:iN, i1:iN] = self.cost["data"]["H"][name].todense()

                if name in self.cost["data"]["dd"]:
                    dd[i1:iN] = self.cost["data"]["dd"][name]

                if name in self.cost["data"]["rh"]:
                    rh[i1:iN] = self.cost["data"]["rh"][name]

                if name in self.cost["data"]["kk"]:
                    kk[i1:iN] = self.cost["data"]["kk"][name]

                if name in self.cost["data"]["mm"]:
                    mm[i1:iN] = self.cost["data"]["mm"][name]

        if nw:
            N = c_sparse(N)
            H = c_sparse(H)

        # save in object
        self.cost["params"] = {
            'N': N, 'Cw': Cw, 'H': H, 'dd': dd, 'rh': rh, 'kk': kk, 'mm': mm}

    def compute_cost(self, x, name=None):
        """ Computes a user-defined cost.

        Computes the value of a user defined cost, either for all user
        defined costs or for a named set of costs. Requires calling
        L{build_cost_params} first to build the full set of parameters.

        Let C{x} be the full set of optimization variables and C{f_u(x, cp)} be
        the user-defined cost at C{x}, corresponding to the set of cost
        parameters in the C{cp} dict returned by L{get_cost_params}, where
        C{cp} is a dict with the following fields::
            N      - nw x nx sparse matrix
            Cw     - nw x 1 vector
            H      - nw x nw sparse matrix (optional, all zeros by default)
            dd, mm - nw x 1 vectors (optional, all ones by default)
            rh, kk - nw x 1 vectors (optional, all zeros by default)

        These parameters are used as follows to compute C{f_u(x, cp)}::

            R  = N*x - rh

                    /  kk(i),  R(i) < -kk(i)
            K(i) = <   0,     -kk(i) <= R(i) <= kk(i)
                    \ -kk(i),  R(i) > kk(i)

            RR = R + K

            U(i) =  /  0, -kk(i) <= R(i) <= kk(i)
                    \  1, otherwise

            DDL(i) = /  1, dd(i) = 1
                     \  0, otherwise

            DDQ(i) = /  1, dd(i) = 2
                     \  0, otherwise

            Dl = diag(mm) * diag(U) * diag(DDL)
            Dq = diag(mm) * diag(U) * diag(DDQ)

            w = (Dl + Dq * diag(RR)) * RR

            F_U(X, CP) = 1/2 * w'*H*w + Cw'*w
        """
        if name is None:
            cp = self.get_cost_params()
        else:
            cp = self.get_cost_params(name)

        N, Cw, H, dd, rh, kk, mm = \
            cp["N"], cp["Cw"], cp["H"], cp["dd"], cp["rh"], cp["kk"], cp["mm"]
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
        kbar = c_sparse((np.r_[np.ones(len(iLT)),
                               np.zeros(len(iEQ)),
                               -np.ones(len(iGT))], (iND, iND)), (nw, nw)) * kk
        rr = r + kbar  # apply non-dead zone shift
        M = c_sparse((mm[iND], (iND, iND)), (nw, nw))  # dead zone or scale
        diagrr = c_sparse((rr, (np.arange(nw), np.arange(nw))), (nw, nw))

        # linear rows multiplied by rr(i), quadratic rows by rr(i)^2
        w = M * (LL + QQ * diagrr) * rr

        f = np.dot(w * H, w) / 2 + np.dot(Cw, w)

        return f

    def get_cost_params(self, name=None):
        """Returns the cost parameter struct for user-defined costs.

        Requires calling L{build_cost_params} first to build the full set of
        parameters. Returns the full cost parameter struct for all user-defined
        costs that incorporates all of the named cost sets added via
        L{add_costs}, or, if a name is provided it returns the cost dict
        corresponding to the named set of cost rows (C{N} still has full number
        of columns).

        The cost parameters are returned in a dict with the following fields::
            N      - nw x nx sparse matrix
            Cw     - nw x 1 vector
            H      - nw x nw sparse matrix (optional, all zeros by default)
            dd, mm - nw x 1 vectors (optional, all ones by default)
            rh, kk - nw x 1 vectors (optional, all zeros by default)
        """
        if not 'params' in self.cost:
            logger.debug('opf_model.get_cost_params: must call build_cost_params first\n')

        cp = self.cost["params"]

        if name is not None:
            if self.getN('cost', name):
                idx = np.arange(self.cost["idx"]["i1"][name], self.cost["idx"]["iN"][name])
                nwa = self.cost["idx"]["i1"][name]
                nwb = self.cost["idx"]["iN"][name]
                cp["N"] = cp["N"][idx, :]
                cp["Cw"] = cp["Cw"][idx]
                cp["H"] = cp["H"][nwa:nwb, nwa:nwb]
                cp["dd"] = cp["dd"][idx]
                cp["rh"] = cp["rh"][idx]
                cp["kk"] = cp["kk"][idx]
                cp["mm"] = cp["mm"][idx]

        return cp

    def get_idx(self):
        """ Returns the idx struct for vars, lin/nln constraints, costs.

        Returns a structure for each with the beginning and ending
        index value and the number of elements for each named block.
        The 'i1' field (that's a one) is a dict with all of the
        starting indices, 'iN' contains all the ending indices and
        'N' contains all the sizes. Each is a dict whose keys are
        the named blocks.

        Examples::
            [vv, ll, nn] = get_idx(om)

        For a variable block named 'z' we have::
                vv['i1']['z'] - starting index for 'z' in optimization vector x
                vv['iN']['z'] - ending index for 'z' in optimization vector x
                vv["N"]    - number of elements in 'z'

        To extract a 'z' variable from x::
                z = x(vv['i1']['z']:vv['iN']['z'])

        To extract the multipliers on a linear constraint set
        named 'foo', where mu_l and mu_u are the full set of
        linear constraint multipliers::
                mu_l_foo = mu_l(ll['i1']['foo']:ll['iN']['foo'])
                mu_u_foo = mu_u(ll['i1']['foo']:ll['iN']['foo'])

        The number of nonlinear constraints in a set named 'bar'::
                nbar = nn["N"].bar
        (note: the following is preferable ::
                nbar = getN(om, 'nln', 'bar')
        ... if you haven't already called L{get_idx} to get C{nn}.)
        """
        vv = self.var["idx"]
        ll = self.lin["idx"]
        nn = self.nln["idx"]
        cc = self.cost["idx"]

        return vv, ll, nn, cc

    def get_ppc(self):
        """Returns the PYPOWER case dict.
        """
        return self.ppc

    def getN(self, selector, name=None):
        """Returns the number of variables, constraints or cost rows.

        Returns either the total number of variables/constraints/cost rows
        or the number corresponding to a specified named block.

        Examples::
            N = getN(om, 'var')         : total number of variables
            N = getN(om, 'lin')         : total number of linear constraints
            N = getN(om, 'nln')         : total number of nonlinear constraints
            N = getN(om, 'cost')        : total number of cost rows (in N)
            N = getN(om, 'var', name)   : number of variables in named set
            N = getN(om, 'lin', name)   : number of linear constraints in named set
            N = getN(om, 'nln', name)   : number of nonlinear cons. in named set
            N = getN(om, 'cost', name)  : number of cost rows (in N) in named set
        """
        if name is None:
            N = getattr(self, selector)["N"]
        else:
            if name in getattr(self, selector)["idx"]["N"]:
                N = getattr(self, selector)["idx"]["N"][name]
            else:
                N = 0
        return N

    def getv(self, name=None):
        """Returns initial value, lower bound and upper bound for opt variables.

        Returns the initial value, lower bound and upper bound for the full
        optimization variable vector, or for a specific named variable set.

        Examples::
            x, xmin, xmax = getv(om)
            Pg, Pmin, Pmax = getv(om, 'Pg')
        """
        if name is None:
            v0 = np.array([])
            vl = np.array([])
            vu = np.array([])
            for k in range(self.var["NS"]):
                name = self.var["order"][k]
                v0 = np.r_[v0, self.var["data"]["v0"][name]]
                vl = np.r_[vl, self.var["data"]["vl"][name]]
                vu = np.r_[vu, self.var["data"]["vu"][name]]
        else:
            if name in self.var["idx"]["N"]:
                v0 = self.var["data"]["v0"][name]
                vl = self.var["data"]["vl"][name]
                vu = self.var["data"]["vu"][name]
            else:
                v0 = np.array([])
                vl = np.array([])
                vu = np.array([])

        return v0, vl, vu

    def linear_constraints(self):
        """Builds and returns the full set of linear constraints.

        Builds the full set of linear constraints based on those added by
        L{add_constraints}::

            L <= A * x <= U
        """

        # initialize A, l and u
#        nnzA = 0
#        for k in range(self.lin["NS"]):
#            nnzA = nnzA + nnz(self.lin["data"].A.(self.lin.order{k}))

        if self.lin["N"]:
            A = sp.sparse.lil_matrix((self.lin["N"], self.var["N"]))
            u = np.Inf * np.ones(self.lin["N"])
            l = -u
        else:
            A = None
            u = np.array([])
            l = np.array([])

            return A, l, u

        # fill in each piece
        for k in range(self.lin["NS"]):
            name = self.lin["order"][k]
            N = self.lin["idx"]["N"][name]
            if N:  # non-zero number of rows to add
                Ak = self.lin["data"]["A"][name]  # A for kth linear constrain set
                i1 = self.lin["idx"]["i1"][name]  # starting row index
                iN = self.lin["idx"]["iN"][name]  # ing row index
                vsl = self.lin["data"]["vs"][name]  # var set list
                kN = 0  # initialize last col of Ak used
                # FIXME: Sparse matrix with fancy indexing
                Ai = np.zeros((N, self.var["N"]))
                for v in vsl:
                    j1 = self.var["idx"]["i1"][v]  # starting column in A
                    jN = self.var["idx"]["iN"][v]  # ing column in A
                    k1 = kN  # starting column in Ak
                    kN = kN + self.var["idx"]["N"][v]  # ing column in Ak
                    Ai[:, j1:jN] = Ak[:, k1:kN].todense()

                A[i1:iN, :] = Ai

                l[i1:iN] = self.lin["data"]["l"][name]
                u[i1:iN] = self.lin["data"]["u"][name]

        return A.tocsr(), l, u

    def userdata(self, name, val=None):
        """Used to save or retrieve values of user data.

        This function allows the user to save any arbitrary data in the object
        for later use. This can be useful when using a user function to add
        variables, constraints, costs, etc. For example, suppose some special
        indexing is constructed when adding some variables or constraints.
        This indexing data can be stored and used later to "unpack" the results
        of the solved case.
        """
        if val is not None:
            self.user_data[name] = val
            return self
        else:
            if name in self.user_data:
                return self.user_data[name]
            else:
                return np.array([])

def opf_execute(om, ppopt):
    """
    Executes the OPF specified by an OPF model object.

    C{results} are returned with internal indexing, all equipment
    in-service, etc.

    @see: L{opf}, L{opf_setup}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # -----  setup  -----
    # options
    alg = ppopt['OPF_ALG']

    # build user-defined costs
    om.build_cost_params()

    # get indexing
    vv, ll, nn, _ = om.get_idx()

    # -----  run AC OPF solver  -----
    # if OPF_ALG not set, choose best available option
    if alg == 0:
        alg = 560  # MIPS

    # update deprecated algorithm codes to new, generalized formulation equivalents
    if alg == 100 | alg == 200:  # CONSTR
        alg = 300
    elif alg == 120 | alg == 220:  # dense LP
        alg = 320
    elif alg == 140 | alg == 240:  # sparse (relaxed) LP
        alg = 340
    elif alg == 160 | alg == 260:  # sparse (full) LP
        alg = 360

    ppopt['OPF_ALG_POLY'] = alg

    # run specific AC OPF solver
    if alg == 560 or alg == 565:  # PIPS
        results, success, raw = pipsopf_solver(om, ppopt)
    elif alg == 580:  # IPOPT
        try:
            __import__('pyipopt')
            results, success, raw = ipoptopf_solver(om, ppopt)
        except ImportError:
            raise ImportError('OPF_ALG %d requires IPOPT '
                              '(see https://projects.coin-or.org/Ipopt/)' %
                              alg)
    else:
        logger.debug('opf_execute: OPF_ALG %d is not a valid algorithm code\n' % alg)

    if ('output' not in raw) or ('alg' not in raw['output']):
        raw['output']['alg'] = alg

    if success:
        # copy bus voltages back to gen matrix
        results['gen'][:, IDX.gen.VG] = results['bus'][
            results['gen'][:, IDX.gen.GEN_BUS].astype(int),
            IDX.bus.VM]

        # gen PQ capability curve multipliers
        if (ll['N']['PQh'] > 0) | (ll['N']['PQl'] > 0):
            mu_PQh = results['mu']['lin']['l'][ll['i1']['PQh']:ll['iN']['PQh']
                                               ] - results['mu']['lin']['u'][ll['i1']['PQh']:ll['iN']['PQh']]
            mu_PQl = results['mu']['lin']['l'][ll['i1']['PQl']:ll['iN']['PQl']
                                               ] - results['mu']['lin']['u'][ll['i1']['PQl']:ll['iN']['PQl']]
            Apqdata = om.userdata('Apqdata')
            results['gen'] = opfcn.update_mupq(results['baseMVA'], results['gen'], mu_PQh, mu_PQl, Apqdata)

        # compute g, dg, f, df, d2f if requested by RETURN_RAW_DER = 1
        if ppopt['RETURN_RAW_DER']:
            # move from results to raw if using v4.0 of MINOPF or TSPOPF
            if 'dg' in results:
                raw = {}
                raw['dg'] = results['dg']
                raw['g'] = results['g']

            # compute g, dg, unless already done by post-v4.0 MINOPF or TSPOPF
            if 'dg' not in raw:
                ppc = om.get_ppc()
                Ybus, Yf, Yt = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])
                g, geq, dg, dgeq = opf_consfcn(results['x'], om, Ybus, Yf, Yt, ppopt)
                raw['g'] = np.r_[geq, g]
                raw['dg'] = np.r_[dgeq.T, dg.T]  # true Jacobian organization

            # compute df, d2f
            _, df, d2f = opf_costfcn(results['x'], om, True)
            raw['df'] = df
            raw['d2f'] = d2f

        # delete g and dg fieldsfrom results if using v4.0 of MINOPF or TSPOPF
        if 'dg' in results:
            del results['dg']
            del results['g']

        # angle limit constraint multipliers
        if ll['N']['ang'] > 0:
            iang = om.userdata('iang')
            results['branch'][
                iang, IDX.branch.MU_ANGMIN] = results['mu']['lin']['l'][
                ll['i1']['ang']: ll['iN']['ang']] * deg2rad
            results['branch'][
                iang, IDX.branch.MU_ANGMAX] = results['mu']['lin']['u'][
                ll['i1']['ang']: ll['iN']['ang']] * deg2rad
    else:
        # assign empty g, dg, f, df, d2f if requested by RETURN_RAW_DER = 1
        raw['dg'] = np.array([])
        raw['g'] = np.array([])
        raw['df'] = np.array([])
        raw['d2f'] = np.array([])

    # assign values and limit shadow prices for variables
    if om.var['order']:
        results['var'] = {'val': {}, 'mu': {'l': {}, 'u': {}}}
    for name in om.var['order']:
        if om.getN('var', name):
            idx = np.arange(vv['i1'][name], vv['iN'][name])
            results['var']['val'][name] = results['x'][idx]
            results['var']['mu']['l'][name] = results['mu']['var']['l'][idx]
            results['var']['mu']['u'][name] = results['mu']['var']['u'][idx]

    # assign shadow prices for linear constraints
    if om.lin['order']:
        results['lin'] = {'mu': {'l': {}, 'u': {}}}
    for name in om.lin['order']:
        if om.getN('lin', name):
            idx = np.arange(ll['i1'][name], ll['iN'][name])
            results['lin']['mu']['l'][name] = results['mu']['lin']['l'][idx]
            results['lin']['mu']['u'][name] = results['mu']['lin']['u'][idx]

    # assign shadow prices for nonlinear constraints
    if om.nln['order']:
        results['nln'] = {'mu': {'l': {}, 'u': {}}}
    for name in om.nln['order']:
        if om.getN('nln', name):
            idx = np.arange(nn['i1'][name], nn['iN'][name])
            results['nln']['mu']['l'][name] = results['mu']['nln']['l'][idx]
            results['nln']['mu']['u'][name] = results['mu']['nln']['u'][idx]

    # assign values for components of user cost
    if om.cost['order']:
        results['cost'] = {}
    for name in om.cost['order']:
        if om.getN('cost', name):
            results['cost'][name] = om.compute_cost(results['x'], name)

    # if single-block PWL costs were converted to POLY, insert dummy y into x
    # Note: The "y" portion of x will be nonsense, but everything should at
    # least be in the expected locations.
    pwl1 = om.userdata('pwl1')
    if (len(pwl1) > 0) and (alg != 545) and (alg != 550):
        # get indexing
        vv, _, _, _ = om.get_idx()
        nx = vv['iN']['Qg']

        y = np.zeros(len(pwl1))
        raw['xr'] = np.r_[raw['xr'][:nx], y, raw['xr'][nx:]]
        results['x'] = np.r_[results['x'][:nx], y, results['x'][nx:]]

    return results, success, raw


def opf_args(*args):
    """Parses and initializes OPF input arguments.

    Returns the full set of initialized OPF input arguments, filling in
    default values for missing arguments. See Examples below for the
    possible calling syntax options.

    Input arguments options::

        opf_args(ppc)
        opf_args(ppc, ppopt)
        opf_args(ppc, userfcn, ppopt)
        opf_args(ppc, A, l, u)
        opf_args(ppc, A, l, u, ppopt)
        opf_args(ppc, A, l, u, ppopt, N, fparm, H, Cw)
        opf_args(ppc, A, l, u, ppopt, N, fparm, H, Cw, z0, zl, zu)

        opf_args(baseMVA, bus, gen, branch, areas, gencost)
        opf_args(baseMVA, bus, gen, branch, areas, gencost, ppopt)
        opf_args(baseMVA, bus, gen, branch, areas, gencost, userfcn, ppopt)
        opf_args(baseMVA, bus, gen, branch, areas, gencost, A, l, u)
        opf_args(baseMVA, bus, gen, branch, areas, gencost, A, l, u, ppopt)
        opf_args(baseMVA, bus, gen, branch, areas, gencost, A, l, u, ...
                                    ppopt, N, fparm, H, Cw)
        opf_args(baseMVA, bus, gen, branch, areas, gencost, A, l, u, ...
                                    ppopt, N, fparm, H, Cw, z0, zl, zu)

    The data for the problem can be specified in one of three ways:
      1. a string (ppc) containing the file name of a PYPOWER case
      which defines the data matrices baseMVA, bus, gen, branch, and
      gencost (areas is not used at all, it is only included for
      backward compatibility of the API).
      2. a dict (ppc) containing the data matrices as fields.
      3. the individual data matrices themselves.

    The optional user parameters for user constraints (C{A, l, u}), user costs
    (C{N, fparm, H, Cw}), user variable initializer (z0), and user variable
    limits (C{zl, zu}) can also be specified as fields in a case dict,
    either passed in directly or defined in a case file referenced by name.

    When specified, C{A, l, u} represent additional linear constraints on the
    optimization variables, C{l <= A*[x z] <= u}. If the user specifies an C{A}
    matrix that has more columns than the number of "C{x}" (OPF) variables,
    then there are extra linearly constrained "C{z}" variables. For an
    explanation of the formulation used and instructions for forming the
    C{A} matrix, see the MATPOWER manual.

    A generalized cost on all variables can be applied if input arguments
    C{N}, C{fparm}, C{H} and C{Cw} are specified.  First, a linear
    transformation of the optimization variables is defined by means of
    C{r = N * [x z]}. Then, to each element of r a function is applied as
    encoded in the C{fparm} matrix (see Matpower manual). If the resulting
    vector is named C{w}, then C{H} and C{Cw} define a quadratic cost on
    C{w}: C{(1/2)*w'*H*w + Cw * w}.
    C{H} and C{N} should be sparse matrices and C{H} should also be symmetric.

    The optional C{ppopt} vector specifies PYPOWER options. See L{ppoption}
    for details and default values.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
#    nargin = len([arg for arg in [baseMVA, bus, gen, branch, areas, gencost,
#                                  Au, lbu, ubu, ppopt, N, fparm, H, Cw,
#                                  z0, zl, zu] if arg is not None])
    nargin = len(args)
    userfcn = np.array([])

    # parsing filename or dict
    if isinstance(args[0], str) or isinstance(args[0], dict):
        if nargin in [1, 2, 3, 4, 5, 9, 12]:
            casefile = args[0]
            if nargin == 12:
                baseMVA, bus, gen, branch, areas, gencost, Au, lbu, ubu, ppopt, N, fparm = args
            elif nargin == 9:
                baseMVA, bus, gen, branch, areas, gencost, Au, lbu, ubu = args
            elif nargin == 5:
                baseMVA, bus, gen, branch, areas = args
            elif nargin == 4:
                baseMVA, bus, gen, branch = args
            elif nargin == 3:
                baseMVA, bus, gen = args
                userfcn = bus
            elif nargin == 2:
                baseMVA, bus = args
            elif nargin == 1:
                pass  # Use default values for all variables
            else:
                logger.debug('opf_args: Incorrect input arg order, number or type\n')

            # Set default values for variables if they are not provided
            zu = np.array([]) if nargin in [1, 2, 3, 4, 5, 9, 12] else zu
            zl = np.array([]) if nargin in [1, 2, 3, 4, 5, 9, 12] else zl
            z0 = np.array([]) if nargin in [1, 2, 3, 4, 5, 9, 12] else z0
            Cw = np.array([]) if nargin in [1, 2, 3, 4, 5, 9, 12] else Cw
            H = None if nargin in [1, 2, 3, 4, 5, 9, 12] else H
            fparm = np.array([]) if nargin in [1, 2, 3, 4, 5, 9, 12] else fparm
            N = None if nargin in [1, 2, 3, 4, 5, 9, 12] else N
            ppopt = ppoption() if nargin in [1, 2, 3, 4, 5, 9, 12] else ppopt
            ubu = np.array([]) if nargin in [1, 2, 3, 4, 5, 9, 12] else ubu
            lbu = np.array([]) if nargin in [1, 2, 3, 4, 5, 9, 12] else lbu
            Au = None if nargin in [1, 2, 3, 4, 5, 9, 12] else Au
        else:
            logger.debug('opf_args: Incorrect input arg order, number or type\n')

        ppc = loadcase(casefile)
        baseMVA, bus, gen, branch, gencost = \
            ppc['baseMVA'], ppc['bus'], ppc['gen'], ppc['branch'], ppc['gencost']
        if 'areas' in ppc:
            areas = ppc['areas']
        else:
            areas = np.array([])
        if Au is None and 'A' in ppc:
            Au, lbu, ubu = ppc["A"], ppc["l"], ppc["u"]
        if N is None and 'N' in ppc:  # these two must go together
            N, Cw = ppc["N"], ppc["Cw"]
        if H is None and 'H' in ppc:  # will default to zeros
            H = ppc["H"]
        if (fparm is None or len(fparm) == 0) and 'fparm' in ppc:  # will default to [1 0 0 1]
            fparm = ppc["fparm"]
        if (z0 is None or len(z0) == 0) and 'z0' in ppc:
            z0 = ppc["z0"]
        if (zl is None or len(zl) == 0) and 'zl' in ppc:
            zl = ppc["zl"]
        if (zu is None or len(zu) == 0) and 'zu' in ppc:
            zu = ppc["zu"]
        if (userfcn is None or len(userfcn) == 0) and 'userfcn' in ppc:
            userfcn = ppc['userfcn']
    else:  # passing individual data matrices
        # ----opf(baseMVA, bus, gen, branch, areas, gencost,      Au, lbu, ubu, ppopt, N, fparm, H, Cw, z0, zl, zu)
        # 17  opf(baseMVA, bus, gen, branch, areas, gencost,      Au, lbu, ubu, ppopt, N, fparm, H, Cw, z0, zl, zu)
        # 14  opf(baseMVA, bus, gen, branch, areas, gencost,      Au, lbu, ubu, ppopt, N, fparm, H, Cw)
        # 10  opf(baseMVA, bus, gen, branch, areas, gencost,      Au, lbu, ubu, ppopt)
        # 9   opf(baseMVA, bus, gen, branch, areas, gencost,      Au, lbu, ubu)
        # 8   opf(baseMVA, bus, gen, branch, areas, gencost, userfcn, ppopt)
        # 7   opf(baseMVA, bus, gen, branch, areas, gencost, ppopt)
        # 6   opf(baseMVA, bus, gen, branch, areas, gencost)
        if nargin in [6, 7, 8, 9, 10, 14, 17]:
            if nargin == 17:
                baseMVA, bus, gen, branch, areas, gencost, Au, lbu, ubu, ppopt,  N, fparm, H, Cw, z0, zl, zu = args
            elif nargin == 14:
                baseMVA, bus, gen, branch, areas, gencost, Au, lbu, ubu, ppopt,  N, fparm, H, Cw = args
                zu = np.array([])
                zl = np.array([])
                z0 = np.array([])
            elif nargin == 10:
                baseMVA, bus, gen, branch, areas, gencost, Au, lbu, ubu, ppopt = args
                zu = np.array([])
                zl = np.array([])
                z0 = np.array([])
                Cw = np.array([])
                H = None
                fparm = np.array([])
                N = None
            elif nargin == 9:
                baseMVA, bus, gen, branch, areas, gencost, Au, lbu, ubu = args
                zu = np.array([])
                zl = np.array([])
                z0 = np.array([])
                Cw = np.array([])
                H = None
                fparm = np.array([])
                N = None
                ppopt = ppoption()
            elif nargin == 8:
                baseMVA, bus, gen, branch, areas, gencost, userfcn, ppopt = args
                zu = np.array([])
                zl = np.array([])
                z0 = np.array([])
                Cw = np.array([])
                H = None
                fparm = np.array([])
                N = None
                ubu = np.array([])
                lbu = np.array([])
                Au = None
            elif nargin == 7:
                baseMVA, bus, gen, branch, areas, gencost, ppopt = args
                zu = np.array([])
                zl = np.array([])
                z0 = np.array([])
                Cw = np.array([])
                H = None
                fparm = np.array([])
                N = None
                ubu = np.array([])
                lbu = np.array([])
                Au = None
            elif nargin == 6:
                baseMVA, bus, gen, branch, areas, gencost = args
                zu = np.array([])
                zl = np.array([])
                z0 = np.array([])
                Cw = np.array([])
                H = None
                fparm = np.array([])
                N = None
                ppopt = ppoption()
                ubu = np.array([])
                lbu = np.array([])
                Au = None
        else:
            logger.debug('opf_args: Incorrect input arg order, number or type\n')

    if N is not None:
        nw = N.shape[0]
    else:
        nw = 0

    if nw:
        if Cw.shape[0] != nw:
            logger.debug('opf_args.m: dimension mismatch between N and Cw in '
                         'generalized cost parameters\n')
        if len(fparm) > 0 and fparm.shape[0] != nw:
            logger.debug('opf_args.m: dimension mismatch between N and fparm '
                         'in generalized cost parameters\n')
        if (H is not None) and (H.shape[0] != nw | H.shape[0] != nw):
            logger.debug('opf_args.m: dimension mismatch between N and H in '
                         'generalized cost parameters\n')
        if Au is not None:
            if Au.shape[0] > 0 and N.shape[1] != Au.shape[1]:
                logger.debug('opf_args.m: A and N must have the same number '
                             'of columns\n')
        # make sure N and H are sparse
        if not sp.issparse(N):
            logger.debug('opf_args.m: N must be sparse in generalized cost '
                         'parameters\n')
        if not sp.issparse(H):
            logger.debug('opf_args.m: H must be sparse in generalized cost parameters\n')

    if Au is not None and not sp.issparse(Au):
        logger.debug('opf_args.m: Au must be sparse\n')
    if ppopt == None or len(ppopt) == 0:
        ppopt = ppoption()

    return baseMVA, bus, gen, branch, gencost, Au, lbu, ubu, \
        ppopt, N, fparm, H, Cw, z0, zl, zu, userfcn, areas


def opf_args2(*args):
    """
    Parses and initializes OPF input arguments.
    """
    baseMVA, bus, gen, branch, gencost, Au, lbu, ubu, \
        ppopt, N, fparm, H, Cw, z0, zl, zu, userfcn, areas = opf_args(*args)

    ppc = args[0] if isinstance(args[0], dict) else {}

    ppc['baseMVA'] = baseMVA
    ppc['bus'] = bus
    ppc['gen'] = gen
    ppc['branch'] = branch
    ppc['gencost'] = gencost

    if areas is not None and len(areas) > 0:
        ppc["areas"] = areas
    if lbu is not None and len(lbu) > 0:
        ppc["A"], ppc["l"], ppc["u"] = Au, lbu, ubu
    if Cw is not None and len(Cw) > 0:
        ppc["N"], ppc["Cw"] = N, Cw
        if len(fparm) > 0:
            ppc["fparm"] = fparm
        # if len(H) > 0:
        ppc["H"] = H
    if z0 is not None and len(z0) > 0:
        ppc["z0"] = z0
    if zl is not None and len(zl) > 0:
        ppc["zl"] = zl
    if zu is not None and len(zu) > 0:
        ppc["zu"] = zu
    if userfcn is not None and len(userfcn) > 0:
        ppc["userfcn"] = userfcn

    return ppc, ppopt


def uopf(*args):
    """Solves combined unit decommitment / optimal power flow.

    Solves a combined unit decommitment and optimal power flow for a single
    time period. Uses an algorithm similar to dynamic programming. It proceeds
    through a sequence of stages, where stage C{N} has C{N} generators shut
    down, starting with C{N=0}. In each stage, it forms a list of candidates
    (gens at their C{Pmin} limits) and computes the cost with each one of them
    shut down. It selects the least cost case as the starting point for the
    next stage, continuing until there are no more candidates to be shut down
    or no more improvement can be gained by shutting something down.
    If C{verbose} in ppopt (see L{ppoption} is C{true}, it prints progress
    info, if it is > 1 it prints the output of each individual opf.

    @see: L{opf}, L{runuopf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # ----- initialization -----
    t0, _ = elapsed()  # start timer

    # process input arguments
    ppc, ppopt = opf_args2(*args)

    # options
    verbose = ppopt["VERBOSE"]
    if verbose:  # turn down verbosity one level for calls to opf
        ppopt = ppoption(ppopt, VERBOSE=verbose - 1)

    # -----  do combined unit commitment/optimal power flow  -----

    # check for sum(Pmin) > total load, decommit as necessary
    on = find((ppc["gen"][:, IDX.gen.GEN_STATUS] > 0) & ~isload(ppc["gen"]))  # gens in service
    onld = find((ppc["gen"][:, IDX.gen.GEN_STATUS] > 0) & isload(ppc["gen"]))  # disp loads in serv
    load_capacity = sum(ppc["bus"][:, IDX.bus.PD]) - sum(ppc["gen"][onld, IDX.gen.PMIN])  # total load capacity
    Pmin = ppc["gen"][on, IDX.gen.PMIN]
    while sum(Pmin) > load_capacity:
        # shut down most expensive unit
        avgPmincost = opfcn.totcost(ppc["gencost"][on, :], Pmin) / Pmin
        _, i = fairmax(avgPmincost)  # pick one with max avg cost at Pmin
        i = on[i]  # convert to generator index

        if verbose:
            print('Shutting down generator %d so all Pmin limits can be satisfied.\n' % i)

        # set generation to zero
        ppc["gen"][i, [IDX.gen.PG, IDX.gen.QG, IDX.gen.GEN_STATUS]] = 0

        # update minimum gen capacity
        on = find((ppc["gen"][:, IDX.gen.GEN_STATUS] > 0) & ~isload(ppc["gen"]))  # gens in service
        Pmin = ppc["gen"][on, IDX.gen.PMIN]

    # run initial opf
    results = fopf(ppc, ppopt)

    # best case so far
    results1 = deepcopy(results)

    # best case for this stage (ie. with n gens shut down, n=0,1,2 ...)
    results0 = deepcopy(results1)
    ppc["bus"] = results0["bus"].copy()  # use these V as starting point for OPF

    while True:
        # get candidates for shutdown
        candidates = find((results0["gen"][:, IDX.gen.MU_PMIN] > 0) & (results0["gen"][:, IDX.gen.PMIN] > 0))
        if len(candidates) == 0:
            break

        # do not check for further decommitment unless we
        # see something better during this stage
        done = True

        for k in candidates:
            # start with best for this stage
            ppc["gen"] = results0["gen"].copy()

            # shut down gen k
            ppc["gen"][k, [IDX.gen.PG, IDX.gen.QG, IDX.gen.GEN_STATUS]] = 0

            # run opf
            results = fopf(ppc, ppopt)

            # something better?
            if results['success'] and (results["f"] < results1["f"]):
                results1 = deepcopy(results)
                k1 = k
                done = False  # make sure we check for further decommitment

        if done:
            # decommits at this stage did not help, so let's quit
            break
        else:
            # shutting something else down helps, so let's keep going
            if verbose:
                print('Shutting down generator %d.\n' % k1)

            results0 = deepcopy(results1)
            ppc["bus"] = results0["bus"].copy()  # use these V as starting point for OPF

    # compute elapsed time
    _, results0['et'] = elapsed(t0)

    return results0
