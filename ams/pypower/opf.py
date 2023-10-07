# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Solves an optimal power flow.
"""

from time import time

from numpy import zeros, c_, shape, ix_

from ams.pypower.idx_bus import MU_VMIN
from ams.pypower.idx_gen import PG, QG, MU_QMIN, MU_PMAX, MU_PMIN
from ams.pypower.idx_brch import PF, QF, PT, QT, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX

from ams.pypower.ext2int import ext2int
from ams.pypower.opf_args import opf_args2
from ams.pypower.opf_setup import opf_setup
from ams.pypower.opf_execute import opf_execute
from ams.pypower.int2ext import int2ext


def opf(*args):
    """
    Solve an optimal power flow, return a `results` dict.

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
    ##----- initialization -----
    t0 = time()         ## start timer

    ## process input arguments
    ppc, ppopt = opf_args2(*args)

    ## add zero columns to bus, gen, branch for multipliers, etc if needed
    nb   = shape(ppc['bus'])[0]    ## number of buses
    nl   = shape(ppc['branch'])[0] ## number of branches
    ng   = shape(ppc['gen'])[0]    ## number of dispatchable injections
    if shape(ppc['bus'])[1] < MU_VMIN + 1:
        ppc['bus'] = c_[ppc['bus'], zeros((nb, MU_VMIN + 1 - shape(ppc['bus'])[1]))]

    if shape(ppc['gen'])[1] < MU_QMIN + 1:
        ppc['gen'] = c_[ppc['gen'], zeros((ng, MU_QMIN + 1 - shape(ppc['gen'])[1]))]

    if shape(ppc['branch'])[1] < MU_ANGMAX + 1:
        ppc['branch'] = c_[ppc['branch'], zeros((nl, MU_ANGMAX + 1 - shape(ppc['branch'])[1]))]

    ##-----  convert to internal numbering, remove out-of-service stuff  -----
    ppc = ext2int(ppc)

    ##-----  construct OPF model object  -----
    om = opf_setup(ppc, ppopt)

    ##-----  execute the OPF  -----
    results, success, raw = opf_execute(om, ppopt)

    ##-----  revert to original ordering, including out-of-service stuff  -----
    results = int2ext(results)

    ## zero out result fields of out-of-service gens & branches
    if len(results['order']['gen']['status']['off']) > 0:
        results['gen'][ ix_(results['order']['gen']['status']['off'], [PG, QG, MU_PMAX, MU_PMIN]) ] = 0

    if len(results['order']['branch']['status']['off']) > 0:
        results['branch'][ ix_(results['order']['branch']['status']['off'], [PF, QF, PT, QT, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX]) ] = 0

    ##-----  finish preparing output  -----
    et = time() - t0      ## compute elapsed time

    results['et'] = et
    results['success'] = success
    results['raw'] = raw

    return results
