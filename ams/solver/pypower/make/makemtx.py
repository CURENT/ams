"""
Construct constraints for branch angle difference limits.
"""
import logging

from numpy import array, ones, zeros, r_, c_, Inf, pi, arange, nonzero
from numpy import flatnonzero as find
from scipy.sparse import csr_matrix as sparse

from ams.solver.pypower.idx_brch import F_BUS, T_BUS, ANGMIN, ANGMAX
from ams.solver.pypower.idx_gen import PC1, PC2, QC1MIN, QC1MAX, QC2MIN, QC2MAX
from ams.solver.pypower.idx_gen import QMAX, QMIN, PMAX, PC1, PC2, QC1MIN, QC1MAX, QC2MIN, QC2MAX

# --- Avl ---

from numpy import array, zeros, arange, sin, cos, arctan2, r_
from numpy import flatnonzero as find
from scipy.sparse import csr_matrix as sparse

from ams.solver.pypower.idx_gen import PG, QG, PMIN, QMIN, QMAX

from ams.solver.pypower.isload import isload
# --- END ---

# --- Ay ---
from numpy import array, diff, any, zeros, r_, flatnonzero as find
#from scipy.sparse import csr_matrix as sparse
from scipy.sparse import lil_matrix as sparse

from ams.solver.pypower.idx_cost import MODEL, PW_LINEAR, NCOST, COST
# --- END ---



logger = logging.getLogger(__name__)


def hasPQcap(gen, hilo='B'):
    """
    Checks for P-Q capability curve constraints.

    Returns a column vector of 1's and 0's. The 1's correspond to rows of
    the C{gen} matrix which correspond to generators which have defined a
    capability curve (with sloped upper and/or lower bound on Q) and require
    that additional linear constraints be added to the OPF.

    The C{gen} matrix in version 2 of the PYPOWER case format includes columns
    for specifying a P-Q capability curve for a generator defined as the
    intersection of two half-planes and the box constraints on P and Q. The
    two half planes are defined respectively as the area below the line
    connecting (Pc1, Qc1max) and (Pc2, Qc2max) and the area above the line
    connecting (Pc1, Qc1min) and (Pc2, Qc2min).

    If the optional 2nd argument is 'U' this function returns C{True} only for
    rows corresponding to generators that require the upper constraint on Q.
    If it is 'L', only for those requiring the lower constraint. If the 2nd
    argument is not specified or has any other value it returns true for rows
    corresponding to gens that require either or both of the constraints.

    It is smart enough to return C{True} only if the corresponding linear
    constraint is not redundant w.r.t the box constraints.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## check for errors capability curve data
    if any( gen[:, PC1] > gen[:, PC2] ):
        logger.debug('hasPQcap: Pc1 > Pc2\n')
    if any( gen[:, QC2MAX] > gen[:, QC1MAX] ):
        logger.debug('hasPQcap: Qc2max > Qc1max\n')
    if any( gen[:, QC2MIN] < gen[:, QC1MIN] ):
        logger.debug('hasPQcap: Qc2min < Qc1min\n')

    L = zeros(gen.shape[0], bool)
    U = zeros(gen.shape[0], bool)
    k = nonzero( gen[:, PC1] != gen[:, PC2] )

    if hilo != 'U':       ## include lower constraint
        Qmin_at_Pmax = gen[k, QC1MIN] + (gen[k, PMAX] - gen[k, PC1]) * \
            (gen[k, QC2MIN] - gen[k, QC1MIN]) / (gen[k, PC2] - gen[k, PC1])
        L[k] = Qmin_at_Pmax > gen[k, QMIN]

    if hilo != 'L':       ## include upper constraint
        Qmax_at_Pmax = gen[k, QC1MAX] + (gen[k, PMAX] - gen[k, PC1]) * \
            (gen[k, QC2MAX] - gen[k, QC1MAX]) / (gen[k, PC2] - gen[k, PC1])
        U[k] = Qmax_at_Pmax < gen[k, QMAX]

    return L | U


def makeAang(baseMVA, branch, nb, ppopt):
    """
    Construct constraints for branch angle difference limits.

    Constructs the parameters for the following linear constraint limiting
    the voltage angle differences across branches, where C{Va} is the vector
    of bus voltage angles. C{nb} is the number of buses::

        lang <= Aang * Va <= uang

    C{iang} is the vector of indices of branches with angle difference limits.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    ## options
    ignore_ang_lim = ppopt['OPF_IGNORE_ANG_LIM']

    if ignore_ang_lim:
        Aang  = zeros((0, nb))
        lang  = array([])
        uang  = array([])
        iang  = array([])
    else:
        iang = find(((branch[:, ANGMIN] != 0) & (branch[:, ANGMIN] > -360)) |
                    ((branch[:, ANGMAX] != 0) & (branch[:, ANGMAX] <  360)))
        iangl = find(branch[iang, ANGMIN])
        iangh = find(branch[iang, ANGMAX])
        nang = len(iang)

        if nang > 0:
            ii = r_[arange(nang), arange(nang)]
            jj = r_[branch[iang, F_BUS], branch[iang, T_BUS]]
            Aang = sparse((r_[ones(nang), -ones(nang)],
                           (ii, jj)), (nang, nb))
            uang = Inf * ones(nang)
            lang = -uang
            lang[iangl] = branch[iang[iangl], ANGMIN] * pi / 180
            uang[iangh] = branch[iang[iangh], ANGMAX] * pi / 180
        else:
            Aang  = zeros((0, nb))
            lang  = array([])
            uang  = array([])

    return Aang, lang, uang, iang


def makeApq(baseMVA, gen):
    """Construct linear constraints for generator capability curves.

    Constructs the parameters for the following linear constraints
    implementing trapezoidal generator capability curves, where
    C{Pg} and C{Qg} are the real and reactive generator injections::

        Apqh * [Pg, Qg] <= ubpqh
        Apql * [Pg, Qg] <= ubpql

    C{data} constains additional information as shown below.

    Example::
        Apqh, ubpqh, Apql, ubpql, data = makeApq(baseMVA, gen)

        data['h']      [Qc1max-Qc2max, Pc2-Pc1]
        data['l']      [Qc2min-Qc1min, Pc1-Pc2]
        data['ipqh']   indices of gens with general PQ cap curves (upper)
        data['ipql']   indices of gens with general PQ cap curves (lower)

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    data = {}
    ## data dimensions
    ng = gen.shape[0]      ## number of dispatchable injections

    ## which generators require additional linear constraints
    ## (in addition to simple box constraints) on (Pg,Qg) to correctly
    ## model their PQ capability curves
    ipqh = find( hasPQcap(gen, 'U') )
    ipql = find( hasPQcap(gen, 'L') )
    npqh = ipqh.shape[0]   ## number of general PQ capability curves (upper)
    npql = ipql.shape[0]   ## number of general PQ capability curves (lower)

    ## make Apqh if there is a need to add general PQ capability curves
    ## use normalized coefficient rows so multipliers have right scaling
    ## in $$/pu
    if npqh > 0:
        data["h"] = c_[gen[ipqh, QC1MAX] - gen[ipqh, QC2MAX],
                       gen[ipqh, PC2] - gen[ipqh, PC1]]
        ubpqh = data["h"][:, 0] * gen[ipqh, PC1] + \
                data["h"][:, 1] * gen[ipqh, QC1MAX]
        for i in range(npqh):
            tmp = linalg.norm(data["h"][i, :])
            data["h"][i, :] = data["h"][i, :] / tmp
            ubpqh[i] = ubpqh[i] / tmp
        Apqh = sparse((data["h"].flatten('F'),
                       (r_[arange(npqh), arange(npqh)], r_[ipqh, ipqh+ng])),
                      (npqh, 2*ng))
        ubpqh = ubpqh / baseMVA
    else:
        data["h"] = array([])
        Apqh  = zeros((0, 2*ng))
        ubpqh = array([])

    ## similarly Apql
    if npql > 0:
        data["l"] = c_[gen[ipql, QC2MIN] - gen[ipql, QC1MIN],
                       gen[ipql, PC1] - gen[ipql, PC2]]
        ubpql = data["l"][:, 0] * gen[ipql, PC1] + \
                data["l"][:, 1] * gen[ipql, QC1MIN]
        for i in range(npql):
            tmp = linalg.norm(data["l"][i, :])
            data["l"][i, :] = data["l"][i, :] / tmp
            ubpql[i] = ubpql[i] / tmp
        Apql = sparse((data["l"].flatten('F'),
                       (r_[arange(npql), arange(npql)], r_[ipql, ipql+ng])),
                      (npql, 2*ng))
        ubpql = ubpql / baseMVA
    else:
        data["l"] = array([])
        Apql  = zeros((0, 2*ng))
        ubpql = array([])

    data["ipql"] = ipql
    data["ipqh"] = ipqh

    return Apqh, ubpqh, Apql, ubpql, data


def makeAvl(baseMVA, gen):
    """Construct linear constraints for constant power factor var loads.

    Constructs parameters for the following linear constraint enforcing a
    constant power factor constraint for dispatchable loads::

         lvl <= Avl * [Pg, Qg] <= uvl

    C{ivl} is the vector of indices of generators representing variable loads.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    ## data dimensions
    ng = gen.shape[0]      ## number of dispatchable injections
    Pg   = gen[:, PG] / baseMVA
    Qg   = gen[:, QG] / baseMVA
    Pmin = gen[:, PMIN] / baseMVA
    Qmin = gen[:, QMIN] / baseMVA
    Qmax = gen[:, QMAX] / baseMVA

    # Find out if any of these "generators" are actually dispatchable loads.
    # (see 'help isload' for details on what constitutes a dispatchable load)
    # Dispatchable loads are modeled as generators with an added constant
    # power factor constraint. The power factor is derived from the original
    # value of Pmin and either Qmin (for inductive loads) or Qmax (for
    # capacitive loads). If both Qmin and Qmax are zero, this implies a unity
    # power factor without the need for an additional constraint.

    ivl = find( isload(gen) & ((Qmin != 0) | (Qmax != 0)) )
    nvl = ivl.shape[0]  ## number of dispatchable loads

    ## at least one of the Q limits must be zero (corresponding to Pmax == 0)
    if any( (Qmin[ivl] != 0) & (Qmax[ivl] != 0) ):
        logger.debug('makeAvl: either Qmin or Qmax must be equal to zero for '
                     'each dispatchable load.\n')

    # Initial values of PG and QG must be consistent with specified power
    # factor This is to prevent a user from unknowingly using a case file which
    # would have defined a different power factor constraint under a previous
    # version which used PG and QG to define the power factor.
    Qlim = (Qmin[ivl] == 0) * Qmax[ivl] + (Qmax[ivl] == 0) * Qmin[ivl]
    if any( abs( Qg[ivl] - Pg[ivl] * Qlim / Pmin[ivl] ) > 1e-6 ):
        logger.debug('makeAvl: For a dispatchable load, PG and QG must be '
                     'consistent with the power factor defined by PMIN and '
                     'the Q limits.\n')

    # make Avl, lvl, uvl, for lvl <= Avl * [Pg Qg] <= uvl
    if nvl > 0:
        xx = Pmin[ivl]
        yy = Qlim
        pftheta = arctan2(yy, xx)
        pc = sin(pftheta)
        qc = -cos(pftheta)
        ii = r_[ arange(nvl), arange(nvl) ]
        jj = r_[ ivl, ivl + ng ]
        Avl = sparse((r_[pc, qc], (ii, jj)), (nvl, 2 * ng))
        lvl = zeros(nvl)
        uvl = lvl
    else:
        Avl = zeros((0, 2*ng))
        lvl = array([])
        uvl = array([])

    return Avl, lvl, uvl, ivl


def makeAvl(baseMVA, gen):
    """Construct linear constraints for constant power factor var loads.

    Constructs parameters for the following linear constraint enforcing a
    constant power factor constraint for dispatchable loads::

         lvl <= Avl * [Pg, Qg] <= uvl

    C{ivl} is the vector of indices of generators representing variable loads.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    ## data dimensions
    ng = gen.shape[0]      ## number of dispatchable injections
    Pg   = gen[:, PG] / baseMVA
    Qg   = gen[:, QG] / baseMVA
    Pmin = gen[:, PMIN] / baseMVA
    Qmin = gen[:, QMIN] / baseMVA
    Qmax = gen[:, QMAX] / baseMVA

    # Find out if any of these "generators" are actually dispatchable loads.
    # (see 'help isload' for details on what constitutes a dispatchable load)
    # Dispatchable loads are modeled as generators with an added constant
    # power factor constraint. The power factor is derived from the original
    # value of Pmin and either Qmin (for inductive loads) or Qmax (for
    # capacitive loads). If both Qmin and Qmax are zero, this implies a unity
    # power factor without the need for an additional constraint.

    ivl = find( isload(gen) & ((Qmin != 0) | (Qmax != 0)) )
    nvl = ivl.shape[0]  ## number of dispatchable loads

    ## at least one of the Q limits must be zero (corresponding to Pmax == 0)
    if any( (Qmin[ivl] != 0) & (Qmax[ivl] != 0) ):
        logger.debug('makeAvl: either Qmin or Qmax must be equal to zero for '
                     'each dispatchable load.\n')

    # Initial values of PG and QG must be consistent with specified power
    # factor This is to prevent a user from unknowingly using a case file which
    # would have defined a different power factor constraint under a previous
    # version which used PG and QG to define the power factor.
    Qlim = (Qmin[ivl] == 0) * Qmax[ivl] + (Qmax[ivl] == 0) * Qmin[ivl]
    if any( abs( Qg[ivl] - Pg[ivl] * Qlim / Pmin[ivl] ) > 1e-6 ):
        logger.debug('makeAvl: For a dispatchable load, PG and QG must be '
                     'consistent with the power factor defined by PMIN and '
                     'the Q limits.\n')

    # make Avl, lvl, uvl, for lvl <= Avl * [Pg Qg] <= uvl
    if nvl > 0:
        xx = Pmin[ivl]
        yy = Qlim
        pftheta = arctan2(yy, xx)
        pc = sin(pftheta)
        qc = -cos(pftheta)
        ii = r_[ arange(nvl), arange(nvl) ]
        jj = r_[ ivl, ivl + ng ]
        Avl = sparse((r_[pc, qc], (ii, jj)), (nvl, 2 * ng))
        lvl = zeros(nvl)
        uvl = lvl
    else:
        Avl = zeros((0, 2*ng))
        lvl = array([])
        uvl = array([])

    return Avl, lvl, uvl, ivl



def makeB(baseMVA, bus, branch, alg):
    """Builds the FDPF matrices, B prime and B double prime.

    Returns the two matrices B prime and B double prime used in the fast
    decoupled power flow. Does appropriate conversions to p.u. C{alg} is the
    value of the C{PF_ALG} option specifying the power flow algorithm.

    @see: L{fdpf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## constants
    nb = bus.shape[0]          ## number of buses
    nl = branch.shape[0]       ## number of lines

    ##-----  form Bp (B prime)  -----
    temp_branch = copy(branch)                 ## modify a copy of branch
    temp_bus = copy(bus)                       ## modify a copy of bus
    temp_bus[:, BS] = zeros(nb)                ## zero out shunts at buses
    temp_branch[:, BR_B] = zeros(nl)           ## zero out line charging shunts
    temp_branch[:, TAP] = ones(nl)             ## cancel out taps
    if alg == 2:                               ## if XB method
        temp_branch[:, BR_R] = zeros(nl)       ## zero out line resistance
    Bp = -1 * makeYbus(baseMVA, temp_bus, temp_branch)[0].imag

    ##-----  form Bpp (B double prime)  -----
    temp_branch = copy(branch)                 ## modify a copy of branch
    temp_branch[:, SHIFT] = zeros(nl)          ## zero out phase shifters
    if alg == 3:                               ## if BX method
        temp_branch[:, BR_R] = zeros(nl)    ## zero out line resistance
    Bpp = -1 * makeYbus(baseMVA, bus, temp_branch)[0].imag

    return Bp, Bpp


def makeBdc(baseMVA, bus, branch):
    """Builds the B matrices and phase shift injections for DC power flow.

    Returns the B matrices and phase shift injection vectors needed for a
    DC power flow.
    The bus real power injections are related to bus voltage angles by::
        P = Bbus * Va + PBusinj
    The real power flows at the from end the lines are related to the bus
    voltage angles by::
        Pf = Bf * Va + Pfinj
    Does appropriate conversions to p.u.

    @see: L{dcpf}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## constants
    nb = bus.shape[0]          ## number of buses
    nl = branch.shape[0]       ## number of lines

    ## check that bus numbers are equal to indices to bus (one set of bus nums)
    if any(bus[:, BUS_I] != list(range(nb))):
        logger.debug('makeBdc: buses must be numbered consecutively in '
                     'bus matrix\n')

    ## for each branch, compute the elements of the branch B matrix and the phase
    ## shift "quiescent" injections, where
    ##
    ##      | Pf |   | Bff  Bft |   | Vaf |   | Pfinj |
    ##      |    | = |          | * |     | + |       |
    ##      | Pt |   | Btf  Btt |   | Vat |   | Ptinj |
    ##
    stat = branch[:, BR_STATUS]               ## ones at in-service branches
    b = stat / branch[:, BR_X]                ## series susceptance
    tap = ones(nl)                            ## default tap ratio = 1
    i = find(branch[:, TAP])               ## indices of non-zero tap ratios
    tap[i] = branch[i, TAP]                   ## assign non-zero tap ratios
    b = b / tap

    ## build connection matrix Cft = Cf - Ct for line and from - to buses
    f = branch[:, F_BUS]                           ## list of "from" buses
    t = branch[:, T_BUS]                           ## list of "to" buses
    i = r_[range(nl), range(nl)]                   ## double set of row indices
    ## connection matrix
    Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))

    ## build Bf such that Bf * Va is the vector of real branch powers injected
    ## at each branch's "from" bus
    Bf = sparse((r_[b, -b], (i, r_[f, t])), shape = (nl, nb))## = spdiags(b, 0, nl, nl) * Cft

    ## build Bbus
    Bbus = Cft.T * Bf

    ## build phase shift injection vectors
    Pfinj = b * (-branch[:, SHIFT] * pi / 180)  ## injected at the from bus ...
    # Ptinj = -Pfinj                            ## and extracted at the to bus
    Pbusinj = Cft.T * Pfinj                ## Pbusinj = Cf * Pfinj + Ct * Ptinj

    return Bbus, Bf, Pbusinj, Pfinj


def makeLODF(branch, PTDF):
    """Builds the line outage distribution factor matrix.

    Returns the DC line outage distribution factor matrix for a given PTDF.
    The matrix is C{nbr x nbr}, where C{nbr} is the number of branches.

    Example::
        H = makePTDF(baseMVA, bus, branch)
        LODF = makeLODF(branch, H)

    @see: L{makePTDF}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    nl, nb = PTDF.shape
    f = branch[:, F_BUS]
    t = branch[:, T_BUS]
    Cft = sparse((r_[ones(nl), -ones(nl)],
                      (r_[f, t], r_[arange(nl), arange(nl)])), (nb, nl))

    H = PTDF * Cft
    h = diag(H, 0)
    LODF = H / (ones((nl, nl)) - ones((nl, 1)) * h.T)
    LODF = LODF - diag(diag(LODF)) - eye(nl, nl)

    return LODF


def makePTDF(baseMVA, bus, branch, slack=None):
    """Builds the DC PTDF matrix for a given choice of slack.

    Returns the DC PTDF matrix for a given choice of slack. The matrix is
    C{nbr x nb}, where C{nbr} is the number of branches and C{nb} is the
    number of buses. The C{slack} can be a scalar (single slack bus) or an
    C{nb x 1} column vector of weights specifying the proportion of the
    slack taken up at each bus. If the C{slack} is not specified the
    reference bus is used by default.

    For convenience, C{slack} can also be an C{nb x nb} matrix, where each
    column specifies how the slack should be handled for injections
    at that bus.

    @see: L{makeLODF}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## use reference bus for slack by default
    if slack is None:
        slack = find(bus[:, BUS_TYPE] == REF)
        slack = slack[0]

    ## set the slack bus to be used to compute initial PTDF
    if isscalar(slack):
        slack_bus = slack
    else:
        slack_bus = 0      ## use bus 1 for temp slack bus

    nb = bus.shape[0]
    nbr = branch.shape[0]
    noref = arange(1, nb)      ## use bus 1 for voltage angle reference
    noslack = find(arange(nb) != slack_bus)

    ## check that bus numbers are equal to indices to bus (one set of bus numbers)
    if any(bus[:, BUS_I] != arange(nb)):
        logger.debug('makePTDF: buses must be numbered consecutively')

    ## compute PTDF for single slack_bus
    Bbus, Bf, _, _ = makeBdc(baseMVA, bus, branch)
    Bbus, Bf = Bbus.todense(), Bf.todense()
    H = zeros((nbr, nb))
    H[:, noslack] = solve( Bbus[ix_(noslack, noref)].T, Bf[:, noref].T ).T
    #             = Bf[:, noref] * inv(Bbus[ix_(noslack, noref)])

    ## distribute slack, if requested
    if not isscalar(slack):
        if len(slack.shape) == 1:  ## slack is a vector of weights
            slack = slack / sum(slack)   ## normalize weights

            ## conceptually, we want to do ...
            ##    H = H * (eye(nb, nb) - slack * ones((1, nb)))
            ## ... we just do it more efficiently
            v = dot(H, slack)
            for k in range(nb):
                H[:, k] = H[:, k] - v
        else:
            H = dot(H, slack)

    return H
