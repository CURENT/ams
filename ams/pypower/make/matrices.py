"""
Make matrices.
"""
import logging  # NOQA

import numpy as np  # NOQA

from numpy import flatnonzero as find  # NOQA
from scipy.sparse import csr_matrix as c_sparse  # NOQA
from scipy.sparse import lil_matrix as l_sparse  # NOQA

from andes.shared import deg2rad  # NOQA
from ams.pypower.idx import IDX  # NOQA

import ams.pypower.utils as putil  # NOQA

logger = logging.getLogger(__name__)


def makeAang(baseMVA, branch, nb, ppopt):
    """
    Constructs the parameters for the following linear constraint limiting
    the voltage angle differences across branches, where C{Va} is the vector
    of bus voltage angles. C{nb} is the number of buses::

        lang <= Aang * Va <= uang

    Parameters
    ----------
    branch: np.ndarray
        The branch matrix.
    nb : int
        The number of buses.
    ppopt : dict
        PYPOWER options dictionary.

    Returns
    -------
    Aang: sparse matrix
        The constraint matrix for branch angle difference limits.
    lang: np.ndarray
        The lower bounds on the angle difference limits.
    uang: np.ndarray
        The upper bounds on the angle difference limits.
    iang: np.ndarray
        The indices of the branches with angle difference limits.
    """
    # options
    ignore_ang_lim = ppopt['OPF_IGNORE_ANG_LIM']

    if ignore_ang_lim:
        Aang = np.zeros((0, nb))
        lang = np.array([])
        uang = np.array([])
        iang = np.array([])
    else:
        iang = find(((branch[:, IDX.branch.ANGMIN] != 0) & (branch[:, IDX.branch.ANGMIN] > -360)) |
                    ((branch[:, IDX.branch.ANGMAX] != 0) & (branch[:, IDX.branch.ANGMAX] < 360)))
        iangl = find(branch[iang, IDX.branch.ANGMIN])
        iangh = find(branch[iang, IDX.branch.ANGMAX])
        nang = len(iang)

        if nang > 0:
            ii = np.r_[np.arange(nang), np.arange(nang)]
            jj = np.r_[branch[iang, IDX.branch.F_BUS], branch[iang, IDX.branch.T_BUS]]
            Aang = c_sparse((np.r_[np.ones(nang), -np.ones(nang)],
                             (ii, jj)), (nang, nb))
            uang = np.Inf * np.ones(nang)
            lang = -uang
            lang[iangl] = branch[iang[iangl], IDX.branch.ANGMIN] * deg2rad
            uang[iangh] = branch[iang[iangh], IDX.branch.ANGMAX] * deg2rad
        else:
            Aang = np.zeros((0, nb))
            lang = np.array([])
            uang = np.array([])

    return Aang, lang, uang, iang


def makeApq(baseMVA, gen):
    """
    Construct linear constraints for generator capability curves,
    where Pg and Qg are the real and reactive generator injections::

        Apqh * [Pg, Qg] <= ubpqh
        Apql * [Pg, Qg] <= ubpql

    Parameters
    ----------
    baseMVA: float
        The system base MVA.
    gen: np.ndarray
        The generator matrix.

    Example
    -------
    >>> Apqh, ubpqh, Apql, ubpql, data = makeApq(baseMVA, gen)

    C{data} constains additional information as shown below.

    data['h']      [Qc1max-Qc2max, Pc2-Pc1]

    data['l']      [Qc2min-Qc1min, Pc1-Pc2]

    data['ipqh']   indices of gens with general PQ cap curves (upper)

    data['ipql']   indices of gens with general PQ cap curves (lower)
    """
    data = {}
    # data dimensions
    ng = gen.shape[0]  # number of dispatchable injections

    # which generators require additional linear constraints
    # (in addition to simple box constraints) on (Pg,Qg) to correctly
    # model their PQ capability curves
    ipqh = find(putil.hasPQcap(gen, 'U'))
    ipql = find(putil.hasPQcap(gen, 'L'))
    npqh = ipqh.shape[0]  # number of general PQ capability curves (upper)
    npql = ipql.shape[0]  # number of general PQ capability curves (lower)

    # make Apqh if there is a need to add general PQ capability curves
    # use normalized coefficient rows so multipliers have right scaling
    # in $$/pu
    if npqh > 0:
        data["h"] = np.c_[gen[ipqh, IDX.gen.QC1MAX] - gen[ipqh, IDX.gen.QC2MAX],
                          gen[ipqh, IDX.gen.PC2] - gen[ipqh, IDX.gen.PC1]]
        ubpqh = data["h"][:, 0] * gen[ipqh, IDX.gen.PC1] + \
            data["h"][:, 1] * gen[ipqh, IDX.gen.QC1MAX]
        for i in range(npqh):
            tmp = np.linalg.norm(data["h"][i, :])
            data["h"][i, :] = data["h"][i, :] / tmp
            ubpqh[i] = ubpqh[i] / tmp
        Apqh = c_sparse((data["h"].flatten('F'),
                         (np.r_[np.arange(npqh), np.arange(npqh)], np.r_[ipqh, ipqh+ng])),
                        (npqh, 2*ng))
        ubpqh = ubpqh / baseMVA
    else:
        data["h"] = np.array([])
        Apqh = np.zeros((0, 2*ng))
        ubpqh = np.array([])

    # similarly Apql
    if npql > 0:
        data["l"] = np.c_[gen[ipql, IDX.gen.QC2MIN] - gen[ipql, IDX.gen.QC1MIN],
                          gen[ipql, IDX.gen.PC1] - gen[ipql, IDX.gen.PC2]]
        ubpql = data["l"][:, 0] * gen[ipql, IDX.gen.PC1] + \
            data["l"][:, 1] * gen[ipql, IDX.gen.QC1MIN]
        for i in range(npql):
            tmp = np.linalg.norm(data["l"][i, :])
            data["l"][i, :] = data["l"][i, :] / tmp
            ubpql[i] = ubpql[i] / tmp
        Apql = c_sparse((data["l"].flatten('F'),
                         (np.r_[np.arange(npql), np.arange(npql)], np.r_[ipql, ipql+ng])),
                        (npql, 2*ng))
        ubpql = ubpql / baseMVA
    else:
        data["l"] = np.array([])
        Apql = np.zeros((0, 2*ng))
        ubpql = np.array([])

    data["ipql"] = ipql
    data["ipqh"] = ipqh

    return Apqh, ubpqh, Apql, ubpql, data


def makeAvl(baseMVA, gen):
    """
    Constructs parameters for the following linear constraint enforcing a
    constant power factor constraint for dispatchable loads::

        lvl <= Avl * [Pg, Qg] <= uvl

    Parameters
    ----------
    baseMVA: float
        The system base MVA.
    gen: np.ndarray
        The generator matrix.
    """
    # data dimensions
    ng = gen.shape[0]  # number of dispatchable injections
    Pg = gen[:, IDX.gen.PG] / baseMVA
    Qg = gen[:, IDX.gen.QG] / baseMVA
    Pmin = gen[:, IDX.gen.PMIN] / baseMVA
    Qmin = gen[:, IDX.gen.QMIN] / baseMVA
    Qmax = gen[:, IDX.gen.QMAX] / baseMVA

    # Find out if any of these "generators" are actually dispatchable loads.
    # (see 'help isload' for details on what constitutes a dispatchable load)
    # Dispatchable loads are modeled as generators with an added constant
    # power factor constraint. The power factor is derived from the original
    # value of Pmin and either Qmin (for inductive loads) or Qmax (for
    # capacitive loads). If both Qmin and Qmax are zero, this implies a unity
    # power factor without the need for an additional constraint.
    # NOTE: C{ivl} is the vector of indices of generators representing variable loads.
    ivl = find(putil.isload(gen) & ((Qmin != 0) | (Qmax != 0)))
    nvl = ivl.shape[0]  # number of dispatchable loads

    # at least one of the Q limits must be zero (corresponding to Pmax == 0)
    if np.any((Qmin[ivl] != 0) & (Qmax[ivl] != 0)):
        logger.debug('makeAvl: either Qmin or Qmax must be equal to zero for '
                     'each dispatchable load.\n')

    # Initial values of IDX.gen.PG and IDX.gen.QG must be consistent with specified power
    # factor This is to prevent a user from unknowingly using a case file which
    # would have defined a different power factor constraint under a previous
    # version which used IDX.gen.PG and IDX.gen.QG to define the power factor.
    Qlim = (Qmin[ivl] == 0) * Qmax[ivl] + (Qmax[ivl] == 0) * Qmin[ivl]
    if np.any(abs(Qg[ivl] - Pg[ivl] * Qlim / Pmin[ivl]) > 1e-6):
        logger.debug('makeAvl: For a dispatchable load, PG and QG must be '
                     'consistent with the power factor defined by PMIN and '
                     'the Q limits.\n')

    # make Avl, lvl, uvl, for lvl <= Avl * [Pg Qg] <= uvl
    if nvl > 0:
        xx = Pmin[ivl]
        yy = Qlim
        pftheta = np.arctan2(yy, xx)
        pc = np.sin(pftheta)
        qc = -np.cos(pftheta)
        ii = np.r_[np.arange(nvl), np.arange(nvl)]
        jj = np.r_[ivl, ivl + ng]
        Avl = c_sparse((np.r_[pc, qc], (ii, jj)), (nvl, 2 * ng))
        lvl = np.zeros(nvl)
        uvl = lvl
    else:
        Avl = np.zeros((0, 2*ng))
        lvl = np.array([])
        uvl = np.array([])

    return Avl, lvl, uvl, ivl


def makeB(baseMVA, bus, branch, alg):
    """
    Builds the FDPF matrices, B prime and B double prime.

    Parameters
    ----------
    baseMVA: float
        The system base MVA.
    bus: np.ndarray
        The bus matrix.
    branch: np.ndarray
        The branch matrix.
    alg: int
        The power flow algorithm, value of the C{PF_ALG} option, see L{runpf}.

    Returns
    -------
    Bp: np.ndarray
        The B prime matrix.
    Bpp: np.ndarray
        The B double prime matrix.
    """
    # constants
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of lines

    # -----  form Bp (B prime)  -----
    temp_branch = np.copy(branch)  # modify a copy of branch
    temp_bus = np.copy(bus)  # modify a copy of bus
    temp_bus[:, IDX.bus.BS] = np.zeros(nb)  # zero out shunts at buses
    temp_branch[:, IDX.branch.BR_B] = np.zeros(nl)  # zero out line charging shunts
    temp_branch[:, IDX.branch.TAP] = np.ones(nl)  # cancel out taps
    if alg == 2:  # if XB method
        temp_branch[:, IDX.branch.BR_R] = np.zeros(nl)  # zero out line resistance
    Bp = -1 * makeYbus(baseMVA, temp_bus, temp_branch)[0].imag

    # -----  form Bpp (B double prime)  -----
    temp_branch = np.copy(branch)  # modify a copy of branch
    temp_branch[:, IDX.branch.SHIFT] = np.zeros(nl)  # zero out phase shifters
    if alg == 3:  # if BX method
        temp_branch[:, IDX.branch.BR_R] = np.zeros(nl)  # zero out line resistance
    Bpp = -1 * makeYbus(baseMVA, bus, temp_branch)[0].imag

    return Bp, Bpp


def makeBdc(baseMVA, bus, branch):
    """
    Returns the B matrices and phase shift injection vectors needed for a
    DC power flow.
    The bus real power injections are related to bus voltage angles by::

        P = Bbus * Va + PBusinj

    The real power flows at the from end the lines are related to the bus
    voltage angles by::

        Pf = Bf * Va + Pfinj

    Parameters
    ----------
    baseMVA: float
        The system base MVA.
    bus: np.ndarray
        The bus matrix.
    branch: np.ndarray
        The branch matrix.

    Returns
    -------
    Bbus: np.ndarray
        The B matrix.
    Bf: np.ndarray
        The Bf matrix.
    PBusinj: np.ndarray
        The real power injection vector at each bus.
    Pfinj: np.ndarray
        The real power injection vector at the "from" end of each branch.
    Qfinj: np.ndarray
        The reactive power injection vector at the "from" end of each branch.
    """
    # constants
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of lines

    # check that bus numbers are equal to indices to bus (one set of bus nums)
    if np.any(bus[:, IDX.bus.BUS_I] != list(range(nb))):
        logger.debug('makeBdc: buses must be numbered consecutively in '
                     'bus matrix\n')

    # for each branch, compute the elements of the branch B matrix and the phase
    # shift "quiescent" injections, where
    ##
    # | Pf |   | Bff  Bft |   | Vaf |   | Pfinj |
    # |    | = |          | * |     | + |       |
    # | Pt |   | Btf  Btt |   | Vat |   | Ptinj |
    ##
    stat = branch[:, IDX.branch.BR_STATUS]  # ones at in-service branches
    b = stat / branch[:, IDX.branch.BR_X]  # series susceptance
    tap = np.ones(nl)  # default tap ratio = 1
    i = find(branch[:, IDX.branch.TAP])  # indices of non-zero tap ratios
    tap[i] = branch[i, IDX.branch.TAP]  # assign non-zero tap ratios
    b = b / tap

    # build connection matrix Cft = Cf - Ct for line and from - to buses
    f = branch[:, IDX.branch.F_BUS]  # list of "from" buses
    t = branch[:, IDX.branch.T_BUS]  # list of "to" buses
    i = np.r_[range(nl), range(nl)]  # double set of row indices
    # connection matrix
    Cft = c_sparse((np.r_[np.ones(nl), -np.ones(nl)], (i, np.r_[f, t])), (nl, nb))

    # build Bf such that Bf * Va is the vector of real branch powers injected
    # at each branch's "from" bus
    Bf = c_sparse((np.r_[b, -b], (i, np.r_[f, t])), shape=(nl, nb))  # = spdiags(b, 0, nl, nl) * Cft

    # build Bbus
    Bbus = Cft.T * Bf

    # build phase shift injection vectors
    Pfinj = b * (-branch[:, IDX.branch.SHIFT] * deg2rad)  # injected at the from bus ...
    # Ptinj = -Pfinj                            ## and extracted at the to bus
    Pbusinj = Cft.T * Pfinj  # Pbusinj = Cf * Pfinj + Ct * Ptinj

    return Bbus, Bf, Pbusinj, Pfinj, Cft


def makeLODF(branch, PTDF):
    """
    Builds the line outage distribution factor matrix.

    Parameters
    ----------
    branch: np.ndarray
        The branch matrix.
    PTDF: np.ndarray
        The PTDF matrix.

    Returns
    -------
    LODF: np.ndarray
        The DC line outage distribution factor matrix for a given PTDF.
        The matrix is C{nbr x nbr}, where C{nbr} is the number of branches.

    Example
    -------
    >>> H = makePTDF(baseMVA, bus, branch)
    >>> LODF = makeLODF(branch, H)
    """
    nl, nb = PTDF.shape
    f = branch[:, IDX.branch.F_BUS]
    t = branch[:, IDX.branch.T_BUS]
    Cft = c_sparse((np.r_[np.ones(nl), -np.ones(nl)],
                    (np.r_[f, t], np.r_[np.arange(nl), np.arange(nl)])), (nb, nl))

    H = PTDF * Cft
    h = np.diag(H, 0)
    LODF = H / (np.ones((nl, nl)) - np.ones((nl, 1)) * h.T)
    LODF = LODF - np.diag(np.diag(LODF)) - np.eye(nl, nl)

    return LODF


def makePTDF(baseMVA, bus, branch, slack=None):
    """
    Builds the DC PTDF matrix for a given choice of slack.

    Parameters
    ----------
    baseMVA: float
        The system base MVA.
    bus: np.ndarray
        The bus matrix.
    branch: np.ndarray
        The branch matrix.
    slack : int, array, or matrix, optional
        The slack bus number or the slack bus weight vector. The default is
        to use the reference bus.

    Returns
    -------
    H: np.ndarray
        The DC PTDF matrix for a given choice of slack.

        The matrix is C{nbr x nb}, where C{nbr} is the number of branches
        and C{nb} is the number of buses.

        The C{slack} can be a scalar (single slack bus) or an C{nb x 1} column
        vector of weights specifying the proportion of the slack taken up at each bus.

        If the C{slack} is not specified the reference bus is used by default.

        For convenience, C{slack} can also be an C{nb x nb} matrix, where each
        column specifies how the slack should be handled for injections at that bus.
    """
    # use reference bus for slack by default
    if slack is None:
        slack = find(bus[:, IDX.bus.BUS_TYPE] == IDX.bus.REF)
        slack = slack[0]

    # set the slack bus to be used to compute initial PTDF
    if np.isscalar(slack):
        slack_bus = slack
    else:
        slack_bus = 0  # use bus 1 for temp slack bus

    nb = bus.shape[0]
    nbr = branch.shape[0]
    noref = np.arange(1, nb)  # use bus 1 for voltage angle reference
    noslack = find(np.arange(nb) != slack_bus)

    # check that bus numbers are equal to indices to bus (one set of bus numbers)
    if np.any(bus[:, IDX.bus.BUS_I] != np.arange(nb)):
        logger.debug('makePTDF: buses must be numbered consecutively')

    # compute PTDF for single slack_bus
    Bbus, Bf, _, _, _ = makeBdc(baseMVA, bus, branch)
    Bbus, Bf = Bbus.todense(), Bf.todense()
    H = np.zeros((nbr, nb))
    H[:, noslack] = np.linalg.solve(Bbus[np.ix_(noslack, noref)].T, Bf[:, noref].T).T
    #             = Bf[:, noref] * inv(Bbus[np.ix_(noslack, noref)])

    # distribute slack, if requested
    if not np.isscalar(slack):
        if len(slack.shape) == 1:  # slack is a vector of weights
            slack = slack / sum(slack)  # normalize weights

            # conceptually, we want to do ...
            # H = H * (eye(nb, nb) - slack * ones((1, nb)))
            # ... we just do it more efficiently
            v = np.dot(H, slack)
            for k in range(nb):
                H[:, k] = H[:, k] - v
        else:
            H = np.dot(H, slack)

    return H


def makeSbus(baseMVA, bus, gen):
    """
    Builds the vector of complex bus power injections.

    Parameters
    ----------
    baseMVA: float
        Base MVA.
    bus: NumPy.array
        Bus data.
    gen: NumPy.array
        Generator data.

    Returns
    -------
    Sbus : NumPy.array
        Complex bus power injections.
    """
    # generator info
    on = find(gen[:, IDX.gen.GEN_STATUS] > 0)  # which generators are on?
    gbus = gen[on, IDX.gen.GEN_BUS]  # what buses are they at?

    # form net complex bus power injection vector
    nb = bus.shape[0]
    ngon = on.shape[0]
    # connection matrix, element i, j is 1 if gen on(j) at bus i is ON
    Cg = c_sparse((np.ones(ngon), (gbus, range(ngon))), (nb, ngon))

    # power injected by gens plus power injected by loads converted to p.u.
    Sbus = (Cg * (gen[on, IDX.gen.PG] + 1j * gen[on, IDX.gen.QG]) -
            (bus[:, IDX.bus.PD] + 1j * bus[:, IDX.bus.QD])) / baseMVA

    return Sbus


def makeYbus(baseMVA, bus, branch):
    """Builds the bus admittance matrix and branch admittance matrices.

    Returns the full bus admittance matrix (i.e. for all buses) and the
    matrices C{Yf} and C{Yt} which, when multiplied by a complex voltage
    vector, yield the vector currents injected into each line from the
    "from" and "to" buses respectively of each line. Does appropriate
    conversions to p.u.

    @see: L{makeSbus}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # constants
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of lines

    # check that bus numbers are equal to indices to bus (one set of bus nums)
    if np.any(bus[:, IDX.bus.BUS_I] != list(range(nb))):
        logger.debug('buses must appear in order by bus number\n')

    # for each branch, compute the elements of the branch admittance matrix where
    ##
    # | If |   | Yff  Yft |   | Vf |
    # |    | = |          | * |    |
    # | It |   | Ytf  Ytt |   | Vt |
    ##
    stat = branch[:, IDX.branch.BR_STATUS]  # ones at in-service branches
    Ys = stat / (branch[:, IDX.branch.BR_R] + 1j * branch[:, IDX.branch.BR_X])  # series admittance
    Bc = stat * branch[:, IDX.branch.BR_B]  # line charging susceptance
    tap = np.ones(nl)  # default tap ratio = 1
    i = np.nonzero(branch[:, IDX.branch.TAP])  # indices of non-zero tap ratios
    tap[i] = branch[i, IDX.branch.TAP]  # assign non-zero tap ratios
    tap = tap * np.exp(1j * deg2rad * branch[:, IDX.branch.SHIFT])  # add phase shifters

    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = - Ys / np.conj(tap)
    Ytf = - Ys / tap

    # compute shunt admittance
    # if Psh is the real power consumed by the shunt at V = 1.0 p.u.
    # and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
    # then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    # i.e. Ysh = Psh + j Qsh, so ...
    # vector of shunt admittances
    Ysh = (bus[:, IDX.bus.GS] + 1j * bus[:, IDX.bus.BS]) / baseMVA

    # build connection matrices
    f = branch[:, IDX.branch.F_BUS]  # list of "from" buses
    t = branch[:, IDX.branch.T_BUS]  # list of "to" buses
    # connection matrix for line & from buses
    Cf = c_sparse((np.ones(nl), (range(nl), f)), (nl, nb))
    # connection matrix for line & to buses
    Ct = c_sparse((np.ones(nl), (range(nl), t)), (nl, nb))

    # build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    # at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = np.r_[range(nl), range(nl)]  # double set of row indices

    Yf = c_sparse((np.r_[Yff, Yft], (i, np.r_[f, t])), (nl, nb))
    Yt = c_sparse((np.r_[Ytf, Ytt], (i, np.r_[f, t])), (nl, nb))
    # Yf = spdiags(Yff, 0, nl, nl) * Cf + spdiags(Yft, 0, nl, nl) * Ct
    # Yt = spdiags(Ytf, 0, nl, nl) * Cf + spdiags(Ytt, 0, nl, nl) * Ct

    # build Ybus
    Ybus = Cf.T * Yf + Ct.T * Yt + \
        c_sparse((Ysh, (range(nb), range(nb))), (nb, nb))

    return Ybus, Yf, Yt


def makeAy(baseMVA, ng, gencost, pgbas, qgbas, ybas):
    """Make the A matrix and RHS for the CCV formulation.

    Constructs the parameters for linear "basin constraints" on C{Pg}, C{Qg}
    and C{Y} used by the CCV cost formulation, expressed as::

         Ay * x <= by

    where C{x} is the vector of optimization variables. The starting index
    within the C{x} vector for the active, reactive sources and the C{y}
    variables should be provided in arguments C{pgbas}, C{qgbas}, C{ybas}.
    The number of generators is C{ng}.

    Assumptions: All generators are in-service.  Filter any generators
    that are offline from the C{gencost} matrix before calling L{makeAy}.
    Efficiency depends on C{Qg} variables being after C{Pg} variables, and
    the C{y} variables must be the last variables within the vector C{x} for
    the dimensions of the resulting C{Ay} to be conformable with C{x}.

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    # find all pwl cost rows in gencost, either real or reactive
    iycost = find(gencost[:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR)

    # this is the number of extra "y" variables needed to model those costs
    ny = iycost.shape[0]

    if ny == 0:
        Ay = np.zeros((0, ybas + ny - 1))  # TODO: Check size (- 1)
        by = np.array([])
        return Ay, by

    # if p(i),p(i+1),c(i),c(i+1) define one of the cost segments, then
    # the corresponding constraint on Pg (or Qg) and Y is
    # c(i+1) - c(i)
    # Y   >=   c(i) + m * (Pg - p(i)),      m = ---------------
    # p(i+1) - p(i)
    ##
    # this becomes   m * Pg - Y   <=   m*p(i) - c(i)

    # Form A matrix.  Use two different loops, one for the PG/Qg coefs,
    # then another for the y coefs so that everything is filled in the
    # same order as the compressed column sparse format used by matlab
    # this should be the quickest.

    m = sum(gencost[iycost, IDX.cost.NCOST].astype(int))  # total number of cost points
    Ay = l_sparse((m - ny, ybas + ny - 1))
    by = np.array([])
    # First fill the Pg or Qg coefficients (since their columns come first)
    # and the rhs
    k = 0
    for i in iycost:
        ns = gencost[i, IDX.cost.NCOST].astype(int)  # of cost points segments = ns-1
        p = gencost[i, IDX.cost.COST:IDX.cost.COST + 2 * ns - 1:2] / baseMVA
        c = gencost[i, IDX.cost.COST + 1:IDX.cost.COST + 2 * ns:2]
        m = np.diff(c) / np.diff(p)  # slopes for Pg (or Qg)
        if np.any(np.diff(p) == 0):
            print('makeAy: bad x axis data in row ##i of gencost matrix' % i)
        b = m * p[:ns - 1] - c[:ns - 1]  # and rhs
        by = np.r_[by,  b]
        if i > ng:
            sidx = qgbas + (i - ng) - 1  # this was for a q cost
        else:
            sidx = pgbas + i - 1  # this was for a p cost

        # FIXME: Bug in SciPy 0.7.2 prevents setting with a sequence
#        Ay[k:k + ns - 1, sidx] = m
        for ii, kk in enumerate(range(k, k + ns - 1)):
            Ay[kk, sidx] = m[ii]

        k = k + ns - 1
    # Now fill the y columns with -1's
    k = 0
    j = 0
    for i in iycost:
        ns = gencost[i, IDX.cost.NCOST].astype(int)
        # FIXME: Bug in SciPy 0.7.2 prevents setting with a sequence
#        Ay[k:k + ns - 1, ybas + j - 1] = -ones(ns - 1)
        for kk in range(k, k + ns - 1):
            Ay[kk, ybas + j - 1] = -1
        k = k + ns - 1
        j = j + 1

    return Ay.tocsr(), by
