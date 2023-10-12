"""
PYPOWER module for toggling OPF elements.
"""
import logging  # NOQA

import numpy as np  # NOQA
from numpy import flatnonzero as find  # NOQA

import scipy.sparse as sp  # NOQA
from scipy.sparse import csr_matrix as c_sparse  # NOQA
from scipy.sparse import lil_matrix as l_sparse  # NOQA

from ams.pypower.make import makeBdc  # NOQA
from ams.pypower.idx import IDX  # NOQA
from ams.pypower.routines.opffcns import (add_userfcn, remove_userfcn,
                                          e2i_field, i2e_field, i2e_data,
                                          isload)  # NOQA

from pprint import pprint


logger = logging.getLogger(__name__)


def toggle_iflims(ppc, on_off):
    """
    Enable or disable set of interface flow constraints.

    Enables or disables a set of OPF userfcn callbacks to implement
    interface flow limits based on a DC flow model.

    These callbacks expect to find an 'if' field in the input C{ppc}, where
    C{ppc['if']} is a dict with the following fields:
        - C{map}     C{n x 2}, defines each interface in terms of a set of
        branch indices and directions. Interface I is defined
        by the set of rows whose 1st col is equal to I. The
        2nd column is a branch index multiplied by 1 or -1
        respectively for lines whose orientation is the same
        as or opposite to that of the interface.
        - C{lims}    C{nif x 3}, defines the DC model flow limits in MW
        for specified interfaces. The 2nd and 3rd columns specify
        the lower and upper limits on the (DC model) flow
        across the interface, respectively. Normally, the lower
        limit is negative, indicating a flow in the opposite
        direction.

    The 'int2ext' callback also packages up results and stores them in
    the following output fields of C{results['if']}:
        - C{P}       - C{nif x 1}, actual flow across each interface in MW
        - C{mu.l}    - C{nif x 1}, shadow price on lower flow limit, ($/MW)
        - C{mu.u}    - C{nif x 1}, shadow price on upper flow limit, ($/MW)

    @see: L{add_userfcn}, L{remove_userfcn}, L{run_userfcn},
        L{t.t_case30_userfcns}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if on_off == 'on':
        # check for proper reserve inputs
        if ('if' not in ppc) | (not isinstance(ppc['if'], dict)) | \
                ('map' not in ppc['if']) | \
                ('lims' not in ppc['if']):
            logger.debug('toggle_iflims: case must contain an \'if\' field, a struct defining \'map\' and \'lims\'')

        # add callback functions
        # note: assumes all necessary data included in 1st arg (ppc, om, results)
        # so, no additional explicit args are needed
        ppc = add_userfcn(ppc, 'ext2int', userfcn_iflims_ext2int)
        ppc = add_userfcn(ppc, 'formulation', userfcn_iflims_formulation)
        ppc = add_userfcn(ppc, 'int2ext', userfcn_iflims_int2ext)
        ppc = add_userfcn(ppc, 'printpf', userfcn_iflims_printpf)
        ppc = add_userfcn(ppc, 'savecase', userfcn_iflims_savecase)
    elif on_off == 'off':
        ppc = remove_userfcn(ppc, 'savecase', userfcn_iflims_savecase)
        ppc = remove_userfcn(ppc, 'printpf', userfcn_iflims_printpf)
        ppc = remove_userfcn(ppc, 'int2ext', userfcn_iflims_int2ext)
        ppc = remove_userfcn(ppc, 'formulation', userfcn_iflims_formulation)
        ppc = remove_userfcn(ppc, 'ext2int', userfcn_iflims_ext2int)
    else:
        logger.debug('toggle_iflims: 2nd argument must be either \'on\' or \'off\'')

    return ppc


def userfcn_iflims_ext2int(ppc, *args):
    """
    This is the 'ext2int' stage userfcn callback that prepares the input
    data for the formulation stage. It expects to find an 'if' field in
    ppc as described above. The optional args are not currently used.
    """
    # initialize some things
    ifmap = ppc['if']['map']
    o = ppc['order']
    nl0 = o['ext']['branch'].shape[0]  # original number of branches
    nl = ppc['branch'].shape[0]  # number of on-line branches

    # save if.map for external indexing
    ppc['order']['ext']['ifmap'] = ifmap

    # -----  convert stuff to internal indexing  -----
    e2i = np.zeros(nl0)
    e2i[o['branch']['status']['on']] = np.arange(nl)  # ext->int branch index mapping
    d = np.sign(ifmap[:, 1])
    br = abs(ifmap[:, 1]).astype(int)
    ifmap[:, 1] = d * e2i[br]

    ifmap = np.delete(ifmap, find(ifmap[:, 1] == 0), 0)  # delete branches that are out

    ppc['if']['map'] = ifmap

    return ppc


def userfcn_iflims_formulation(om, *args):
    """
    This is the 'formulation' stage userfcn callback that defines the
    user costs and constraints for interface flow limits. It expects to
    find an 'if' field in the ppc stored in om, as described above. The
    optional args are not currently used.
    """
    # initialize some things
    ppc = om.get_ppc()
    baseMVA, bus, branch = ppc['baseMVA'], ppc['bus'], ppc['branch']
    ifmap = ppc['if']['map']
    iflims = ppc['if']['lims']

    # form B matrices for DC model
    _, Bf, _, Pfinj = makeBdc(baseMVA, bus, branch)
    n = Bf.shape[1]  # dim of theta

    # form constraints
    ifidx = np.unique(iflims[:, 0])  # interface number list
    nifs = len(ifidx)  # number of interfaces
    Aif = l_sparse((nifs, n))
    lif = np.zeros(nifs)
    uif = np.zeros(nifs)
    for k in range(nifs):
        # extract branch indices
        br = ifmap[ifmap[:, 0] == ifidx[k], 1]
        if len(br) == 0:
            logger.debug('userfcn_iflims_formulation: interface %d has no in-service branches\n' % k)

        d = np.sign(br)
        br = abs(br)
        Ak = c_sparse((1, n))  # Ak = sum( d(i) * Bf(i, :) )
        bk = 0  # bk = sum( d(i) * Pfinj(i) )
        for i in range(len(br)):
            Ak = Ak + d[i] * Bf[br[i], :]
            bk = bk + d[i] * Pfinj[br[i]]

        Aif[k, :] = Ak
        lif[k] = iflims[k, 1] / baseMVA - bk
        uif[k] = iflims[k, 2] / baseMVA - bk

    # add interface constraint
    om.add_constraints('iflims',  Aif, lif, uif, ['Va'])  # nifs

    return om


def userfcn_iflims_int2ext(results, *args):
    """
    This is the 'int2ext' stage userfcn callback that converts everything
    back to external indexing and packages up the results. It expects to
    find an 'if' field in the C{results} dict as described for ppc above.
    It also expects the results to contain solved branch flows and linear
    constraints named 'iflims' which are used to populate output fields
    in C{results['if']}. The optional args are not currently used.
    """
    # get internal ifmap
    ifmap = results['if']['map']
    iflims = results['if']['lims']

    # -----  convert stuff back to external indexing  -----
    results['if']['map'] = results['order']['ext']['ifmap']

    # -----  results post-processing  -----
    ifidx = np.unique(iflims[:, 0])  # interface number list
    nifs = len(ifidx)  # number of interfaces
    results['if']['P'] = np.zeros(nifs)
    for k in range(nifs):
        # extract branch indices
        br = ifmap[ifmap[:, 0] == ifidx[k], 1]
        d = np.sign(br)
        br = abs(br)
        results['if']['P'][k] = sum(d * results['branch'][br, IDX.branch.PF])

    if 'mu' not in results['if']:
        results['if']['mu'] = {}
    results['if']['mu']['l'] = results['lin']['mu']['l']['iflims']
    results['if']['mu']['u'] = results['lin']['mu']['u']['iflims']

    return results


def userfcn_iflims_printpf(results, fd, ppopt, *args):
    """
    This is the 'printpf' stage userfcn callback that pretty-prints the
    results. It expects a C{results} dict, a file descriptor and a PYPOWER
    options vector. The optional args are not currently used.
    """
    # -----  print results  -----
    OUT_ALL = ppopt['OUT_ALL']
    # ctol = ppopt['OPF_VIOLATION']   ## constraint violation tolerance
    ptol = 1e-6  # tolerance for displaying shadow prices

    if OUT_ALL != 0:
        iflims = results['if']['lims']
        fd.write('\n================================================================================')
        fd.write('\n|     Interface Flow Limits                                                    |')
        fd.write('\n================================================================================')
        fd.write('\n Interface  Shadow Prc  Lower Lim      Flow      Upper Lim   Shadow Prc')
        fd.write('\n     #        ($/MW)       (MW)        (MW)        (MW)       ($/MW)   ')
        fd.write('\n----------  ----------  ----------  ----------  ----------  -----------')
        ifidx = np.unique(iflims[:, 0])  # interface number list
        nifs = len(ifidx)  # number of interfaces
        for k in range(nifs):
            fd.write('\n%6d ', iflims(k, 1))
            if results['if']['mu']['l'][k] > ptol:
                fd.write('%14.3f' % results['if']['mu']['l'][k])
            else:
                fd.write('          -   ')

            fd.write('%12.2f%12.2f%12.2f' % (iflims[k, 1], results['if']['P'][k], iflims[k, 2]))
            if results['if']['mu']['u'][k] > ptol:
                fd.write('%13.3f' % results['if']['mu']['u'][k])
            else:
                fd.write('         -     ')

        fd.write('\n')

    return results


def userfcn_iflims_savecase(ppc, fd, prefix, *args):
    """
    This is the 'savecase' stage userfcn callback that prints the Python
    file code to save the 'if' field in the case file. It expects a
    PYPOWER case dict (ppc), a file descriptor and variable prefix
    (usually 'ppc'). The optional args are not currently used.
    """
    ifmap = ppc['if']['map']
    iflims = ppc['if']['lims']

    fd.write('\n####-----  Interface Flow Limit Data  -----####\n')
    fd.write('#### interface<->branch map data\n')
    fd.write('##\tifnum\tbranchidx (negative defines opposite direction)\n')
    fd.write('%sif.map = [\n' % prefix)
    fd.write('\t%d\t%d;\n' % ifmap.T)
    fd.write('];\n')

    fd.write('\n#### interface flow limit data (based on DC model)\n')
    fd.write('#### (lower limit should be negative for opposite direction)\n')
    fd.write('##\tifnum\tlower\tupper\n')
    fd.write('%sif.lims = [\n' % prefix)
    fd.write('\t%d\t%g\t%g;\n' % iflims.T)
    fd.write('];\n')

    # save output fields for solved case
    if ('P' in ppc['if']):
        fd.write('\n#### solved values\n')
        fd.write('%sif.P = %s\n' % (prefix, pprint(ppc['if']['P'])))
        fd.write('%sif.mu.l = %s\n' % (prefix, pprint(ppc['if']['mu']['l'])))
        fd.write('%sif.mu.u = %s\n' % (prefix, pprint(ppc['if']['mu']['u'])))

    return ppc


def toggle_dcline(ppc, on_off):
    """
    Enable or disable DC line modeling.

    Enables or disables a set of OPF userfcn callbacks to implement
    DC lines as a pair of linked generators. While it uses the OPF
    extension mechanism, this implementation works for simple power
    flow as well as OPF problems.

    These callbacks expect to find a 'dcline' field in the input MPC,
    where MPC.dcline is an ndc x 17 matrix with columns as defined
    in IDX_DCLINE, where ndc is the number of DC lines.

    The 'int2ext' callback also packages up flow results and stores them
    in appropriate columns of MPC.dcline.

    NOTE: Because of the way this extension modifies the number of
    rows in the gen and gencost matrices, caution must be taken
    when using it with other extensions that deal with generators.

    Examples:
        ppc = loadcase('t_case9_dcline')
        ppc = toggle_dcline(ppc, 'on')
        results1 = runpf(ppc)
        results2 = runopf(ppc)

    @see: L{idx_dcline}, L{add_userfcn}, L{remove_userfcn}, L{run_userfcn}.
    """
    if on_off == 'on':

        # check for proper input data

        if 'dcline' not in ppc or ppc['dcline'].shape[1] < IDX.dcline.LOSS1 + 1:
            raise ValueError('toggle_dcline: case must contain a '
                             '\'dcline\' field, an ndc x %d matrix.', IDX.dcline.LOSS1)

        if 'dclinecost' in ppc and ppc['dcline'].shape[0] != ppc['dclinecost'].shape[0]:
            raise ValueError('toggle_dcline: number of rows in \'dcline\''
                             ' field (%d) and \'dclinecost\' field (%d) do not match.' %
                             (ppc['dcline'].shape[0], ppc['dclinecost'].shape[0]))

        k = find(ppc['dcline'][:, IDX.dcline.LOSS1] < 0)
        if len(k) > 0:
            logger.warning('toggle_dcline: linear loss term is negative for DC line '
                           'from bus %d to %d\n' %
                           ppc['dcline'][k, IDX.dcline.F_BUS:IDX.dcline.T_BUS + 1].T)

        # add callback functions
        # note: assumes all necessary data included in 1st arg (ppc, om, results)
        # so, no additional explicit args are needed
        ppc = add_userfcn(ppc, 'ext2int', userfcn_dcline_ext2int)
        ppc = add_userfcn(ppc, 'formulation', userfcn_dcline_formulation)
        ppc = add_userfcn(ppc, 'int2ext', userfcn_dcline_int2ext)
        ppc = add_userfcn(ppc, 'printpf', userfcn_dcline_printpf)
        ppc = add_userfcn(ppc, 'savecase', userfcn_dcline_savecase)
    elif on_off == 'off':
        ppc = remove_userfcn(ppc, 'savecase', userfcn_dcline_savecase)
        ppc = remove_userfcn(ppc, 'printpf', userfcn_dcline_printpf)
        ppc = remove_userfcn(ppc, 'int2ext', userfcn_dcline_int2ext)
        ppc = remove_userfcn(ppc, 'formulation', userfcn_dcline_formulation)
        ppc = remove_userfcn(ppc, 'ext2int', userfcn_dcline_ext2int)
    else:
        raise ValueError('toggle_dcline: 2nd argument must be either '
                         '\'on\' or \'off\'')

    return ppc


# -----  ext2int  ------------------------------------------------------
def userfcn_dcline_ext2int(ppc, args):
    """This is the 'ext2int' stage userfcn callback that prepares the input
    data for the formulation stage. It expects to find a 'dcline' field
    in ppc as described above. The optional args are not currently used.
    It adds two dummy generators for each in-service DC line, with the
    appropriate upper and lower generation bounds and corresponding
    zero-cost entries in gencost.
    """
    # initialize some things
    if 'dclinecost' in ppc:
        havecost = True
    else:
        havecost = False

    # save version with external indexing
    ppc['order']['ext']['dcline'] = ppc['dcline']  # external indexing
    if havecost:
        ppc['order']['ext']['dclinecost'] = ppc['dclinecost']  # external indexing

    ppc['order']['ext']['status'] = {}
    # work with only in-service DC lines
    ppc['order']['ext']['status']['on'] = find(ppc['dcline'][:, IDX.dcline.BR_STATUS] > 0)
    ppc['order']['ext']['status']['off'] = find(ppc['dcline'][:, IDX.dcline.BR_STATUS] <= 0)

    # remove out-of-service DC lines
    dc = ppc['dcline'][ppc['order']['ext']['status']['on'], :]  # only in-service DC lines
    if havecost:
        dcc = ppc['dclinecost'][ppc['order']['ext']['status']['on'], :]  # only in-service DC lines
        ppc['dclinecost'] = dcc

    ndc = dc.shape[0]  # number of in-service DC lines
    o = ppc['order']

    # -----  convert stuff to internal indexing  -----
    dc[:, IDX.dcline.F_BUS] = o['bus']['e2i'][dc[:, IDX.dcline.F_BUS]]
    dc[:, IDX.dcline.T_BUS] = o['bus']['e2i'][dc[:, IDX.dcline.T_BUS]]
    ppc['dcline'] = dc

    # -----  create gens to represent DC line terminals  -----
    # ensure consistency of initial values of IDX.branch.PF, PT and losses
    # (for simple power flow cases)
    dc[:, IDX.dcline.PT] = dc[:, IDX.dcline.PF] - (dc[:, IDX.dcline.LOSS0] + dc[:,
                                                   IDX.dcline.LOSS1] * dc[:, IDX.dcline.PF])

    # create gens
    fg = np.zeros((ndc, ppc['gen'].shape[1]))
    fg[:, IDX.gen.MBSAE] = 100
    fg[:, IDX.gen.GEN_STATUS] = dc[:, IDX.dcline.BR_STATUS]  # status (should be all 1's)
    fg[:, IDX.gen.PMIN] = -np.Inf
    fg[:, IDX.gen.PMAX] = np.Inf
    tg = fg.copy()
    fg[:, IDX.gen.GEN_BUS] = dc[:, IDX.dcline.F_BUS]  # from bus
    tg[:, IDX.gen.GEN_BUS] = dc[:, IDX.dcline.T_BUS]  # to bus
    fg[:, IDX.gen.PG] = -dc[:, IDX.dcline.PF]  # flow (extracted at "from")
    tg[:, IDX.gen.PG] = dc[:, IDX.dcline.PT]  # flow (injected at "to")
    fg[:, IDX.gen.QG] = dc[:, IDX.dcline.QF]  # VAr injection at "from"
    tg[:, IDX.gen.QG] = dc[:, IDX.dcline.QT]  # VAr injection at "to"
    fg[:, IDX.gen.VG] = dc[:, IDX.dcline.VF]  # voltage set-point at "from"
    tg[:, IDX.gen.VG] = dc[:, IDX.dcline.VT]  # voltage set-point at "to"
    k = find(dc[:, IDX.dcline.PMIN] >= 0)  # min positive direction flow
    if len(k) > 0:  # contrain at "from" end
        fg[k, IDX.gen.PMAX] = -dc[k, IDX.dcline.PMIN]  # "from" extraction lower lim

    k = find(dc[:, IDX.dcline.IDX.gen.PMAX] >= 0)  # max positive direction flow
    if len(k) > 0:  # contrain at "from" end
        fg[k, IDX.gen.PMIN] = -dc[k, IDX.dcline.IDX.gen.PMAX]  # "from" extraction upper lim

    k = find(dc[:, IDX.dcline.PMIN] < 0)  # max negative direction flow
    if len(k) > 0:  # contrain at "to" end
        tg[k, IDX.gen.PMIN] = dc[k, IDX.dcline.PMIN]  # "to" injection lower lim

    k = find(dc[:, IDX.dcline.IDX.gen.PMAX] < 0)  # min negative direction flow
    if len(k) > 0:  # contrain at "to" end
        tg[k, IDX.gen.PMAX] = dc[k, IDX.dcline.IDX.gen.PMAX]  # "to" injection upper lim

    fg[:, IDX.gen.QMIN] = dc[:, IDX.dcline.QMINF]  # "from" VAr injection lower lim
    fg[:, IDX.gen.QMAX] = dc[:, IDX.dcline.QMAXF]  # "from" VAr injection upper lim
    tg[:, IDX.gen.QMIN] = dc[:, IDX.dcline.QMINT]  # "to"  VAr injection lower lim
    tg[:, IDX.gen.QMAX] = dc[:, IDX.dcline.QMAXT]  # "to"  VAr injection upper lim

    # fudge IDX.gen.PMAX a bit if necessary to avoid triggering
    # dispatchable load constant power factor constraints
    fg[isload(fg), IDX.gen.PMAX] = -1e-6
    tg[isload(tg), IDX.gen.PMAX] = -1e-6

    # set all terminal buses to IDX.bus.PV (except ref bus)
    refbus = find(ppc['bus'][:, IDX.bus.BUS_TYPE] == IDX.bus.REF)
    ppc['bus'][dc[:, IDX.dcline.F_BUS], IDX.bus.BUS_TYPE] = IDX.bus.PV
    ppc['bus'][dc[:, IDX.dcline.T_BUS], IDX.bus.BUS_TYPE] = IDX.bus.PV
    ppc['bus'][refbus, IDX.bus.BUS_TYPE] = IDX.bus.REF

    # append dummy gens
    ppc['gen'] = np.r_[ppc['gen'], fg, tg]

    # gencost
    if 'gencost' in ppc and len(ppc['gencost']) > 0:
        ngcr, ngcc = ppc['gencost'].shape  # dimensions of gencost
        if havecost:  # user has provided costs
            ndccc = dcc.shape[1]  # number of dclinecost columns
            ccc = max(np.r_[ngcc, ndccc])  # number of columns in new gencost
            if ccc > ngcc:  # right zero-pad gencost
                ppc.gencost = np.c_[ppc['gencost'], np.zeros(ngcr, ccc-ngcc)]

            # flip function across vertical axis and append to gencost
            # (PF for DC line = -PG for dummy gen at "from" bus)
            for k in range(ndc):
                if dcc[k, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL:
                    nc = dcc[k, IDX.cost.NCOST]
                    temp = dcc[k, IDX.cost.NCOST + range(nc + 1)]
                    # flip sign on coefficients of odd terms
                    # (every other starting with linear term,
                    # that is, the next to last one)
#                    temp((nc-1):-2:1) = -temp((nc-1):-2:1)
                    temp[range(nc, 0, -2)] = -temp[range(nc, 0, -2)]
                else:  # dcc(k, IDX.cost.MODEL) == PW_LINEAR
                    nc = dcc[k, IDX.cost.NCOST]
                    temp = dcc[k, IDX.cost.NCOST + range(2*nc + 1)]
                    # switch sign on horizontal coordinate
                    xx = -temp[range(0, 2 * nc + 1, 2)]
                    yy = temp[range(1, 2 * nc + 1, 2)]
                    temp[range(0, 2*nc + 1, 2)] = xx[-1::-1]
                    temp[range(1, 2*nc + 1, 2)] = yy[-1::-1]

                padding = np.zeros(ccc - IDX.cost.NCOST - len(temp))
                gck = np.c_[dcc[k, :IDX.cost.NCOST + 1], temp, padding]

                # append to gencost
                ppc['gencost'] = np.r_[ppc['gencost'], gck]

            # use zero cost on "to" end gen
            tgc = np.ones((ndc, 1)) * [2, 0, 0, 2, np.zeros(ccc-4)]
            ppc['gencost'] = np.c_[ppc['gencost'], tgc]
        else:
            # use zero cost as default
            dcgc = np.ones((2 * ndc, 1)) * np.concatenate([np.array([2, 0, 0, 2]), np.zeros(ngcc-4)])
            ppc['gencost'] = np.r_[ppc['gencost'], dcgc]

    return ppc


# -----  formulation  --------------------------------------------------
def userfcn_dcline_formulation(om, args):
    """
    This is the 'formulation' stage userfcn callback that defines the
    user constraints for the dummy generators representing DC lines.
    It expects to find a 'dcline' field in the ppc stored in om, as
    described above. By the time it is passed to this callback,
    MPC.dcline should contain only in-service lines and the from and
    two bus columns should be converted to internal indexing. The
    optional args are not currently used.

    If Pf, Pt and Ploss are the flow at the "from" end, flow at the
    "to" end and loss respectively, and L0 and L1 are the linear loss
    coefficients, the the relationships between them is given by:
        Pf - Ploss = Pt
        Ploss = L0 + L1 * Pf
    If Pgf and Pgt represent the injections of the dummy generators
    representing the DC line injections into the network, then
    Pgf = -Pf and Pgt = Pt, and we can combine all of the above to
    get the following constraint on Pgf ang Pgt:
        -Pgf - (L0 - L1 * Pgf) = Pgt
    which can be written:
        -L0 <= (1 - L1) * Pgf + Pgt <= -L0
    """
    # initialize some things
    ppc = om.get_ppc()
    dc = ppc['dcline']
    ndc = dc.shape[0]  # number of in-service DC lines
    ng = ppc['gen'].shape[0] - 2 * ndc  # number of original gens/disp loads

    # constraints
    nL0 = -dc[:, IDX.dcline.LOSS0] / ppc['baseMVA']
    L1 = dc[:, IDX.dcline.LOSS1]
    Adc = sp.hstack([c_sparse((ndc, ng)), sp.spdiags(1-L1, 0, ndc, ndc), sp.eye(ndc, ndc)], format="csr")

    # add them to the model
    om = om.add_constraints('dcline', Adc, nL0, nL0, ['Pg'])

    return om


# -----  int2ext  ------------------------------------------------------
def userfcn_dcline_int2ext(results, args):
    """
    This is the 'int2ext' stage userfcn callback that converts everything
    back to external indexing and packages up the results. It expects to
    find a 'dcline' field in the results struct as described for ppc
    above. It also expects that the last 2*ndc entries in the gen and
    gencost matrices correspond to the in-service DC lines (where ndc is
    the number of rows in MPC.dcline. These extra rows are removed from
    gen and gencost and the flow is taken from the PG of these gens and
    placed in the flow column of the appropiate dcline row. The
    optional args are not currently used.
    """
    # initialize some things
    o = results['order']
    k = find(o['ext']['dcline'][:, IDX.dcline.BR_STATUS])
    ndc = len(k)  # number of in-service DC lines
    ng = results['gen'].shape[0] - 2*ndc  # number of original gens/disp loads

    # extract dummy gens
    fg = results['gen'][ng:ng + ndc, :]
    tg = results['gen'][ng + ndc:ng + 2 * ndc, :]

    # remove dummy gens
    # results['gen']     = results['gen'][:ng + 1, :]
    # results['gencost'] = results['gencost'][:ng + 1, :]
    results['gen'] = results['gen'][:ng, :]
    results['gencost'] = results['gencost'][:ng, :]

    # get the solved flows
    results['dcline'][:, IDX.dcline.PF] = -fg[:, IDX.gen.PG]
    results['dcline'][:, IDX.dcline.PT] = tg[:, IDX.gen.PG]
    results['dcline'][:, IDX.dcline.QF] = fg[:, IDX.gen.QG]
    results['dcline'][:, IDX.dcline.QT] = tg[:, IDX.gen.QG]
    results['dcline'][:, IDX.dcline.VF] = fg[:, IDX.gen.VG]
    results['dcline'][:, IDX.dcline.VT] = tg[:, IDX.gen.VG]
    if fg.shape[1] >= IDX.gen.MU_QMIN:
        results['dcline'] = np.c_[results['dcline'], np.zeros((ndc, 6))]
        results['dcline'][:, IDX.dcline.MU_PMIN] = fg[:, IDX.gen.MU_PMAX] + tg[:, IDX.gen.MU_PMIN]
        results['dcline'][:, IDX.dcline.MU_PMAX] = fg[:, IDX.gen.MU_PMIN] + tg[:, IDX.gen.MU_PMAX]
        results['dcline'][:, IDX.dcline.MU_QMINF] = fg[:, IDX.gen.MU_QMIN]
        results['dcline'][:, IDX.dcline.MU_QMAXF] = fg[:, IDX.gen.MU_QMAX]
        results['dcline'][:, IDX.dcline.MU_QMINT] = tg[:, IDX.gen.MU_QMIN]
        results['dcline'][:, IDX.dcline.MU_QMAXT] = tg[:, IDX.gen.MU_QMAX]

    results['order']['int'] = {}
    # -----  convert stuff back to external indexing  -----
    results['order']['int']['dcline'] = results['dcline']  # save internal version
    # copy results to external version
    o['ext']['dcline'][k, IDX.dcline.PF:c['VT'] + 1] = results['dcline'][:, IDX.dcline.PF:c['VT'] + 1]
    if results['dcline'].shape[1] == IDX.dcline.MU_QMAXT + 1:
        o['ext']['dcline'] = np.c_[o['ext']['dcline'], np.zeros((ndc, 6))]
        o['ext']['dcline'][k, IDX.dcline.MU_PMIN:IDX.dcline.MU_QMAXT + 1] = \
            results['dcline'][:, IDX.dcline.MU_PMIN:IDX.dcline.MU_QMAXT + 1]

    results['dcline'] = o['ext']['dcline']  # use external version

    return results


# -----  printpf  ------------------------------------------------------
def userfcn_dcline_printpf(results, fd, ppopt, args):
    """
    This is the 'printpf' stage userfcn callback that pretty-prints the
    results. It expects a results struct, a file descriptor and a MATPOWER
    options vector. The optional args are not currently used.
    """
    # options
    OUT_ALL = ppopt['OUT_ALL']
    OUT_BRANCH = OUT_ALL == 1 or (OUT_ALL == -1 and ppopt['OUT_BRANCH'])
    if OUT_ALL == -1:
        OUT_ALL_LIM = ppopt['OUT_ALL_LIM']
    elif OUT_ALL == 1:
        OUT_ALL_LIM = 2
    else:
        OUT_ALL_LIM = 0

    if OUT_ALL_LIM == -1:
        OUT_LINE_LIM = ppopt['OUT_LINE_LIM']
    else:
        OUT_LINE_LIM = OUT_ALL_LIM

    ctol = ppopt['OPF_VIOLATION']  # constraint violation tolerance
    ptol = 1e-4  # tolerance for displaying shadow prices

    # -----  print results  -----
    dc = results['dcline']
    ndc = dc.shape[0]
    kk = find(dc[:, IDX.dcline.BR_STATUS] != 0)
    if OUT_BRANCH:
        fd.write('\n================================================================================')
        fd.write('\n|     DC Line Data                                                             |')
        fd.write('\n================================================================================')
        fd.write('\n Line    From     To        Power Flow           Loss     Reactive Inj (MVAr)')
        fd.write('\n   #      Bus     Bus   From (MW)   To (MW)      (MW)       From        To   ')
        fd.write('\n------  ------  ------  ---------  ---------  ---------  ---------  ---------')
        loss = 0
        for k in range(ndc):
            if dc[k, IDX.dcline.BR_STATUS]:  # status on
                fd.write(
                    '\n{0:5.0f}{1:8.0f}{2:8.0f}{3:11.2f}{4:11.2f}{5:11.2f}{6:11.2f}{7:11.2f}'.format(
                        *np.r_
                        [k, dc[k, IDX.dcline.F_BUS: IDX.dcline.T_BUS + 1],
                         dc[k, IDX.dcline.PF: IDX.dcline.PT + 1],
                         dc[k, IDX.dcline.PF] - dc[k, IDX.dcline.PT],
                         dc[k, IDX.dcline.QF: IDX.dcline.QT + 1]]))

                loss = loss + dc[k, IDX.dcline.PF] - dc[k, IDX.dcline.PT]
            else:
                fd.write('\n%5d%8d%8d%11s%11s%11s%11s%11s' %
                         (k, dc[k, IDX.dcline.F_BUS:IDX.dcline.T_BUS + 1], '-  ', '-  ', '-  ', '-  ', '-  '))

        fd.write('\n                                              ---------')
        fd.write('\n                                     Total:{0:11.2f}\n'.format(loss))

    if OUT_LINE_LIM == 2 or (OUT_LINE_LIM == 1 and
                             (np.any(dc[kk, IDX.dcline.PF] > dc[kk, IDX.dcline.PMAX] - ctol) or
                              np.any(dc[kk, IDX.dcline.MU_PMIN] > ptol) or
                                 np.any(dc[kk, IDX.dcline.MU_PMAX] > ptol))):
        fd.write('\n================================================================================')
        fd.write('\n|     DC Line Constraints                                                      |')
        fd.write('\n================================================================================')
        fd.write('\n Line    From     To          Minimum        Actual Flow       Maximum')
        fd.write('\n   #      Bus     Bus    Pmin mu     Pmin       (MW)       Pmax      Pmax mu ')
        fd.write('\n------  ------  ------  ---------  ---------  ---------  ---------  ---------')
        for k in range(ndc):
            if OUT_LINE_LIM == 2 or (OUT_LINE_LIM == 1 and
                                     (dc[k, IDX.dcline.PF] > dc[k, IDX.dcline.PMAX] - ctol or
                                      dc[k, IDX.dcline.MU_PMIN] > ptol or
                                         dc[k, IDX.dcline.MU_PMAX] > ptol)):
                if dc[k, IDX.dcline.BR_STATUS]:  # status on
                    fd.write('\n{0:5.0f}{1:8.0f}{2:8.0f}'.format(
                        *np.r_[k, dc[k, IDX.dcline.F_BUS:IDX.dcline.T_BUS + 1]]))
                    # fd.write('\n%5d%8d%8d' % (k + 1, dc[k, IDX.dcline.F_BUS:IDX.dcline.T_BUS + 1] ))
                    if dc[k, IDX.dcline.MU_PMIN] > ptol:
                        fd.write('{0:11.3f}'.format(dc[k, IDX.dcline.MU_PMIN]))
                    else:
                        fd.write('%11s' % ('-  '))

                    fd.write('{0:11.2f}{1:11.2f}{2:11.2f}'
                             .format(*np.r_[dc[k, IDX.dcline.PMIN], dc[k, IDX.dcline.PF], dc[k, IDX.dcline.PMAX]]))
                    if dc[k, IDX.dcline.MU_PMAX] > ptol:
                        fd.write('{0:11.3f}'.format(dc[k, IDX.dcline.MU_PMAX]))
                    else:
                        fd.write('%11s' % ('-  '))

                else:
                    fd.write('\n%5d%8d%8d%11s%11s%11s%11s%11s' %
                             (k, dc[k, IDX.dcline.F_BUS:IDX.dcline.T_BUS + 1], '-  ', '-  ', '-  ', '-  ', '-  '))

        fd.write('\n')

    return results


# -----  savecase  -----------------------------------------------------
def userfcn_dcline_savecase(ppc, fd, prefix, args):
    """
    This is the 'savecase' stage userfcn callback that prints the Py-file
    code to save the 'dcline' field in the case file. It expects a
    PYPOWER case dict (ppc), a file descriptor and variable prefix
    (usually 'ppc.'). The optional args are not currently used.
    """
    # save it
    ncols = ppc['dcline'].shape[1]
    fd.write('\n####-----  DC Line Data  -----####\n')
    if ncols < IDX.dcline.MU_QMAXT:
        fd.write('##\tfbus\ttbus\tstatus\tPf\tPt\tQf\tQt\tVf\tVt\tPmin\tPmax\tQminF\tQmaxF\tQminT\tQmaxT\tloss0\tloss1\n')
    else:
        fd.write('##\tfbus\ttbus\tstatus\tPf\tPt\tQf\tQt\tVf\tVt\tPmin\tPmax\tQminF\tQmaxF\tQminT\tQmaxT\tloss0\tloss1\tmuPmin\tmuPmax\tmuQminF\tmuQmaxF\tmuQminT\tmuQmaxT\n')

    template = '\t%d\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g'
    if ncols == IDX.dcline.MU_QMAXT + 1:
        template = [template, '\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f']

    template = template + ';\n'
    fd.write('%sdcline = [\n' % prefix)
    fd.write(template, ppc['dcline'].T)
    fd.write('];\n')

    return ppc


def toggle_reserves(ppc, on_off):
    """
    Enable or disable fixed reserve requirements.

    Enables or disables a set of OPF userfcn callbacks to implement
    co-optimization of reserves with fixed zonal reserve requirements.

    These callbacks expect to find a 'reserves' field in the input C{ppc},
    where C{ppc['reserves']} is a dict with the following fields:
        - C{zones}   C{nrz x ng}, C{zone(i, j) = 1}, if gen C{j} belongs
        to zone C{i} 0, otherwise
        - C{req}     C{nrz x 1}, zonal reserve requirement in MW
        - C{cost}    (C{ng} or C{ngr}) C{x 1}, cost of reserves in $/MW
        - C{qty}     (C{ng} or C{ngr}) C{x 1}, max quantity of reserves
        in MW (optional)
    where C{nrz} is the number of reserve zones and C{ngr} is the number of
    generators belonging to at least one reserve zone and C{ng} is the total
    number of generators.

    The 'int2ext' callback also packages up results and stores them in
    the following output fields of C{results['reserves']}:
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

    @see: L{runopf_w_res}, L{add_userfcn}, L{remove_userfcn}, L{run_userfcn},
        L{t.t_case30_userfcns}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if on_off == 'on':
        # check for proper reserve inputs
        if ('reserves' not in ppc) | (not isinstance(ppc['reserves'], dict)) | \
                ('zones' not in ppc['reserves']) | \
                ('req' not in ppc['reserves']) | \
                ('cost' not in ppc['reserves']):
            logger.debug(
                'toggle_reserves: case must contain a \'reserves\' field, a struct defining \'zones\', \'req\' and \'cost\'\n')

        # add callback functions
        # note: assumes all necessary data included in 1st arg (ppc, om, results)
        # so, no additional explicit args are needed
        ppc = add_userfcn(ppc, 'ext2int', userfcn_reserves_ext2int)
        ppc = add_userfcn(ppc, 'formulation', userfcn_reserves_formulation)
        ppc = add_userfcn(ppc, 'int2ext', userfcn_reserves_int2ext)
        ppc = add_userfcn(ppc, 'printpf', userfcn_reserves_printpf)
        ppc = add_userfcn(ppc, 'savecase', userfcn_reserves_savecase)
    elif on_off == 'off':
        ppc = remove_userfcn(ppc, 'savecase', userfcn_reserves_savecase)
        ppc = remove_userfcn(ppc, 'printpf', userfcn_reserves_printpf)
        ppc = remove_userfcn(ppc, 'int2ext', userfcn_reserves_int2ext)
        ppc = remove_userfcn(ppc, 'formulation', userfcn_reserves_formulation)
        ppc = remove_userfcn(ppc, 'ext2int', userfcn_reserves_ext2int)
    else:
        logger.debug('toggle_reserves: 2nd argument must be either ''on'' or ''off''')

    return ppc


def userfcn_reserves_ext2int(ppc, *args):
    """
    This is the 'ext2int' stage userfcn callback that prepares the input
    data for the formulation stage. It expects to find a 'reserves' field
    in ppc as described above. The optional args are not currently used.
    """
    # initialize some things
    r = ppc['reserves']
    o = ppc['order']
    ng0 = o['ext']['gen'].shape[0]  # number of original gens (+ disp loads)
    nrz = r['req'].shape[0]  # number of reserve zones
    if nrz > 1:
        ppc['reserves']['rgens'] = np.any(r['zones'], 0)  # mask of gens available to provide reserves
    else:
        ppc['reserves']['rgens'] = r['zones']

    igr = find(ppc['reserves']['rgens'])  # indices of gens available to provide reserves
    ngr = len(igr)  # number of gens available to provide reserves

    # check data for consistent dimensions
    if r['zones'].shape[0] != nrz:
        logger.debug('userfcn_reserves_ext2int: the number of rows in ppc[\'reserves\'][\'req\'] (%d) and ppc[\'reserves\'][\'zones\'] (%d) must match\n' % (
            nrz, r['zones'].shape[0]))

    if (r['cost'].shape[0] != ng0) & (r['cost'].shape[0] != ngr):
        logger.debug('userfcn_reserves_ext2int: the number of rows in ppc[\'reserves\'][\'cost\'] (%d) must equal the total number of generators (%d) or the number of generators able to provide reserves (%d)\n' % (
            r['cost'].shape[0], ng0, ngr))

    if 'qty' in r:
        if r['qty'].shape[0] != r['cost'].shape[0]:
            logger.debug('userfcn_reserves_ext2int: ppc[\'reserves\'][\'cost\'] (%d x 1) and ppc[\'reserves\'][\'qty\'] (%d x 1) must be the same dimension\n' % (
                r['cost'].shape[0], r['qty'].shape[0]))

    # convert both cost and qty from ngr x 1 to full ng x 1 vectors if necessary
    if r['cost'].shape[0] < ng0:
        if 'original' not in ppc['reserves']:
            ppc['reserves']['original'] = {}
        ppc['reserves']['original']['cost'] = r['cost'].copy()  # save original
        cost = np.zeros(ng0)
        cost[igr] = r['cost']
        ppc['reserves']['cost'] = cost
        if 'qty' in r:
            ppc['reserves']['original']['qty'] = r['qty'].copy()  # save original
            qty = np.zeros(ng0)
            qty[igr] = r['qty']
            ppc['reserves']['qty'] = qty

    # -----  convert stuff to internal indexing  -----
    # convert all reserve parameters (zones, costs, qty, rgens)
    if 'qty' in r:
        ppc = e2i_field(ppc, ['reserves', 'qty'], 'gen')

    ppc = e2i_field(ppc, ['reserves', 'cost'], 'gen')
    ppc = e2i_field(ppc, ['reserves', 'zones'], 'gen', 1)
    ppc = e2i_field(ppc, ['reserves', 'rgens'], 'gen', 1)

    # save indices of gens available to provide reserves
    ppc['order']['ext']['reserves']['igr'] = igr  # external indexing
    ppc['reserves']['igr'] = find(ppc['reserves']['rgens'])  # internal indexing

    return ppc


def userfcn_reserves_formulation(om, *args):
    """
    This is the 'formulation' stage userfcn callback that defines the
    user costs and constraints for fixed reserves. It expects to find
    a 'reserves' field in the ppc stored in om, as described above.
    By the time it is passed to this callback, ppc['reserves'] should
    have two additional fields:
        - C{igr}     C{1 x ngr}, indices of generators available for reserves
        - C{rgens}   C{1 x ng}, 1 if gen avaiable for reserves, 0 otherwise
    It is also assumed that if cost or qty were C{ngr x 1}, they have been
    expanded to C{ng x 1} and that everything has been converted to
    internal indexing, i.e. all gens are on-line (by the 'ext2int'
    callback). The optional args are not currently used.
    """
    # initialize some things
    ppc = om.get_ppc()
    r = ppc['reserves']
    igr = r['igr']  # indices of gens available to provide reserves
    ngr = len(igr)  # number of gens available to provide reserves
    ng = ppc['gen'].shape[0]  # number of on-line gens (+ disp loads)

    # variable bounds
    Rmin = np.zeros(ngr)  # bound below by 0
    Rmax = np.Inf * np.ones(ngr)  # bound above by ...
    k = find(ppc['gen'][igr, IDX.gen.RAMP_10])
    Rmax[k] = ppc['gen'][igr[k], IDX.gen.RAMP_10]  # ... ramp rate and ...
    if 'qty' in r:
        k = find(r['qty'][igr] < Rmax)
        Rmax[k] = r['qty'][igr[k]]  # ... stated max reserve qty
    Rmax = Rmax / ppc['baseMVA']

    # constraints
    I = sp.eye(ngr, ngr, format='csr')  # identity matrix
    Ar = sp.hstack([c_sparse((np.ones(ngr), (np.arange(ngr), igr)), (ngr, ng)), I], 'csr')
    ur = ppc['gen'][igr, IDX.gen.PMAX] / ppc['baseMVA']
    lreq = r['req'] / ppc['baseMVA']

    # cost
    Cw = r['cost'][igr] * ppc['baseMVA']  # per unit cost coefficients

    # add them to the model
    om.add_vars('R', ngr, [], Rmin, Rmax)
    om.add_constraints('Pg_plus_R', Ar, [], ur, ['Pg', 'R'])
    om.add_constraints('Rreq', c_sparse(r['zones'][:, igr]), lreq, [], ['R'])
    om.add_costs('Rcost', {'N': I, 'Cw': Cw}, ['R'])

    return om


def userfcn_reserves_int2ext(results, *args):
    """
    This is the 'int2ext' stage userfcn callback that converts everything
    back to external indexing and packages up the results. It expects to
    find a 'reserves' field in the results struct as described for ppc
    above, including the two additional fields 'igr' and 'rgens'. It also
    expects the results to contain a variable 'R' and linear constraints
    'Pg_plus_R' and 'Rreq' which are used to populate output fields in
    results.reserves. The optional args are not currently used.
    """
    # initialize some things
    r = results['reserves']

    # grab some info in internal indexing order
    igr = r['igr']  # indices of gens available to provide reserves
    ng = results['gen'].shape[0]  # number of on-line gens (+ disp loads)

    # -----  convert stuff back to external indexing  -----
    # convert all reserve parameters (zones, costs, qty, rgens)
    if 'qty' in r:
        results = i2e_field(results, ['reserves', 'qty'], ordering='gen')

    results = i2e_field(results, ['reserves', 'cost'], ordering='gen')
    results = i2e_field(results, ['reserves', 'zones'], ordering='gen', dim=1)
    results = i2e_field(results, ['reserves', 'rgens'], ordering='gen', dim=1)
    results['order']['int']['reserves']['igr'] = results['reserves']['igr']  # save internal version
    results['reserves']['igr'] = results['order']['ext']['reserves']['igr']  # use external version
    r = results['reserves']  # update
    o = results['order']  # update

    # grab same info in external indexing order
    igr0 = r['igr']  # indices of gens available to provide reserves
    ng0 = o['ext']['gen'].shape[0]  # number of gens (+ disp loads)

    # -----  results post-processing  -----
    # get the results (per gen reserves, multipliers) with internal gen indexing
    # and convert from p.u. to per MW units
    _, Rl, Ru = results['om'].getv('R')
    R = np.zeros(ng)
    Rmin = np.zeros(ng)
    Rmax = np.zeros(ng)
    mu_l = np.zeros(ng)
    mu_u = np.zeros(ng)
    mu_Pmax = np.zeros(ng)
    R[igr] = results['var']['val']['R'] * results['baseMVA']
    Rmin[igr] = Rl * results['baseMVA']
    Rmax[igr] = Ru * results['baseMVA']
    mu_l[igr] = results['var']['mu']['l']['R'] / results['baseMVA']
    mu_u[igr] = results['var']['mu']['u']['R'] / results['baseMVA']
    mu_Pmax[igr] = results['lin']['mu']['u']['Pg_plus_R'] / results['baseMVA']

    # store in results in results struct
    z = np.zeros(ng0)
    results['reserves']['R'] = i2e_data(results, R, z, 'gen')
    results['reserves']['Rmin'] = i2e_data(results, Rmin, z, 'gen')
    results['reserves']['Rmax'] = i2e_data(results, Rmax, z, 'gen')
    if 'mu' not in results['reserves']:
        results['reserves']['mu'] = {}
    results['reserves']['mu']['l'] = i2e_data(results, mu_l, z, 'gen')
    results['reserves']['mu']['u'] = i2e_data(results, mu_u, z, 'gen')
    results['reserves']['mu']['Pmax'] = i2e_data(results, mu_Pmax, z, 'gen')
    results['reserves']['prc'] = z
    for k in igr0:
        iz = find(r['zones'][:, k])
        results['reserves']['prc'][k] = sum(results['lin']['mu']['l']['Rreq'][iz]) / results['baseMVA']

    results['reserves']['totalcost'] = results['cost']['Rcost']

    # replace ng x 1 cost, qty with ngr x 1 originals
    if 'original' in r:
        if 'qty' in r:
            results['reserves']['qty'] = r['original']['qty']
        results['reserves']['cost'] = r['original']['cost']
        del results['reserves']['original']

    return results


def userfcn_reserves_printpf(results, fd, ppopt, *args):
    """
    This is the 'printpf' stage userfcn callback that pretty-prints the
    results. It expects a C{results} dict, a file descriptor and a PYPOWER
    options vector. The optional args are not currently used.
    """
    # -----  print results  -----
    r = results['reserves']
    nrz = r['req'].shape[0]
    OUT_ALL = ppopt['OUT_ALL']
    if OUT_ALL != 0:
        fd.write('\n================================================================================')
        fd.write('\n|     Reserves                                                                 |')
        fd.write('\n================================================================================')
        fd.write('\n Gen   Bus   Status  Reserves   Price')
        fd.write('\n  #     #              (MW)     ($/MW)     Included in Zones ...')
        fd.write('\n----  -----  ------  --------  --------   ------------------------')
        for k in r['igr']:
            iz = find(r['zones'][:, k])
            fd.write('\n%3d %6d     %2d ' %
                     (k, results['gen'][k, IDX.gen.GEN_BUS],
                      results['gen'][k, IDX.gen.GEN_STATUS]))
            if (results['gen'][k, IDX.gen.GEN_STATUS] > 0) & (abs(results['reserves']['R'][k]) > 1e-6):
                fd.write('%10.2f' % results['reserves']['R'][k])
            else:
                fd.write('       -  ')

            fd.write('%10.2f     ' % results['reserves']['prc'][k])
            for i in range(len(iz)):
                if i != 0:
                    fd.write(', ')
                fd.write('%d' % iz[i])

        fd.write('\n                     --------')
        fd.write('\n            Total:%10.2f              Total Cost: $%.2f' %
                 (sum(results['reserves']['R'][r['igr']]), results['reserves']['totalcost']))
        fd.write('\n')

        fd.write('\nZone  Reserves   Price  ')
        fd.write('\n  #     (MW)     ($/MW) ')
        fd.write('\n----  --------  --------')
        for k in range(nrz):
            iz = find(r['zones'][k, :])  # gens in zone k
            fd.write('\n%3d%10.2f%10.2f' % (k, sum(results['reserves']['R'][iz]),
                                            results['lin']['mu']['l']['Rreq'][k] / results['baseMVA']))
        fd.write('\n')

        fd.write('\n================================================================================')
        fd.write('\n|     Reserve Limits                                                           |')
        fd.write('\n================================================================================')
        fd.write('\n Gen   Bus   Status  Rmin mu     Rmin    Reserves    Rmax    Rmax mu   Pmax mu ')
        fd.write('\n  #     #             ($/MW)     (MW)      (MW)      (MW)     ($/MW)    ($/MW) ')
        fd.write('\n----  -----  ------  --------  --------  --------  --------  --------  --------')
        for k in r['igr']:
            fd.write('\n%3d %6d     %2d ' %
                     (k, results['gen'][k, IDX.gen.GEN_BUS],
                      results['gen'][k, IDX.gen.GEN_STATUS]))
            if (results['gen'][k, IDX.gen.GEN_STATUS] > 0) & (results['reserves']['mu']['l'][k] > 1e-6):
                fd.write('%10.2f' % results['reserves']['mu']['l'][k])
            else:
                fd.write('       -  ')

            fd.write('%10.2f' % results['reserves']['Rmin'][k])
            if (results['gen'][k, IDX.gen.GEN_STATUS] > 0) & (abs(results['reserves']['R'][k]) > 1e-6):
                fd.write('%10.2f' % results['reserves']['R'][k])
            else:
                fd.write('       -  ')

            fd.write('%10.2f' % results['reserves']['Rmax'][k])
            if (results['gen'][k, IDX.gen.GEN_STATUS] > 0) & (results['reserves']['mu']['u'][k] > 1e-6):
                fd.write('%10.2f' % results['reserves']['mu']['u'][k])
            else:
                fd.write('       -  ')

            if (results['gen'][k, IDX.gen.GEN_STATUS] > 0) & (results['reserves']['mu']['Pmax'][k] > 1e-6):
                fd.write('%10.2f' % results['reserves']['mu']['Pmax'][k])
            else:
                fd.write('       -  ')

        fd.write('\n                                         --------')
        fd.write('\n                                Total:%10.2f' % sum(results['reserves']['R'][r['igr']]))
        fd.write('\n')

    return results


def userfcn_reserves_savecase(ppc, fd, prefix, *args):
    """
    This is the 'savecase' stage userfcn callback that prints the Python
    file code to save the 'reserves' field in the case file. It expects a
    PYPOWER case dict (ppc), a file descriptor and variable prefix
    (usually 'ppc'). The optional args are not currently used.
    """
    r = ppc['reserves']

    fd.write('\n####-----  Reserve Data  -----####\n')
    fd.write('#### reserve zones, element i, j is 1 if gen j is in zone i, 0 otherwise\n')
    fd.write('%sreserves.zones = [\n' % prefix)
    template = ''
    for _ in range(r['zones'].shape[1]):
        template = template + '\t%d'
    template = template + ';\n'
    fd.write(template, r.zones.T)
    fd.write('];\n')

    fd.write('\n#### reserve requirements for each zone in MW\n')
    fd.write('%sreserves.req = [\t%g' % (prefix, r['req'][0]))
    if len(r['req']) > 1:
        fd.write(';\t%g' % r['req'][1:])
    fd.write('\t];\n')

    fd.write('\n#### reserve costs in $/MW for each gen that belongs to at least 1 zone\n')
    fd.write('#### (same order as gens, but skipping any gen that does not belong to any zone)\n')
    fd.write('%sreserves.cost = [\t%g' % (prefix, r['cost'][0]))
    if len(r['cost']) > 1:
        fd.write(';\t%g' % r['cost'][1:])
    fd.write('\t];\n')

    if 'qty' in r:
        fd.write('\n#### OPTIONAL max reserve quantities for each gen that belongs to at least 1 zone\n')
        fd.write('#### (same order as gens, but skipping any gen that does not belong to any zone)\n')
        fd.write('%sreserves.qty = [\t%g' % (prefix, r['qty'][0]))
        if len(r['qty']) > 1:
            fd.write(';\t%g' % r['qty'][1:])
        fd.write('\t];\n')

    # save output fields for solved case
    if 'R' in r:
        fd.write('\n#### solved values\n')
        fd.write('%sreserves.R = %s\n' % (prefix, pprint(r['R'])))
        fd.write('%sreserves.Rmin = %s\n' % (prefix, pprint(r['Rmin'])))
        fd.write('%sreserves.Rmax = %s\n' % (prefix, pprint(r['Rmax'])))
        fd.write('%sreserves.mu.l = %s\n' % (prefix, pprint(r['mu']['l'])))
        fd.write('%sreserves.mu.u = %s\n' % (prefix, pprint(r['mu']['u'])))
        fd.write('%sreserves.prc = %s\n' % (prefix, pprint(r['prc'])))
        fd.write('%sreserves.totalcost = %s\n' % (prefix, pprint(r['totalcost'])))

    return ppc
