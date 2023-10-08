"""
Internal and external data conversion functions.
"""
import logging

from copy import deepcopy
import numpy as np
from numpy import flatnonzero as find
from numpy import argsort
from scipy.sparse import issparse, vstack, hstack
from scipy.sparse import csr_matrix as sparse

from ams.pypower.get_reorder import get_reorder
from ams.pypower.set_reorder import set_reorder
from ams.pypower.run_userfcn import run_userfcn
import ams.pypower.utils.constants as pidx

logger = logging.getLogger(__name__)


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
    from ams.pypower.int2ext import int2ext

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
            ppc["bus"][o["bus"]["status"]["on"], pidx.bus['BUS_I']] = \
                o["bus"]["i2e"][ppc["bus"][o["bus"]["status"]["on"], pidx.bus['BUS_I']].astype(int)]
            ppc["branch"][o["branch"]["status"]["on"], pidx.branch['F_BUS']] = \
                o["bus"]["i2e"][ppc["branch"]
                                [o["branch"]["status"]["on"], pidx.branch['F_BUS']].astype(int)]
            ppc["branch"][o["branch"]["status"]["on"], pidx.branch['T_BUS']] = \
                o["bus"]["i2e"][ppc["branch"]
                                [o["branch"]["status"]["on"], pidx.branch['T_BUS']].astype(int)]
            ppc["gen"][o["gen"]["status"]["on"], pidx.gen['GEN_BUS']] = \
                o["bus"]["i2e"][ppc["gen"]
                                [o["gen"]["status"]["on"], pidx.gen['GEN_BUS']].astype(int)]
            if 'areas' in ppc:
                ppc["areas"][o["areas"]["status"]["on"], idx.area['PRICE_REF_BUS']] = \
                    o["bus"]["i2e"][ppc["areas"]
                                    [o["areas"]["status"]["on"], idx.area['PRICE_REF_BUS']].astype(int)]

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
            logger.warning('Calls of the form VAL = INT2EXT(MPC, VAL, ...) have been deprecated. Please replace INT2EXT with I2E_DATA.')
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
    bus[:, pidx.bus['BUS_I']] = i2e[bus[:, pidx.bus['BUS_I']].astype(int)]
    gen[:, pidx.gen['GEN_BUS']] = i2e[gen[:, pidx.gen['GEN_BUS']].astype(int)]
    branch[:, pidx.branch['F_BUS']] = i2e[branch[:, pidx.branch['F_BUS']].astype(int)]
    branch[:, pidx.branch['T_BUS']] = i2e[branch[:, pidx.branch['T_BUS']].astype(int)]

    if areas != None and len(areas) > 0:
        areas[:, idx.area['PRICE_REF_BUS']] = i2e[areas[:, idx.area['PRICE_REF_BUS']].astype(int)]
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

        if issparse(new_v[0]):
            if dim == 0:
                vstack(new_v, 'csr')
            elif dim == 1:
                hstack(new_v, 'csr')
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
            bt = ppc["bus"][:, pidx.bus['BUS_TYPE']]
            err = find(~((bt == pidx.bus['PQ']) | (bt == pidx.bus['PV']) |
                       (bt == pidx.bus['REF']) | (bt == pidx.bus['NONE'])))
            if len(err) > 0:
                logger.debug('ext2int: bus %d has an invalid BUS_TYPE\n' % err)

            # determine which buses, branches, gens are connected and
            # in-service
            n2i = sparse((range(nb), (ppc["bus"][:, pidx.bus['BUS_I']], np.zeros(nb))),
                         shape=(max(ppc["bus"][:, pidx.bus['BUS_I']].astype(int)) + 1, 1))
            n2i = (np.array(n2i.todense().flatten())[0, :]).astype(int)  # as 1D array
            bs = (bt != pidx.bus['NONE'])  # bus status
            o["bus"]["status"]["on"] = find(bs)  # connected
            o["bus"]["status"]["off"] = find(~bs)  # isolated
            gs = ((ppc["gen"][:, pidx.gen['GEN_STATUS']] > 0) &  # gen status
                  bs[n2i[ppc["gen"][:, pidx.gen['GEN_BUS']].astype(int)]])
            o["gen"]["status"]["on"] = find(gs)  # on and connected
            o["gen"]["status"]["off"] = find(~gs)  # off or isolated
            brs = (ppc["branch"][:, pidx.branch['BR_STATUS']].astype(int) &  # branch status
                   bs[n2i[ppc["branch"][:, pidx.branch['F_BUS']].astype(int)]] &
                   bs[n2i[ppc["branch"][:, pidx.branch['T_BUS']].astype(int)]]).astype(bool)
            o["branch"]["status"]["on"] = find(brs)  # on and conn
            o["branch"]["status"]["off"] = find(~brs)
            if 'areas' in ppc:
                ar = bs[n2i[ppc["areas"][:, idx.area['PRICE_REF_BUS']].astype(int)]]
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
            o["bus"]["i2e"] = ppc["bus"][:, pidx.bus['BUS_I']].copy()
            o["bus"]["e2i"] = np.zeros(max(o["bus"]["i2e"]).astype(int) + 1)
            o["bus"]["e2i"][o["bus"]["i2e"].astype(int)] = np.arange(nb)
            ppc["bus"][:, pidx.bus['BUS_I']] = \
                o["bus"]["e2i"][ppc["bus"][:, pidx.bus['BUS_I']].astype(int)].copy()
            ppc["gen"][:, pidx.gen['GEN_BUS']] = \
                o["bus"]["e2i"][ppc["gen"][:, pidx.gen['GEN_BUS']].astype(int)].copy()
            ppc["branch"][:, pidx.branch['F_BUS']] = \
                o["bus"]["e2i"][ppc["branch"][:, pidx.branch['F_BUS']].astype(int)].copy()
            ppc["branch"][:, pidx.branch['T_BUS']] = \
                o["bus"]["e2i"][ppc["branch"][:, pidx.branch['T_BUS']].astype(int)].copy()
            if 'areas' in ppc:
                ppc["areas"][:, idx.area['PRICE_REF_BUS']] = \
                    o["bus"]["e2i"][ppc["areas"][:,
                                                 idx.area['PRICE_REF_BUS']].astype(int)].copy()

            # reorder gens in order of increasing bus number
            o["gen"]["e2i"] = argsort(ppc["gen"][:, pidx.gen['GEN_BUS']])
            o["gen"]["i2e"] = argsort(o["gen"]["e2i"])

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
    i2e = bus[:, pidx.bus['BUS_I']].astype(int)
    e2i = np.zeros(max(i2e) + 1)
    e2i[i2e] = np.arange(bus.shape[0])

    bus[:, pidx.bus['BUS_I']] = e2i[bus[:, pidx.bus['BUS_I']].astype(int)]
    gen[:, pidx.gen['GEN_BUS']] = e2i[gen[:, pidx.gen['GEN_BUS']].astype(int)]
    branch[:, pidx.branch['F_BUS']] = e2i[branch[:, pidx.branch['F_BUS']].astype(int)]
    branch[:, pidx.branch['T_BUS']] = e2i[branch[:, pidx.branch['T_BUS']].astype(int)]
    if areas is not None and len(areas) > 0:
        areas[:, idx.area['PRICE_REF_BUS']] = e2i[areas[:, idx.area['PRICE_REF_BUS']].astype(int)]

        return i2e, bus, gen, branch, areas

    return i2e, bus, gen, branch
