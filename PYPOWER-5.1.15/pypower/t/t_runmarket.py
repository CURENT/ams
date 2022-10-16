# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Tests for code in C{runmkt}, C{smartmkt} and C{auction}.
"""

from numpy import array, ones, flatnonzero as find

from scipy.sparse import csr_matrix as sparse

from pypower.ppoption import ppoption
from pypower.loadcase import loadcase
from pypower.isload import isload

from pypower.idx_bus import BUS_I, LAM_P, LAM_Q
from pypower.idx_gen import GEN_BUS

from pypower.t.t_begin import t_begin
from pypower.t.t_is import t_is
from pypower.t.t_skip import t_skip
from pypower.t.t_end import t_end


def t_runmarket(quiet=False):
    """Tests for code in C{runmkt}, C{smartmkt} and C{auction}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    n_tests = 20

    t_begin(n_tests, quiet)

    try:
        from pypower.extras.smartmarket import runmarket
    except ImportError:
        t_skip(n_tests, 'smartmarket code not available')
        t_end;
        return

    ppc = loadcase('t_auction_case')

    ppopt = ppoption(OPF_ALG=560, OUT_ALL_LIM=1,
                     OUT_BRANCH=0, OUT_SYS_SUM=0)
    ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=1)
    #ppopt = ppoption(ppopt, OUT_GEN=1, OUT_BRANCH=0, OUT_SYS_SUM=0)

    offers = {'P': {}, 'Q': {}}
    bids = {'P': {}, 'Q': {}}

    offers['P']['qty'] = array([
        [12, 24, 24],
        [12, 24, 24],
        [12, 24, 24],
        [12, 24, 24],
        [12, 24, 24],
        [12, 24, 24]
    ])
    offers['P']['prc'] = array([
        [20, 50, 60],
        [20, 40, 70],
        [20, 42, 80],
        [20, 44, 90],
        [20, 46, 75],
        [20, 48, 60]
    ])
    bids['P']['qty'] = array([
        [10, 10, 10],
        [10, 10, 10],
        [10, 10, 10]
    ])
    bids['P']['prc'] = array([
        [100, 70, 60],
#         [100, 64.3, 20],
#         [100, 30.64545, 0],
        [100, 50, 20],
        [100, 60, 50]
    ])

    offers['Q']['qty'] = [ 60, 60, 60, 60, 60, 60, 0, 0, 0 ]
    offers['Q']['prc'] = [ 0, 0, 0, 0, 0, 3, 0, 0, 0 ]
    bids.Q['qty'] = [ 15, 15, 15, 15, 15, 15, 15, 12, 7.5 ]
#     bids.Q['prc'] = [ 0, 0, 0, 0, 0, 0, 0, 83.9056, 0 ]
    bids.Q['prc'] = [ 0, 0, 0, 0, 0, 0, 0, 20, 0 ]

    t = 'marginal Q offer, marginal PQ bid, auction_type = 5'
    mkt = {'auction_type': 5,
                      't': [],
                     'u0': [],
                    'lim': []}
    r, co, cb, _, _, _, _ = runmarket(ppc, offers, bids, mkt, ppopt)
    co5 = co.copy()
    cb5 = cb.copy()

#     [ co['P']['qty'] co['P']['prc'] ]
#     [ cb['P']['qty'] cb['P']['prc'] ]
#     [ co['Q']['qty'] co['Q']['prc'] ]
#     [ cb['Q']['qty'] cb['Q']['prc'] ]

    i2e = r['bus'][:, BUS_I]
    e2i = sparse((max(i2e), 1))
    e2i[i2e] = range(r['bus'].size)
    G = find( isload(r['gen']) == 0 )   ## real generators
    L = find( isload(r['gen']) )        ## dispatchable loads
    Gbus = e2i[r['gen'][G, GEN_BUS]]
    Lbus = e2i[r['gen'][L, GEN_BUS]]

    t_is( co['P']['qty'], ones((6, 1)) * [12, 24, 0], 2, [t, ' : gen P quantities'] )
    t_is( co['P']['prc'][0, :], 50.1578, 3, [t, ' : gen 1 P prices'] )
    t_is( cb['P']['qty'], [[10, 10, 10], [10, 0.196, 0], [10, 10, 0]], 2, [t, ' : load P quantities'] )
    t_is( cb['P']['prc'][1, :], 56.9853, 4, [t, ' : load 2 P price'] )
    t_is( co['P']['prc'][:, 0], r['bus'][Gbus, LAM_P], 8, [t, ' : gen P prices'] )
    t_is( cb['P']['prc'][:, 0], r['bus'][Lbus, LAM_P], 8, [t, ' : load P prices'] )

    t_is( co['Q']['qty'], [4.2722, 11.3723, 14.1472, 22.8939, 36.7886, 12.3375, 0, 0, 0], 2, [t, ' : Q offer quantities'] )
    t_is( co['Q']['prc'], [0, 0, 0, 0, 0, 3, 0.4861, 2.5367, 1.3763], 4, [t, ' : Q offer prices'] )
    t_is( cb['Q']['qty'], [0, 0, 0, 0, 0, 0, 15, 4.0785, 5], 2, [t, ' : Q bid quantities'] )
    t_is( cb['Q']['prc'], [0, 0, 0, 0, 0, 3, 0.4861, 2.5367, 1.3763], 4, [t, ' : Q bid prices'] )
    t_is( co['Q']['prc'], r['bus'][[Gbus, Lbus], LAM_Q], 8, [t, ' : Q offer prices'] )
    t_is( cb['Q']['prc'], co['Q']['prc'], 8, [t, ' : Q bid prices'] )

    t = 'marginal Q offer, marginal PQ bid, auction_type = 0'
    mkt['auction_type'] = 0
    r, co, cb, _, _, _, _ = runmarket(ppc, offers, bids, mkt, ppopt)
    t_is( co['P']['qty'], co5['P']['qty'], 8, [t, ' : gen P quantities'] )
    t_is( cb['P']['qty'], cb5['P']['qty'], 8, [t, ' : load P quantities'] )
    t_is( co['P']['prc'], offers['P']['prc'], 8, [t, ' : gen P prices'] )
    t_is( cb['P']['prc'], bids['P']['prc'], 8, [t, ' : load P prices'] )

    t_is( co['Q']['qty'], co5['Q']['qty'], 8, [t, ' : gen Q quantities'] )
    t_is( cb['Q']['qty'], cb5['Q']['qty'], 8, [t, ' : load Q quantities'] )
    t_is( co['Q']['prc'], offers['Q']['prc'], 8, [t, ' : gen Q prices'] )
    t_is( cb['Q']['prc'], bids['Q']['prc'], 8, [t, ' : load Q prices'] )


    t_end
