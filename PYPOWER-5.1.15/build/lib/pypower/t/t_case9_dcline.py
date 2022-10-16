# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 9 bus, 3 generator case.
"""

from numpy import array


def t_case9_dcline():
    """Same as L{t_case9_opfv2} with addition of DC line data.
    Please see L{caseformat} for details on the case file format.

    @return: Power flow data for 9 bus, 3 generator case, with OPF
    and DC line data.
    @see: L{toggle_dcline}, L{idx_dcline}.
    """
    ppc = {}

    ## PYPOWER Case Format : Version 2
    ppc['version'] = '2'

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc['baseMVA'] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc['bus'] = array([
        [1,  3, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [2,  2, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [30, 2, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [4,  1, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [5,  1, 90,  30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [6,  1, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [7,  1, 100, 35, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [8,  1, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [9,  1, 125, 50, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc['gen'] = array([
        [1,  0,   0, 300, -300, 1, 100, 1, 250, 90, 0, 0,    0,  0,   0,  0,  0, 0, 0, 0, 0],
        [2,  163, 0, 300, -300, 1, 100, 1, 300, 10, 0, 200, -20, 20, -10, 10, 0, 0, 0, 0, 0],
        [30, 85,  0, 300, -300, 1, 100, 1, 270, 10, 0, 200, -30, 30, -15, 15, 0, 0, 0, 0, 0]
    ], float)

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc['branch'] = array([
        [1,  4, 0,      0.0576, 0,       0, 250, 250, 0, 0, 1, -360, 2.48],
        [4,  5, 0.017,  0.092,  0.158,   0, 250, 250, 0, 0, 1, -360, 360],
        [5,  6, 0.039,  0.17,   0.358, 150, 150, 150, 0, 0, 1, -360, 360],
        [30, 6, 0,      0.0586, 0,       0, 300, 300, 0, 0, 1, -360, 360],
        [6,  7, 0.0119, 0.1008, 0.209,  40, 150, 150, 0, 0, 1, -360, 360],
        [7,  8, 0.0085, 0.072,  0.149, 250, 250, 250, 0, 0, 1, -360, 360],
        [8,  2, 0,      0.0625, 0,     250, 250, 250, 0, 0, 1, -360, 360],
        [8,  9, 0.032,  0.161,  0.306, 250, 250, 250, 0, 0, 1, -360, 360],
        [9,  4, 0.01,   0.085,  0.176, 250, 250, 250, 0, 0, 1, -2,   360]
    ])

    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc['areas'] = array([
        [1, 5]
    ])

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc['gencost'] = array([
        [1, 0, 0, 4,  0,        0,   100, 2500, 200, 5500, 250, 7250],
        [2, 0, 0, 2, 24.035, -403.5,   0,    0,   0,    0,   0,    0],
        [1, 0, 0, 3,  0,        0,   200, 3000, 300, 5000,   0,    0]
    ])


    ##-----  DC Line Data  -----
    # fbus tbus status Pf Pt Qf Qt Vf Vt Pmin Pmax QminF QmaxF QminT QmaxT loss0 loss1
    ppc['dcline'] = array([
        [30, 4, 1, 10, 8.9,  0, 0, 1.01, 1,    1, 10, -10, 10, -10, 10, 1, 0.01],
        [7,  9, 1,  2, 1.96, 0, 0, 1,    1,    2, 10,   0,  0,   0,  0, 0, 0],
        [5,  8, 0,  0, 0,    0, 0, 1,    1,    1, 10, -10, 10, -10, 10, 0, 0],
        [5,  9, 1, 10, 9.5,  0, 0, 1,    0.98, 0, 10, -10, 10, -10, 10, 0, 0.05]
    ])

    ## DC line cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc['dclinecost'] = array([
        [2, 0, 0, 2, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 2, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 2, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 2, 7.3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    return ppc
