.. _input-pypower:

PYPOWER
--------

AMS includes `PYPOWER cases <https://github.com/jinningwang/ams/tree/develop/ams/cases/pypower>`_
in version 2 for dispatch modeling and analysis. PYPOWER cases follow the same format as MATPOWER.

The PYPOWER case is defined as a Python dictionary that includes ``bus``, ``gen``, ``branch``,
``areas``, and ``gencost``.
Defines the PYPOWER case file format.

A PYPOWER case file is a Python file or MAT-file that defines or returns a dict named ``ppc``, referred to
as a "PYPOWER case dict". The keys of this dict are ``bus``, ``gen``, ``branch``, ``areas``, and
``gencost``.
With the exception of C{baseMVA}, a scalar, each data variable is an array, where a row corresponds
to a single bus, branch, gen, etc. The format of the data is similar to the PTI format described in
`PTI Load Flow Data Format <http://www.ee.washington.edu/research/pstca/formats/pti.txt>`_.


Example Case9
~~~~~~~~~~~~~~~~~~

.. code:: python

    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 3, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [2, 2, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [3, 2, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [4, 1, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [5, 1, 90,  30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [6, 1, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [7, 1, 100, 35, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [8, 1, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [9, 1, 125, 50, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [1, 0,   0, 300, -300, 1, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 163, 0, 300, -300, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 85,  0, 300, -300, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [1, 4, 0,      0.0576, 0,     250, 250, 250, 0, 0, 1, -360, 360],
        [4, 5, 0.017,  0.092,  0.158, 250, 250, 250, 0, 0, 1, -360, 360],
        [5, 6, 0.039,  0.17,   0.358, 150, 150, 150, 0, 0, 1, -360, 360],
        [3, 6, 0,      0.0586, 0,     300, 300, 300, 0, 0, 1, -360, 360],
        [6, 7, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
        [7, 8, 0.0085, 0.072,  0.149, 250, 250, 250, 0, 0, 1, -360, 360],
        [8, 2, 0,      0.0625, 0,     250, 250, 250, 0, 0, 1, -360, 360],
        [8, 9, 0.032,  0.161,  0.306, 250, 250, 250, 0, 0, 1, -360, 360],
        [9, 4, 0.01,   0.085,  0.176, 250, 250, 250, 0, 0, 1, -360, 360]
    ])

    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = array([
        [1, 5]
    ])

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 1500, 0, 3, 0.11,   5,   150],
        [2, 2000, 0, 3, 0.085,  1.2, 600],
        [2, 3000, 0, 3, 0.1225, 1,   335]
    ])


Version Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two versions of the PYPOWER case file format. The current
version of PYPOWER uses version 2 of the PYPOWER case format
internally and includes a ``version`` field with a value of ``2`` to make
the version explicit. Earlier versions of PYPOWER used the version 1
case format, which defined the data matrices as individual variables,
as opposed to keys of a dict. Case files in version 1 format with
OPF data also included an (unused) ``areas`` variable. While the version 1
format has now been deprecated, it is still handled automatically by
``loadcase`` and ``savecase`` which are able to load and save case files in both
version 1 and version 2 formats.

See also doc for `idx_bus`, `idx_brch`, `idx_gen`, `idx_area` and `idx_cost`
regarding constants which can be used as named column indices for the data
matrices. Also described in the first three are additional results columns
that are added to the bus, branch, and gen matrices by the power flow and OPF
solvers.

The case dict also allows for additional fields to be included.
The OPF is designed to recognize fields named ``A``, ``l``, ``u``, ``H``, ``Cw``,
``N``, ``fparm``, ``z0``, ``zl``, and ``zu`` as parameters used to directly extend
the OPF formulation (see doc for `opf` for details). Other user-defined fields may
also be included and will be automatically loaded by the ``loadcase`` function
and, given an appropriate 'savecase' callback function (see doc for
`add_userfcn`), saved by the ``savecase`` function.


Bus
~~~~~~~~~

#.  bus number (positive integer)
#.  bus type
    - PQ bus = 1
    - PV bus = 2
    - reference bus = 3
    - isolated bus = 4
#.  ``Pd``, real power demand (MW)
#.  ``Qd``, reactive power demand (MVAr)
#.  ``Gs``, shunt conductance (MW demanded at V = 1.0 p.u.)
#.  ``Bs``, shunt susceptance (MVAr injected at V = 1.0 p.u.)
#.  area number (positive integer)
#.  ``Vm``, voltage magnitude (p.u.)
#.  ``Va``, voltage angle (degrees)
#.  ``baseKV``, base voltage (kV)
#.  ``zone``, loss zone (positive integer)
#.  ``maxVm``, maximum voltage magnitude (p.u.)
#.  ``minVm``, minimum voltage magnitude (p.u.)

Generator
~~~~~~~~~~~~~~~~~~

#.  bus number
#.  ``Pg``, real power output (MW)
#.  ``Qg``, reactive power output (MVAr)
#.  ``Qmax``, maximum reactive power output (MVAr)
#.  ``Qmin``, minimum reactive power output (MVAr)
#.  ``Vg``, voltage magnitude setpoint (p.u.)
#.  ``mBase``, total MVA base of this machine, defaults to baseMVA
#.  status
    - ``>  0`` - machine in service
    - ``<= 0`` - machine out of service
#.  ``Pmax``, maximum real power output (MW)
#.  ``Pmin``, minimum real power output (MW)
#.  ``Pc1``, lower real power output of PQ capability curve (MW)
#.  ``Pc2``, upper real power output of PQ capability curve (MW)
#.  ``Qc1min``, minimum reactive power output at Pc1 (MVAr)
#.  ``Qc1max``, maximum reactive power output at Pc1 (MVAr)
#.  ``Qc2min``, minimum reactive power output at Pc2 (MVAr)
#.  ``Qc2max``, maximum reactive power output at Pc2 (MVAr)
#.  ramp rate for load following/AGC (MW/min)
#.  ramp rate for 10-minute reserves (MW)
#.  ramp rate for 30-minute reserves (MW)
#.  ramp rate for reactive power (2-sec timescale) (MVAr/min)
#.  APF, area participation factor


Branch
~~~~~~~~~

#.  ``f``, from bus number
#.  ``t``, to bus number
#.  ``r``, resistance (p.u.)
#.  ``x``, reactance (p.u.)
#.  ``b``, total line charging susceptance (p.u.)
#.  ``rateA``, MVA rating A (long-term rating)
#.  ``rateB``, MVA rating B (short-term rating)
#.  ``rateC``, MVA rating C (emergency rating)
#.  ``ratio``, transformer off nominal turns ratio (``= 0`` for lines)
#.  ``angle``, transformer phase shift angle (degrees), positive -> delay

    -  (Gf, shunt conductance at from bus p.u.)
    -  (Bf, shunt susceptance at from bus p.u.)
    -  (Gt, shunt conductance at to bus p.u.)
    -  (Bt, shunt susceptance at to bus p.u.)
#.  initial branch status, 1 - in service, 0 - out of service
#.  minimum angle difference, angle(Vf) - angle(Vt) (degrees)
#.  maximum angle difference, angle(Vf) - angle(Vt) (degrees)


Generator Cost
~~~~~~~~~~~~~~~~~~

.. note::

   If ``gen`` has ``ng`` rows, then the first ``ng`` rows of ``gencost`` contain
   the cost for active power produced by the corresponding generators.
   If ``gencost`` has :math:`2 \times ng` rows then rows :math:`ng + 1` to :math:`2 \times ng`
   contain the reactive power costs in the same format.

#.  ``model``, 1 - piecewise linear, 2 - polynomial
#.  ``startup``, startup cost in US dollars
#.  ``shutdown``, shutdown cost in US dollars
#.  ``N``, number of cost coefficients to follow for polynomial cost function,
    or number of data points for piecewise linear.
    The following parameters define the total cost function ``f(p)``,
    where units of ``f`` and ``p`` are $/hr and MW (or MVAr), respectively.

    -  For MODEL = 1: ``p0, f0, p1, f1, ..., pn, fn``
       where ``p0 < p1 < ... < pn`` and the cost ``f(p)`` is defined by
       the coordinates ``(p0,f0), (p1,f1), ..., (pn,fn)`` of the
       end/break-points of the piecewise linear cost function.
    -  For MODEL = 2: ``cn, ..., c1, c0``
       ``n + 1`` coefficients of an ``n``-th order polynomial cost function,
       starting with the highest order, where cost is
       :math:`f(p) = c_n \times p^n + \ldots + c_1 \times p + c_0`.


Area (deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   This data is not used by PYPOWER and is no longer necessary for version 2 case files with OPF data.

#.  ``i``, area number
#.  ``price_ref_bus``, reference bus for that area
