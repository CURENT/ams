.. _input-matpower:

MATPOWER
--------
The data file format of MATPOWER is excerpted below for quick reference. For more information, see
the `MATPOWER User’s Manual <https://matpower.org/docs/MATPOWER-manual.pdf>`_.

Bus Data
.........
+--------------+--------+-----------------------------------------------------------------+
| name         | column | description                                                     |
+--------------+--------+-----------------------------------------------------------------+
| BUS_I        | 1      | bus number (positive integer)                                   |
+--------------+--------+-----------------------------------------------------------------+
| BUS_TYPE     | 2      | bus type (1 = PQ, 2 = PV, 3 = ref, 4 = isolated)                |
+--------------+--------+-----------------------------------------------------------------+
| PD           | 3      | real power demand (MW)                                          |
+--------------+--------+-----------------------------------------------------------------+
| QD           | 4      | reactive power demand (MVAr)                                    |
+--------------+--------+-----------------------------------------------------------------+
| GS           | 5      | shunt conductance (MW demanded at V = 1.0 p.u.)                 |
+--------------+--------+-----------------------------------------------------------------+
| BS           | 6      | shunt susceptance (MVAr injected at V = 1.0 p.u.)               |
+--------------+--------+-----------------------------------------------------------------+
| BUS AREA     | 7      | area number (positive integer)                                  |
+--------------+--------+-----------------------------------------------------------------+
| VM           | 8      | voltage magnitude (p.u.)                                        |
+--------------+--------+-----------------------------------------------------------------+
| VA           | 9      | voltage angle (degrees)                                         |
+--------------+--------+-----------------------------------------------------------------+
| BASE_KV      | 10     | base voltage (kV)                                               |
+--------------+--------+-----------------------------------------------------------------+
| ZONE         | 11     | loss zone (positive integer)                                    |
+--------------+--------+-----------------------------------------------------------------+
| VMAX         | 12     | maximum voltage magnitude (p.u.)                                |
+--------------+--------+-----------------------------------------------------------------+
| VMIN         | 13     | minimum voltage magnitude (p.u.)                                |
+--------------+--------+-----------------------------------------------------------------+
| LAM_P [1]_   | 14     | Lagrange multiplier on real power mismatch (:math:`u`/MW)       |
+--------------+--------+-----------------------------------------------------------------+
| LAM_Q [1]_   | 15     | Lagrange multiplier on reactive power mismatch (:math:`u`/MVar) |
+--------------+--------+-----------------------------------------------------------------+
| MU_VMAX [1]_ | 16     | Kuhn-Tucker multiplier on upper voltage limit (:math:`u`/p.u.)  |
+--------------+--------+-----------------------------------------------------------------+
| MU_VMIN [1]_ | 17     | Kuhn-Tucker multiplier on lower voltage limit (:math:`u`/p.u.)  |
+--------------+--------+-----------------------------------------------------------------+

.. [1] Included in OPF output, typically not included (or ignored) in input matrix. Here we assume the objective function has units u.

Generator Data
...............
+------------+--------+-----------------------------------------------------------------------+
| name       | column | description                                                           |
+------------+--------+-----------------------------------------------------------------------+
| GEN_BUS    | 1      | bus number                                                            |
+------------+--------+-----------------------------------------------------------------------+
| PG         | 2      | real power output (MW)                                                |
+------------+--------+-----------------------------------------------------------------------+
| QG         | 3      | reactive power output (MVAr)                                          |
+------------+--------+-----------------------------------------------------------------------+
| QMAX       | 4      | maximum reactive power output (MVAr)                                  |
+------------+--------+-----------------------------------------------------------------------+
| QMIN       | 5      | minimum reactive power output (MVAr)                                  |
+------------+--------+-----------------------------------------------------------------------+
| VG         | 6      | voltage magnitude setpoint (p.u.)                                     |
+------------+--------+-----------------------------------------------------------------------+
| MBASE      | 7      | total MVA base of machine, defaults to baseMVA                        |
+------------+--------+-----------------------------------------------------------------------+
| GEN_STATUS | 8      | > 0 : machine in-service machine status, < 0 = machine out-of-service |
+------------+--------+-----------------------------------------------------------------------+
| PMAX       | 9      | maximum real power output (MW)                                        |
+------------+--------+-----------------------------------------------------------------------+
| PMIN       | 10     | minimum real power output (MW)                                        |
+------------+--------+-----------------------------------------------------------------------+
| PC1        | 11     | lower real power output of PQ capability curve (MW)                   |
+------------+--------+-----------------------------------------------------------------------+
| PC2        | 12     | upper real power output of PQ capability curve (MW)                   |
+------------+--------+-----------------------------------------------------------------------+
| QC1MIN     | 13     | minimum reactive power output at PC1 (MVAr)                           |
+------------+--------+-----------------------------------------------------------------------+
| QC1MAX     | 14     | maximum reactive power output at PC1 (MVAr)                           |
+------------+--------+-----------------------------------------------------------------------+
| QC2MIN     | 15     | minimum reactive power output at PC2 (MVAr)                           |
+------------+--------+-----------------------------------------------------------------------+
| QC2MAX     | 16     | maximum reactive power output at PC2 (MVAr)                           |
+------------+--------+-----------------------------------------------------------------------+
| RAMP_AGC   | 17     | ramp rate for load following/AGC (MW/min)                             |
+------------+--------+-----------------------------------------------------------------------+
| RAMP_10    | 18     | ramp rate for 10 minute reserves (MW)                                 |
+------------+--------+-----------------------------------------------------------------------+
| RAMP_30    | 19     | ramp rate for 30 minute reserves (MW)                                 |
+------------+--------+-----------------------------------------------------------------------+
| RAMP_Q     | 20     | ramp rate for reactive power (2 sec timescale) (MVAr/min)             |
+------------+--------+-----------------------------------------------------------------------+
| APF        | 21     | area participation factor                                             |
+------------+--------+-----------------------------------------------------------------------+
| MU_PMAX    | 22     | Kuhn-Tucker multiplier on upper Pg limit (:math:`u`/MW)               |
+------------+--------+-----------------------------------------------------------------------+
| MU_PMIN    | 23     | Kuhn-Tucker multiplier on lower Pg limit (:math:`u`/MW)               |
+------------+--------+-----------------------------------------------------------------------+
| MU_QMAX    | 24     | Kuhn-Tucker multiplier on upper Qg limit (:math:`u`/MVar)             |
+------------+--------+-----------------------------------------------------------------------+
| MU_QMIN    | 25     | Kuhn-Tucker multiplier on lower Q9 limit (:math:`u`/MVar)             |
+------------+--------+-----------------------------------------------------------------------+

``QC1MIN``, ``QC1MAX``, ``QC2MIN``, ``QC2MAX``, ``RAMP_AGC``, ``RAMP_10``, ``RAMP_30``,
``RAMP_Q``, ``APF``: Not included in version 1 case format.

``MU_PMAX``, ``MU_PMIN``, ``MU_QMAX``, ``MU_QMIN``: Included in OPF output, typically not
included (or ignored) in input matrix. Here we assume the objective function has units u.

``VG``: Used to determine voltage setpoint for optimal power flow only if ``opf.use_vg`` option is non-zero (0
by default). Otherwise generator voltage range is determined by limits set for corresponding bus in bus matrix.

Branch Data
............
+------------+--------+----------------------------------------------------------------+
| name       | column | description                                                    |
+------------+--------+----------------------------------------------------------------+
| F_BUS      | 1      | "from" bus number                                              |
+------------+--------+----------------------------------------------------------------+
| T_BUS      | 2      | "to" bus number                                                |
+------------+--------+----------------------------------------------------------------+
| BR_R       | 3      | resistance (p.u.)                                              |
+------------+--------+----------------------------------------------------------------+
| BR_X       | 4      | reactance (p.u.)                                               |
+------------+--------+----------------------------------------------------------------+
| BR_B       | 5      | total line charging susceptance (p.u.)                         |
+------------+--------+----------------------------------------------------------------+
| RATE_A     | 6      | MVA rating A (long term rating), set to 0 for unlimited        |
+------------+--------+----------------------------------------------------------------+
| RATE_B     | 7      | MVA rating B (short term rating), set to 0 for unlimited       |
+------------+--------+----------------------------------------------------------------+
| RATE_C     | 8      | MVA rating C (emergency rating), set to 0 for unlimited        |
+------------+--------+----------------------------------------------------------------+
| TAP        | 9      | transformer off nominal turns ratio                            |
+------------+--------+----------------------------------------------------------------+
| SHIFT      | 10     | transformer phase shift angle (degrees), positive => delay     |
+------------+--------+----------------------------------------------------------------+
| BR_STATUS  | 11     | initial branch status, 1 = in-service, 0 = out-of-service      |
+------------+--------+----------------------------------------------------------------+
| ANGMIN     | 12     | minimum angle difference, Of - Ot (degrees)                    |
+------------+--------+----------------------------------------------------------------+
| ANGMAX     | 13     | maximum angle difference, 0,-0 - (degrees)                     |
+------------+--------+----------------------------------------------------------------+
| PF         | 14     | real power injected at "from" bus end (MW)                     |
+------------+--------+----------------------------------------------------------------+
| QF         | 15     | reactive power injected at "from" bus end (MVAr)               |
+------------+--------+----------------------------------------------------------------+
| PT         | 16     | real power injected at "to" bus end (MW)                       |
+------------+--------+----------------------------------------------------------------+
| QT         | 17     | reactive power injected at "to" bus end (MVAr)                 |
+------------+--------+----------------------------------------------------------------+
| MU_SF      | 18     | Kuhn-Tucker multiplier on MVA limit at "from" bus (u/MVA)      |
+------------+--------+----------------------------------------------------------------+
| MU_ST      | 19     | Kuhn-Tucker multiplier on MVA limit at "to" bus (u/MVA)        |
+------------+--------+----------------------------------------------------------------+
| MU_ANGMINS | 20     | Kuhn-Tucker multiplier lower angle difference limit (u/degree) |
+------------+--------+----------------------------------------------------------------+
| MU_ANGMAX  | 21     | Kuhn-Tucker multiplier upper angle difference limit (u/degree) |
+------------+--------+----------------------------------------------------------------+

``RATE_A``, ``RATE_B``, ``RATE_C``: Used to specify branch flow limits. By default these are limits
on apparent power with units in MVA. However, the 'opf.flow lim' option can be used to specify that the
limits are active power or current, in which case the ratings are specified in MW or :math:`kA·V_{basekV}`,
respectively. For current this is equivalent to an MVA value at a 1 p.u. voltage.

``ANGMIN``, ``ANGMAX``: Not included in version 1 case format. The voltage angle difference is taken
to be unbounded below if ``ANGMIN ≤ −360`` and unbounded above if ``ANGMAX ≥ 360``. If both parameters
are zero, the voltage angle difference is unconstrained.

 ``PF``,
``QF``, ``PT``, ``QT``, ``MU_SF``, ``MU_ST``, ``MU_ANGMINS``, ``MU_ANGMAX``: Not included in version 1 case format.

``MU_SF``, ``MU_ST``, ``MU_ANGMINS``, ``MU_ANGMAX``: Included in OPF output, typically not

Generator Cost Data
....................
+----------+--------+---------------------------------------------------------------------------------------------------------------------------+
| name     | column | description                                                                                                               |
+----------+--------+---------------------------------------------------------------------------------------------------------------------------+
| MODEL    | 1      | cost model, 1 = piecewise linear, 2 = polynomial                                                                          |
+----------+--------+---------------------------------------------------------------------------------------------------------------------------+
| STARTUP  | 2      | startup cost in US dollars                                                                                                |
+----------+--------+---------------------------------------------------------------------------------------------------------------------------+
| SHUTDOWN | 3      | shutdown cost in US dollars                                                                                               |
+----------+--------+---------------------------------------------------------------------------------------------------------------------------+
| NCOST    | 4      | number of points of an n-segment piecewise linear cost function or coefficients of an n-th order polynomial cost function |
+----------+--------+---------------------------------------------------------------------------------------------------------------------------+
| COST     | 5      | parameters defining total cost function f(p)                                                                              |
+----------+--------+---------------------------------------------------------------------------------------------------------------------------+
