"""
Column indices.
"""

from collections import OrderedDict  # NOQA


gen_src = OrderedDict([
    ('GEN_BUS', 0),     # bus number
    ('PG', 1),          # Pg, real power output (MW)
    ('QG', 2),          # Qg, reactive power output (MVAr)
    ('QMAX', 3),        # Qmax, maximum reactive power output at Pmin (MVAr)
    ('QMIN', 4),        # Qmin, minimum reactive power output at Pmin (MVAr)
    ('VG', 5),          # Vg, voltage magnitude setpoint (p.u.)
    ('MBASE', 6),       # mBase, total MVA base of this machine, defaults to baseMVA
    ('GEN_STATUS', 7),  # status, 1 - machine in service, 0 - machine out of service
    ('PMAX', 8),        # Pmax, maximum real power output (MW)
    ('PMIN', 9),        # Pmin, minimum real power output (MW)
    ('PC1', 10),        # Pc1, lower real power output of PQ capability curve (MW)
    ('PC2', 11),        # Pc2, upper real power output of PQ capability curve (MW)
    ('QC1MIN', 12),     # Qc1min, minimum reactive power output at Pc1 (MVAr)
    ('QC1MAX', 13),     # Qc1max, maximum reactive power output at Pc1 (MVAr)
    ('QC2MIN', 14),     # Qc2min, minimum reactive power output at Pc2 (MVAr)
    ('QC2MAX', 15),     # Qc2max, maximum reactive power output at Pc2 (MVAr)
    ('RAMP_AGC', 16),   # ramp rate for load following/AGC (MW/min)
    ('RAMP_10', 17),    # ramp rate for 10 minute reserves (MW)
    ('RAMP_30', 18),    # ramp rate for 30 minute reserves (MW)
    ('RAMP_Q', 19),     # ramp rate for reactive power (2 sec timescale) (MVAr/min)
    ('APF', 20),        # area participation factor
    # -----  OPF Data  -----
    # NOTE: included in opf solution, not necessarily in input
    # NOTE: assume objective function has units, u
    ('MU_PMAX', 21),       # Kuhn-Tucker multiplier on upper Pg limit (u/MW)
    ('MU_PMIN', 22),       # Kuhn-Tucker multiplier on lower Pg limit (u/MW)
    ('MU_QMAX', 23),       # Kuhn-Tucker multiplier on upper Qg limit (u/MVAr)
    ('MU_QMIN', 24),       # Kuhn-Tucker multiplier on lower Qg limit (u/MVAr)
])

dcline_src = OrderedDict([
    ('F_BUS', 0),     # f, "from" bus number
    ('T_BUS', 1),     # t,  "to"  bus number
    ('BR_STATUS', 2),  # initial branch status, 1 - in service, 0 - out of service
    ('PF', 3),        # MW flow at "from" bus ("from" -> "to")
    ('PT', 4),        # MW flow at  "to"  bus ("from" -> "to")
    ('QF', 5),        # MVAr injection at "from" bus ("from" -> "to")
    ('QT', 6),        # MVAr injection at  "to"  bus ("from" -> "to")
    ('VF', 7),        # voltage setpoint at "from" bus (p.u.)
    ('VT', 8),        # voltage setpoint at  "to"  bus (p.u.)
    ('PMIN', 9),      # lower limit on PF (MW flow at "from" end)
    ('PMAX', 10),     # upper limit on PF (MW flow at "from" end)
    ('QMINF', 11),    # lower limit on MVAr injection at "from" bus
    ('QMAXF', 12),    # upper limit on MVAr injection at "from" bus
    ('QMINT', 13),    # lower limit on MVAr injection at  "to"  bus
    ('QMAXT', 14),    # upper limit on MVAr injection at  "to"  bus
    ('LOSS0', 15),    # constant term of linear loss function (MW)
    ('LOSS1', 16),    # linear term of linear loss function (MW)
    ('MU_PMIN', 17),  # Kuhn-Tucker multiplier on lower flow limit at "from" bus (u/MW)
    ('MU_PMAX', 18),  # Kuhn-Tucker multiplier on upper flow limit at "from" bus (u/MW)
    ('MU_QMINF', 19),  # Kuhn-Tucker multiplier on lower Q limit at "from" bus (u/MVAr)
    ('MU_QMAXF', 20),  # Kuhn-Tucker multiplier on upper Q limit at "from" bus (u/MVAr)
    ('MU_QMINT', 21),  # Kuhn-Tucker multiplier on lower Q limit at  "to"  bus (u/MVAr)
    ('MU_QMAXT', 22),  # Kuhn-Tucker multiplier on upper Q limit at  "to"  bus (u/MVAr)
])

cost_src = OrderedDict([
    ('PW_LINEAR', 1),   # cost model, 1 - piecewise linear, 2 - polynomial
    ('POLYNOMIAL', 2),  # cost model, 1 - piecewise linear, 2 - polynomial
    ('MODEL', 0),       # cost model, 1 - piecewise linear, 2 - polynomial
    ('STARTUP', 1),     # startup cost in US dollars
    ('SHUTDOWN', 2),    # shutdown cost in US dollars
    ('NCOST', 3),       # number of cost coefficients to follow
    ('COST', 4),        # cost coefficients, see below
])

bus_src = OrderedDict([
    # --- bus types ---
    ('PQ', 1),          # PQ bus
    ('PV', 2),          # PV bus
    ('REF', 3),         # reference bus
    ('NONE', 4),    # isolated bus
    # --- indices ---
    ('BUS_I', 0),       # bus number (1 to 29997)
    ('BUS_TYPE', 1),    # bus type
    ('PD', 2),          # Pd, real power demand (MW)
    ('QD', 3),          # Qd, reactive power demand (MVAr)
    ('GS', 4),          # Gs, shunt conductance (MW at V = 1.0 p.u.)
    ('BS', 5),          # Bs, shunt susceptance (MVAr at V = 1.0 p.u.)
    ('BUS_AREA', 6),    # area number, 1-100
    ('VM', 7),          # Vm, voltage magnitude (p.u.)
    ('VA', 8),          # Va, voltage angle (degrees)
    ('BASE_KV', 9),     # baseKV, base voltage (kV)
    ('ZONE', 10),       # zone, loss zone (1-999)
    ('VMAX', 11),       # maxVm, maximum voltage magnitude (p.u.)
    ('VMIN', 12),       # minVm, minimum voltage magnitude (p.u.)
    # NOTE: included in opf solution, not necessarily in input
    # NOTE: assume objective function has units, u
    ('LAM_P', 13),      # Lagrange multiplier on real power mismatch (u/MW)
    ('LAM_Q', 14),      # Lagrange multiplier on reactive power mismatch (u/MVAr)
    ('MU_VMAX', 15),    # Kuhn-Tucker multiplier on upper voltage limit (u/p.u.)
    ('MU_VMIN', 16),    # Kuhn-Tucker multiplier on lower voltage limit (u/p.u.)
])

branch_src = OrderedDict([
    ('F_BUS', 0),       # f, from bus number
    ('T_BUS', 1),       # t, to bus number
    ('BR_R', 2),        # r, resistance (p.u.)
    ('BR_X', 3),        # x, reactance (p.u.)
    ('BR_B', 4),        # b, total line charging susceptance (p.u.)
    ('RATE_A', 5),      # rateA, MVA rating A (long term rating)
    ('RATE_B', 6),      # rateB, MVA rating B (short term rating)
    ('RATE_C', 7),      # rateC, MVA rating C (emergency rating)
    ('TAP', 8),         # ratio, transformer off nominal turns ratio
    ('SHIFT', 9),       # angle, transformer phase shift angle (degrees)
    ('BR_STATUS', 10),  # initial branch status, 1 - in service, 0 - out of service
    ('ANGMIN', 11),     # minimum angle difference, angle(Vf) - angle(Vt) (degrees)
    ('ANGMAX', 12),     # maximum angle difference, angle(Vf) - angle(Vt) (degrees)
    # NOTE: included in opf solution, not necessarily in input
    ('PF', 13),         # real power injected at "from" bus (MW)
    ('QF', 14),         # reactive power injected at "from" bus (MVAr)
    ('PT', 15),         # real power injected at "to" bus (MW)
    ('QT', 16),         # reactive power injected at "to" bus (MVAr)
    # NOTE: included in opf solution, not necessarily in input
    # NOTE: assume objective function has units, u
    ('MU_SF', 17),      # Kuhn-Tucker multiplier on MVA limit at "from" bus (u/MVA)
    ('MU_ST', 18),      # Kuhn-Tucker multiplier on MVA limit at "to" bus (u/MVA)
    ('MU_ANGMIN', 19),  # Kuhn-Tucker multiplier on lower angle difference limit (u/degree)
    ('MU_ANGMAX', 20),  # Kuhn-Tucker multiplier on upper angle difference limit (u/degree)
])

area_src = OrderedDict([
    ('AREA_I', 0),    # area number
    ('PRICE_REF_BUS', 1),    # price reference bus for this area
])


class Consts:
    """
    Base class for constants collection.
    """

    def __init__(self, source):
        """
        Parameters
        ----------
        source : OrderedDict
            Dictionary of constants.
        """
        self.init(source)

    def init(self, source):
        for key, value in source.items():
            setattr(self, key, value)


class IDXClass:
    """
    Column indices for named columns.
    """

    def __init__(self):
        """
        Parameters
        ----------
        source : OrderedDict
            Dictionary of column indices.
        """
        self.gen = Consts(gen_src)
        self.dcline = Consts(dcline_src)
        self.cost = Consts(cost_src)
        self.bus = Consts(bus_src)
        self.branch = Consts(branch_src)
        self.area = Consts(area_src)

IDX = IDXClass()
