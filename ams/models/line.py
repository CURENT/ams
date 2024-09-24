from andes.models.line.line import LineData
from andes.models.line.jumper import JumperData
from andes.core.param import NumParam
from andes.shared import deg2rad, np, spmatrix

from ams.core.model import Model


class Line(LineData, Model):
    """
    AC transmission line model.

    The model is also used for two-winding transformer. Transformers can set the
    tap ratio in ``tap`` and/or phase shift angle ``phi``.

    Note that the bus admittance matrix is built on fly and is not stored in the
    object.

    Notes
    -----
    There is a known issue that adding Algeb ``ud`` will cause Line.algebs run into
    AttributeError: 'NoneType' object has no attribute 'n'. Not figured out why yet.
    """

    def __init__(self, system=None, config=None) -> None:
        LineData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'ACLine'

        self.amin = NumParam(default=-360 * deg2rad,
                             info="minimum angle difference, from bus - to bus",
                             unit='rad',
                             tex_name=r'a_{min}',
                             )
        self.amax = NumParam(default=360 * deg2rad,
                             info="maximum angle difference, from bus - to bus",
                             unit='rad',
                             tex_name=r'a_{max}',
                             )
        self.rate_a.unit = 'p.u.'
        self.rate_b.unit = 'p.u.'
        self.rate_c.unit = 'p.u.'
        self.rate_a.default = 999.0
        self.rate_b.default = 999.0
        self.rate_c.default = 999.0

        # NOTE: following parameters are prepared for building matrices
        # they are initialized here but populated in ``System.setup()``.
        self.a1a = None
        self.a2a = None

    # NOTE: following code are minly copied from `andes.models.line.Line`
    # and they are not fully verified
    # potential issues:
    # `build_Bp` contains 'fdxb', which is not included in the input parameters,
    # and the results are the negative of `Bbus` from `makeBdc` in PYPOWER
    # `build_Bpp` ignores the line resistance for all three methods
    # `build_Bdc` results are the negative of `Bbus` from `makeBdc` in PYPOWER
    # `build_y` results have inignorable differences at diagonal elements with `makeYbus` in PYPOWER

    def build_y(self):
        """
        Build bus admittance matrix. Copied from ``andes.models.line.line.Line``.

        Returns
        -------
        Y : spmatrix
            Bus admittance matrix.
        """

        nb = self.system.Bus.n

        y1 = self.u.v * (self.g1.v + self.b1.v * 1j)
        y2 = self.u.v * (self.g2.v + self.b2.v * 1j)
        y12 = self.u.v / (self.r.v + self.x.v * 1j)
        m = self.tap.v * np.exp(1j * self.phi.v)
        m2 = self.tap.v**2
        mconj = np.conj(m)

        # build self and mutual admittances into Y
        Y = spmatrix((y12 + y1 / m2), self.a1a, self.a1a, (nb, nb), 'z')
        Y -= spmatrix(y12 / mconj, self.a1a, self.a2a, (nb, nb), 'z')
        Y -= spmatrix(y12 / m, self.a2a, self.a1a, (nb, nb), 'z')
        Y += spmatrix(y12 + y2, self.a2a, self.a2a, (nb, nb), 'z')

        return Y

    def build_Bp(self, method='fdpf'):
        """
        Function for building B' matrix.

        Parameters
        ----------
        method : str
            Method for building B' matrix. Choose from 'fdpf', 'fdbx', 'dcpf'.

        Returns
        -------
        Bp : spmatrix
            B' matrix.
        """
        nb = self.system.Bus.n

        if method not in ("fdpf", "fdbx", "dcpf"):
            raise ValueError(f"Invalid method {method}; choose from 'fdpf', 'fdbx', 'dcpf'")

        # Build B prime matrix -- FDPF
        # `y1`` neglects line charging shunt, and g1 is usually 0 in HV lines
        # `y2`` neglects line charging shunt, and g2 is usually 0 in HV lines
        y1 = self.u.v * self.g1.v
        y2 = self.u.v * self.g2.v

        # `m` neglected tap ratio
        m = np.exp(self.phi.v * 1j)
        mconj = np.conj(m)
        m2 = np.ones(self.n)

        if method in ('fdxb', 'dcpf'):
            # neglect line resistance in Bp in XB method
            y12 = self.u.v / (self.x.v * 1j)
        else:
            y12 = self.u.v / (self.r.v + self.x.v * 1j)

        Bdc = spmatrix((y12 + y1) / m2, self.a1a, self.a1a, (nb, nb), 'z')
        Bdc -= spmatrix(y12 / mconj, self.a1a, self.a2a, (nb, nb), 'z')
        Bdc -= spmatrix(y12 / m, self.a2a, self.a1a, (nb, nb), 'z')
        Bdc += spmatrix(y12 + y2, self.a2a, self.a2a, (nb, nb), 'z')
        Bdc = Bdc.imag()

        for item in range(nb):
            if abs(Bdc[item, item]) == 0:
                Bdc[item, item] = 1e-6 + 0j

        return Bdc

    def build_Bpp(self, method='fdpf'):
        """
        Function for building B'' matrix.

        Parameters
        ----------
        method : str
            Method for building B'' matrix. Choose from 'fdpf', 'fdbx', 'dcpf'.

        Returns
        -------
        Bpp : spmatrix
            B'' matrix.
        """

        nb = self.system.Bus.n

        if method not in ("fdpf", "fdbx", "dcpf"):
            raise ValueError(f"Invalid method {method}; choose from 'fdpf', 'fdbx', 'dcpf'")

        # Build B double prime matrix
        # y1 neglected line charging shunt, and g1 is usually 0 in HV lines
        # y2 neglected line charging shunt, and g2 is usually 0 in HV lines
        # m neglected phase shifter
        y1 = self.u.v * (self.g1.v + self.b1.v * 1j)
        y2 = self.u.v * (self.g2.v + self.b2.v * 1j)

        m = self.tap.v
        m2 = abs(m)**2

        if method in ('fdbx', 'fdpf', 'dcpf'):
            # neglect line resistance in Bpp in BX method
            y12 = self.u.v / (self.x.v * 1j)
        else:
            y12 = self.u.v / (self.r.v + self.x.v * 1j)

        Bpp = spmatrix((y12 + y1) / m2, self.a1a, self.a1a, (nb, nb), 'z')
        Bpp -= spmatrix(y12 / np.conj(m), self.a1a, self.a2a, (nb, nb), 'z')
        Bpp -= spmatrix(y12 / m, self.a2a, self.a1a, (nb, nb), 'z')
        Bpp += spmatrix(y12 + y2, self.a2a, self.a2a, (nb, nb), 'z')
        Bpp = Bpp.imag()

        for item in range(nb):
            if abs(Bpp[item, item]) == 0:
                Bpp[item, item] = 1e-6 + 0j

        return Bpp

    def build_Bdc(self):
        """
        The MATPOWER-flavor Bdc matrix for DC power flow.

        The method neglects line charging and line resistance. It retains tap ratio.

        Returns
        -------
        Bdc : spmatrix
            Bdc matrix.
        """

        nb = self.system.Bus.n

        y12 = self.u.v / (self.x.v * 1j)
        y12 = y12 / self.tap.v

        Bdc = spmatrix(y12, self.a1a, self.a1a, (nb, nb), 'z')
        Bdc -= spmatrix(y12, self.a1a, self.a2a, (nb, nb), 'z')
        Bdc -= spmatrix(y12, self.a2a, self.a1a, (nb, nb), 'z')
        Bdc += spmatrix(y12, self.a2a, self.a2a, (nb, nb), 'z')
        Bdc = Bdc.imag()

        for item in range(nb):
            if abs(Bdc[item, item]) == 0:
                Bdc[item, item] = 1e-6

        return Bdc


class Jumper(JumperData, Model):
    """
    Jumper is a device to short two buses (merging two buses into one).

    Jumper can connect two buses satisfying one of the following conditions:

    - neither bus is voltage-controlled
    - either bus is voltage-controlled
    - both buses are voltage-controlled, and the voltages are the same.

    If the buses are controlled in different voltages, power flow will
    not solve (as the power flow through the jumper will be infinite).

    In the solutions, the ``p`` and ``q`` are flowing out of bus1
    and flowing into bus2.

    Setting a Jumper's connectivity status ``u`` to zero will disconnect the two
    buses. In the case of a system split, one will need to call
    ``System.connectivity()`` immediately following the split to detect islands.
    """

    def __init__(self, system=None, config=None) -> None:
        JumperData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'ACShort'
