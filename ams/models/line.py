from andes.models.line.line import LineData
from andes.models.line.jumper import JumperData
from andes.core.param import NumParam
from andes.shared import deg2rad

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
    1. Adding Algeb ``ud`` causes Line.algebs to encounter an AttributeError: 'NoneType'
       object has no attribute 'n'. The root cause is still under investigation.
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
