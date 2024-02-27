from andes.models.line.line import LineData
from andes.core.param import NumParam
from andes.shared import deg2rad

from ams.core.model import Model


class Line(LineData, Model):
    """
    AC transmission line model.

    The model is also used for two-winding transformer. Transformers can set the
    tap ratio in ``tap`` and/or phase shift angle ``phi``.

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
        self.rate_a.default = 999.0
        self.rate_b.default = 999.0
        self.rate_c.default = 999.0
