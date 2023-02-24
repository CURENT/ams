from andes.models.line.line import LineData  # NOQA
from andes.core.param import NumParam  # NOQA
from andes.shared import deg2rad  # NOQA

from ams.core.model import Model  # NOQA


class Line(LineData, Model):
    """
    AC transmission line model.

    The model is also used for two-winding transformer. Transformers can set the
    tap ratio in ``tap`` and/or phase shift angle ``phi``.
    """

    def __init__(self, system=None, config=None) -> None:
        LineData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'ACLine'

        self.amin = NumParam(default=- 360 * deg2rad,
                           info="minimum angle difference, from bus - to bus",
                           unit='rad',
                           tex_name=r'a_{min}',
                           )
        self.amax = NumParam(default=360 * deg2rad,
                            info="maximum angle difference, from bus - to bus",
                            unit='rad',
                            tex_name=r'a_{max}',
                            )
