"""
REGCV1 model.

Voltage-controlled converter model (virtual synchronous generator) with inertia emulation.
"""

from andes.core.param import NumParam, IdxParam  # NOQA
from andes.core.model import ModelData  # NOQA
from ams.core.model import Model  # NOQA


class REGCV1(ModelData, Model):
    """
    REGCV1 model data.
    """

    def __init__(self, system=None, config=None) -> None:
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'RenGen'

        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            mandatory=True,
                            )

        self.M = NumParam(default=10, tex_name='M',
                          info='Emulated startup time constant (M=2H)',
                          unit='s',
                          power=True,
                          )
        self.D = NumParam(default=0, tex_name='D',
                          info='Emulated damping coefficient',
                          unit='p.u.',
                          power=True,
                          )
