"""
REGCV1 model.

Voltage-controlled converter model (virtual synchronous generator) with inertia emulation.
"""

from andes.core.param import NumParam, IdxParam  # NOQA
from andes.core.model import ModelData  # NOQA
from ams.core.model import Model  # NOQA


class REGCV1(ModelData, Model):
    """
    Voltage-controlled converter model (virtual synchronous generator) with inertia emulation.

    Notes
    -----
    - The generation is defined by group :ref:`StaticGen`
    - Generation cost is defined by model :ref:`GCost`
    - Inertia emulation cost is defined by model :ref:`REGCV1Cost`
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
