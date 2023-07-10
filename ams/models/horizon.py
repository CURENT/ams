"""
Model for rolling horizon used in dispatch.
"""

from andes.core import (ModelData, NumParam)  # NOQA

from ams.core.model import Model  # NOQA

# TODO: rename to TimeSlot to avoid confusion with horizon in UC
# TODO: input load curve, do the load computation
class Horizon(ModelData, Model):
    """
    Rolling horizon.
    """

    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'TimeHorizon'

        self.scale = NumParam(default=1.0,
                              info='scaling factor for load',
                              tex_name=r's_{load}',
                              )
