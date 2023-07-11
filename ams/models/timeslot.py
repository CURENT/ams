"""
Model for rolling horizon used in dispatch.
"""

from andes.core import (ModelData, NumParam, DataParam)  # NOQA
from andes.models.timeseries import (str_list_iconv, str_list_oconv)  # NOQA
from ams.core.model import Model  # NOQA

# TODO: rename to TimeSlot to avoid confusion with horizon in UC
# TODO: input load curve, do the load computation
class TimeSlot(ModelData, Model):
    """
    Rolling horizon.
    """

    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Horizon'

        self.scale = NumParam(info='zonal load scaling factor',
                              tex_name=r's_{load}',
                               iconvert=str_list_iconv,
                               oconvert=str_list_oconv,
                               vtype=float,
                              )
