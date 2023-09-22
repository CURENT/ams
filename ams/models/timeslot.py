"""
Model for rolling horizon used in dispatch.
"""

from andes.core import (ModelData, NumParam, DataParam)  # NOQA
from andes.models.timeseries import (str_list_iconv, str_list_oconv)  # NOQA
from ams.core.model import Model  # NOQA


class TimeSlot(ModelData, Model):
    """
    Time slot data for rolling horizon.
    """

    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Horizon'

        self.sd = NumParam(info='zonal load scaling factor',
                           tex_name=r's_{d}',
                           iconvert=str_list_iconv,
                           oconvert=str_list_oconv,
                           vtype=float,
                           )


class EDTSlot(TimeSlot):
    """
    Time slot model for ED.
    """

    def __init__(self, system=None, config=None):
        TimeSlot.__init__(self, system, config)


class UCTSlot(TimeSlot):
    """
    Time slot model for UC.
    """

    def __init__(self, system=None, config=None):
        TimeSlot.__init__(self, system, config)
