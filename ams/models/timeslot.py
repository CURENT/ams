"""
Model for rolling horizon used in dispatch.
"""

from andes.core import ModelData, NumParam
from andes.models.timeseries import str_list_iconv
from ams.core.model import Model


def str_list_oconv(x):
    """
    Convert list into a list literal.
    """
    # NOTE: convert elements to string from number first, then join them
    str_x = [str(i) for i in x]
    return ','.join(str_x)


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
                           vtype=float)


class EDTSlot(TimeSlot):
    """
    Time slot model for ED.

    `sd` is the zonal load scaling factor.
    Cells in `sd` should have `nz` values seperated by comma,
    where `nz` is the number of `Region` in the system.

    `ug` is the unit commitment decisions.
    Cells in `ug` should have `ng` values seperated by comma,
    where `ng` is the number of `StaticGen` in the system.
    """

    def __init__(self, system=None, config=None):
        TimeSlot.__init__(self, system, config)

        self.ug = NumParam(info='unit commitment decisions',
                           tex_name=r'u_{g}',
                           iconvert=str_list_iconv,
                           oconvert=str_list_oconv,
                           vtype=int)


class UCTSlot(TimeSlot):
    """
    Time slot model for UC.

    `sd` is the zonal load scaling factor.
    Cells in `sd` should have `nz` values seperated by comma,
    where `nz` is the number of `Region` in the system.
    """

    def __init__(self, system=None, config=None):
        TimeSlot.__init__(self, system, config)
