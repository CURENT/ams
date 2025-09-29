"""
Models for multi-period scheduling.
"""

from andes.core import ModelData, NumParam
from andes.models.timeseries import str_list_iconv
from ams.core.model import Model


def str_list_oconv(x):
    """
    Convert list into a list literal.

    Revised from `andes.models.timeseries.str_list_oconv`, where
    the output type is converted to a list of strings.
    """
    # NOTE: convert elements to string from number first, then join them
    str_x = [str(i) for i in x]
    return ','.join(str_x)


class GCommit(ModelData, Model):
    """
    UNDER DEVELOPMENT!
    Time slot model for generator commitment decisions.

    This class holds commitment decisions for generators,
    and should be used in multi-period scheduling routines that need
    generator commitment decisions.

    For example, in Unit Commitment (UC) problems, there is a variable
    `ugd` representing the unit commitment decisions for each generator.
    After solving the UC problem, the `ugd` values can be used for
    Economic Dispatch (ED) as a parameter.
    """

    # TODO: .. versionadded:: 1.0.13
    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)

        # TODO: add IdxParam of generator
        self.ug = NumParam(info='unit commitment decisions',
                           tex_name=r'u_{g}',
                           iconvert=str_list_iconv,
                           oconvert=str_list_oconv,
                           vtype=int)


class TimeSlot(ModelData, Model):
    """
    Base model for time slot data used in multi-interval scheduling.
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
    where `nz` is the number of `Zone` in the system.

    `ug` is the unit commitment decisions.
    Cells in `ug` should have `ng` values seperated by comma,
    where `ng` is the number of `StaticGen` in the system.

    Warnings
    --------
    The order of generators in `ug` is determined by the input
    file, not by explicit mapping. This may cause misinterpretation
    if the loaded data order changes.
    Involved routines include: `ED` `UC` and their derivatives.
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
    where `nz` is the number of `Zone` in the system.

    Warnings
    --------
    The order of generators in `ug` is determined by the input
    file, not by explicit mapping. This may cause misinterpretation
    if the loaded data order changes.
    Involved routines include: `ED` `UC` and their derivatives.
    """

    def __init__(self, system=None, config=None):
        TimeSlot.__init__(self, system, config)
