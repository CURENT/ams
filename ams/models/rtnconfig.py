"""
Model for routine configuration.
"""

from andes.core import (ModelData, NumParam, DataParam)  # NOQA
from andes.models.timeseries import (str_list_iconv, str_list_oconv)  # NOQA
from ams.core.model import Model  # NOQA


class BaseConfig(ModelData, Model):
    """
    Base configuration model.
    """

    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Horizon'


class RTEDCFG(BaseConfig):
    """
    RTED configuration.
    """

    def __init__(self, system=None, config=None):
        BaseConfig.__init__(self, system, config)
        self.dt = NumParam(info='RTED time duration',
                           tex_name=r'Delta t', unit='min',
                           default=5, vtype=float,)
