"""
Reserve model.
"""

from andes.core import (ModelData, IdxParam, NumParam, ExtParam)

from ams.core.model import Model


class ReserveData(ModelData):
    def __init__(self):
        super().__init__()
        self.zone = IdxParam(model='Zone',
                             default=None, info="Zone code",)


class SFR(ReserveData, Model):
    """
    Zonal secondary frequency reserve (SFR) model.

    Notes
    -----
    - ``Zone`` model is required for this model, and zone is defined by Param ``Bus.zone``.
    """

    def __init__(self, system, config):
        ReserveData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Reserve'
        self.du = NumParam(default=0,
                           info='Zonal RegUp reserve demand',
                           tex_name=r'd_{u}', unit='%',)
        self.dd = NumParam(default=0,
                           info='Zonal RegDown reserve demand',
                           tex_name=r'd_{d}', unit='%',)


class SR(ReserveData, Model):
    """
    Zonal spinning reserve (SR) model.

    Notes
    -----
    - ``Zone`` model is required for this model, and zone is defined by Param ``Bus.zone``.
    - ``demand`` is multiplied to online unused generation capacity.
    """

    def __init__(self, system, config):
        ReserveData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Reserve'
        self.demand = NumParam(default=0.1, unit='%',
                               info='Zonal spinning reserve demand',
                               tex_name=r'd_{SR}')


class NSR(ReserveData, Model):
    """
    Zonal non-spinning reserve (NSR) model.

    Notes
    -----
    - ``Zone`` model is required for this model, and zone is defined by Param ``Bus.zone``.
    - ``demand`` is multiplied to offline generation capacity.
    """

    def __init__(self, system, config):
        ReserveData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Reserve'
        self.demand = NumParam(default=0.1, unit='%',
                               info='Zonal non-spinning reserve demand',
                               tex_name=r'd_{NSR}')
