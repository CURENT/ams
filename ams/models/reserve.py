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
        self.du = NumParam(default=0,
                           info='Zonal RegUp reserve demand (system base)',
                           tex_name=r'd_{u}', unit=r'p.u.',)
        self.dd = NumParam(default=0,
                           info='Zonal RegDown reserve demand (system base)',
                           tex_name=r'd_{d}', unit=r'p.u.',)


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


class SR(ReserveData, Model):
    """
    Zonal spinning reserve (SR) model.

    Notes
    -----
    - ``Zone`` model is required for this model, and zone is defined by Param ``Bus.zone``.
    """

    def __init__(self, system, config):
        ReserveData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Reserve'


class NSR(ReserveData, Model):
    """
    Zonal non-spinning reserve (NSR) model.

    Notes
    -----
    - ``Zone`` model is required for this model, and zone is defined by Param ``Bus.zone``.
    """

    def __init__(self, system, config):
        ReserveData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Reserve'
