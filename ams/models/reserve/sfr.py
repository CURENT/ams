"""
SFR model.
"""

from andes.core import (ModelData, IdxParam, NumParam, ExtParam)

from ams.core.model import Model


class SFRData(ModelData):
    def __init__(self):
        super().__init__()
        self.zone = IdxParam(model='Zone',
                             default=None,
                             info="Zone code",
                             )
        self.du = NumParam(default=0,
                           info='Zonal RegUp reserve demand (system base)',
                           tex_name=r'd_{u}',
                           unit=r'p.u.',
                           )
        self.dd = NumParam(default=0,
                           info='Zonal RegDown reserve demand (system base)',
                           tex_name=r'd_{d}',
                           unit=r'p.u.',
                           )


class SFR(SFRData, Model):
    """
    Zonal secondary frequency reserve (SFR) model.

    ``Zone`` model is required for this model, and zone is
    defined by Param ``Bus.zone``.
    """

    def __init__(self, system, config):
        SFRData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Reserve'
