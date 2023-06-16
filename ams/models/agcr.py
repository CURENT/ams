"""
Cost model.
"""

from andes.core import (ModelData, IdxParam, NumParam)

from ams.core.model import Model


class AGCRData(ModelData):
    def __init__(self):
        super().__init__()
        self.zone = IdxParam(model='Zone',
                             default=None,
                             info="Zone code",
                             )
        self.du = NumParam(default=0,
                           info='RegUp reserve demand',
                           tex_name=r'd_{u}',
                           unit=r'p.u.',
                           )
        self.dd = NumParam(default=0,
                           info='RegDown reserve demand',
                           tex_name=r'd_{d}',
                           unit=r'p.u.',
                           )


class AGCR(AGCRData, Model):
    """
    Zonal AGC reserve model.
    """

    def __init__(self, system, config):
        AGCRData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Reserve'
