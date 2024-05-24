"""
EV model.
"""

from andes.core.param import NumParam, IdxParam
from andes.core.model import ModelData

from ams.core.model import Model


class EV1(ModelData, Model):
    """
    Aggregated EV model for scheduling at transmission level.

    For co-simulation with ADNES, it is expected to be used in
    conjunction with the dynamic models `EV1` or `EV2`.

    Reference:

    [1] J. Wang et al., "Electric Vehicles Charging Time Constrained Deliverable Provision of Secondary
    Frequency Regulation," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2024.3356948.
    """

    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'DG'

        self.bus = IdxParam(model='Bus',
                            info="interface bus idx",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            mandatory=True,
                            )
        self.Sn = NumParam(default=100.0, tex_name='S_n',
                           info='device MVA rating',
                           unit='MVA',
                           )
        self.gammap = NumParam(default=1.0,
                               info="P ratio of linked static gen",
                               tex_name=r'\gamma_P'
                               )
        self.gammaq = NumParam(default=1.0,
                               info="Q ratio of linked static gen",
                               tex_name=r'\gamma_Q'
                               )
        self.N = NumParam(default=10000,
                          info="number of related EVs",
                          tex_name='N'
                          )


class EV2(EV1):
    """
    EV aggregation model at transmission level, identical to :ref:`EV1`.
    """

    def __init__(self, system=None, config=None) -> None:
        EV1.__init__(self, system, config)
