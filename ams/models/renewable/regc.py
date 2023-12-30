"""
RenGen dispatch model.
"""

from andes.core.param import NumParam, IdxParam
from andes.core.model import ModelData
from ams.core.model import Model


class REGCData(ModelData):
    """
    Data container for RenGen dispatch model.
    """

    def __init__(self):
        ModelData.__init__(self)
        self.bus = IdxParam(model='Bus',
                            info="interface bus id (place holder)",
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


class REGCA1(REGCData, Model):
    """
    Renewable generator dispatch model.

    Reference:

    [1] ANDES Documentation, REGCA1

    Available:
    https://docs.andes.app/en/latest/groupdoc/RenGen.html#regca1
    """

    def __init__(self, system=None, config=None) -> None:
        REGCData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'RenGen'


class REGCV1(REGCData, Model):
    """
    Voltage-controlled converter model (virtual synchronous generator) with inertia emulation.

    Notes
    -----
    - The generation is defined by group :ref:`StaticGen`
    - Generation cost is defined by model :ref:`GCost`
    - Inertia emulation cost is defined by model :ref:`REGCV1Cost`

    Reference:

    [1] ANDES Documentation, REGCV1

    Available:

    https://docs.andes.app/en/latest/groupdoc/RenGen.html#regcv1
    """

    def __init__(self, system=None, config=None) -> None:
        REGCData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'VSG'


class REGCV2(REGCData, Model):
    """
    Voltage-controlled VSC.

    Reference:

    [1] ANDES Documentation, REGCV2

    Available:

    https://docs.andes.app/en/latest/groupdoc/RenGen.html#regcv2
    """

    def __init__(self, system=None, config=None) -> None:
        REGCData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'VSG'
