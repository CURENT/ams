"""
Distributed energy storage model.
"""

from andes.core.param import NumParam
from andes.models.distributed.esd1 import ESD1Data  # NOQA

from ams.core.model import Model  # NOQA
from ams.core.var import Algeb  # NOQA


class ESD1Data(ESD1Data):
    """
    ESD1 model data, revised from ANDES ``ESD1Data``.
    """

    def __init__(self):
        super().__init__()
        self.pqflag = NumParam(info='P/Q priority for I limit; 0-Q priority, 1-P priority',
                               mandatory=False,
                               unit='bool',
                               )
        attr_to_remove = ['fn', 'busf', 'xc', 'pqflag', 'igreg',
                          'v0', 'v1', 'dqdv', 'fdbd', 'ddn',
                          'ialim', 'vt0', 'vt1', 'vt2', 'vt3',
                          'vrflag', 'ft0', 'ft1', 'ft2', 'ft3',
                          'frflag', 'tip', 'tiq',
                          'recflag', 'Tf']
        for attr in attr_to_remove:
            delattr(self, attr)


class ESD1Model(Model):
    """
    ESD1 model for dispatch.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)


class ESD1(ESD1Data, ESD1Model):
    """
    Distributed energy storage model.
    Revised from ANDES ``ESD1`` model for dispatch purpose.

    Notes
    -----
    1. some parameters are removed from the original dynamic model, including:
    ``fn``, ``busf``, ``xc``, ``pqflag``, ``igreg``, ``v0``, ``v1``, ``dqdv``,
    ``fdbd``, ``ddn``, ``ialim``, ``vt0``, ``vt1``, ``vt2``, ``vt3``,
    ``vrflag``, ``ft0``, ``ft1``, ``ft2``, ``ft3``, ``frflag``, ``tip``,
    ``tiq``, ``recflag``.

    Reference:

    [1] Powerworld, Renewable Energy Electrical Control Model REEC_C

    [2] ANDES Documentation, ESD1

    Available:

    https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Exciter%20REEC_C.htm
    https://docs.andes.app/en/latest/groupdoc/DG.html#esd1
    """

    def __init__(self, system, config):
        ESD1Data.__init__(self)
        ESD1Model.__init__(self, system, config)
        self.group = 'DG'
