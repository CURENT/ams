"""
Distributed PV models.
"""

from andes.core.param import IdxParam, NumParam
from andes.core.model import ModelData

from ams.core.model import Model


class PVD1Data(ModelData):
    """
    PVD1 model data.
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

        self.gammap = NumParam(default=1.0, tex_name=r'\gamma_p',
                               info='Ratio of ESD1.pref0 w.r.t to that of static PV',
                               vrange='(0, 1]',
                               )

        self.gammaq = NumParam(default=1.0, tex_name=r'\gamma_q',
                               info='Ratio of ESD1.qref0 w.r.t to that of static PV',
                               vrange='(0, 1]',
                               )


class PVD1(PVD1Data, Model):
    """
    Distributed PV model, revised from ANDES ``PVD1`` model for
    dispatch.

    Following parameters are omitted from the original dynamic model:
    ``fn``, ``busf``, ``xc``, ``pqflag``, ``igreg``, ``v0``, ``v1``,
    ``dqdv``, ``fdbd``, ``ddn``, ``ialim``, ``vt0``, ``vt1``, ``vt2``,
    ``vt3``, ``vrflag``, ``ft0``, ``ft1``, ``ft2``, ``ft3``, ``frflag``,
    ``tip``, ``tiq``, ``recflag``.

    Reference:

    [1] ANDES Documentation, PVD1

    Available:

    https://docs.andes.app/en/latest/groupdoc/DG.html#pvd1
    """

    def __init__(self, system, config):
        PVD1Data.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'DG'
