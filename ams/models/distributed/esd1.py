"""
Distributed energy storage model.
"""

from andes.core.param import IdxParam, NumParam  # NOQA
from andes.core.model import ModelData  # NOQA

from ams.core.model import Model  # NOQA
from ams.core.var import Algeb  # NOQA


class ESD1Data(ModelData):
    """
    ESD1 model data.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
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
                               info='Ratio of PVD1.pref0 w.r.t to that of static PV',
                               vrange='(0, 1]',
                               )

        self.gammaq = NumParam(default=1.0, tex_name=r'\gamma_q',
                               info='Ratio of PVD1.qref0 w.r.t to that of static PV',
                               vrange='(0, 1]',
                               )

        self.SOCmin = NumParam(default=0.0, tex_name='SOC_{min}',
                               info='Minimum required value for SOC in limiter',
                               )

        self.SOCmax = NumParam(default=1.0, tex_name='SOC_{max}',
                               info='Maximum allowed value for SOC in limiter',
                               )

        self.SOCinit = NumParam(default=0.5, tex_name='SOC_{init}',
                                info='Initial state of charge',
                                )

        self.En = NumParam(default=100.0, tex_name='E_n',
                           info='Rated energy capacity',
                           unit="MWh"
                           )

        self.EtaC = NumParam(default=1.0, tex_name='Eta_C',
                             info='Efficiency during charging',
                             vrange=(0, 1),
                             )

        self.EtaD = NumParam(default=1.0, tex_name='Eta_D',
                             info='Efficiency during discharging',
                             vrange=(0, 1),
                             )


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
