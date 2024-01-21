"""
Distributed energy storage model.
"""

from andes.core.param import NumParam

from ams.core.model import Model

from ams.models.distributed.pvd1 import PVD1Data


class ESD1Data(PVD1Data):
    """
    ESD1 model data.
    """

    def __init__(self):
        PVD1Data.__init__(self)

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


class ESD1(ESD1Data, Model):
    """
    Distributed energy storage model, revised from ANDES ``ESD1`` model for
    dispatch.

    Following parameters are omitted from the original dynamic model:
    ``fn``, ``busf``, ``xc``, ``pqflag``, ``igreg``, ``v0``, ``v1``,
    ``dqdv``, ``fdbd``, ``ddn``, ``ialim``, ``vt0``, ``vt1``, ``vt2``,
    ``vt3``, ``vrflag``, ``ft0``, ``ft1``, ``ft2``, ``ft3``, ``frflag``,
    ``tip``, ``tiq``, ``recflag``.

    Reference:

    [1] ANDES Documentation, ESD1

    Available:

    https://docs.andes.app/en/latest/groupdoc/DG.html#esd1
    """

    def __init__(self, system, config):
        ESD1Data.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'DG'
