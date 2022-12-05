import logging
from collections import OrderedDict

import numpy as np

from ams.core.model import ModelData
from ams.core.param import DataParam, IdxParam, NumParam

logger = logging.getLogger(__name__)


class BusData(ModelData):
    """
    Class for Bus data
    """

    def __init__(self):
        super().__init__()
        self.Vn = NumParam(default=110,
                           info="AC voltage rating",
                           unit='kV',
                           non_zero=True,
                           tex_name=r'V_n',
                           )
        self.vmax = NumParam(default=1.1,
                             info="Voltage upper limit",
                             tex_name=r'V_{max}',
                             unit='p.u.',
                             )
        self.vmin = NumParam(default=0.9,
                             info="Voltage lower limit",
                             tex_name=r'V_{min}',
                             unit='p.u.',
                             )

        self.v0 = NumParam(default=1.0,
                           info="initial voltage magnitude",
                           non_zero=True,
                           tex_name=r'V_0',
                           unit='p.u.',
                           )
        self.a0 = NumParam(default=0,
                           info="initial voltage phase angle",
                           unit='rad',
                           tex_name=r'\theta_0',
                           )

        self.xcoord = DataParam(default=0,
                                info='x coordinate (longitude)',
                                )
        self.ycoord = DataParam(default=0,
                                info='y coordinate (latitude)',
                                )

        self.area = IdxParam(model='Area',
                             default=None,
                             info="Area code",
                             )
        self.zone = IdxParam(model='Region',
                             default=None,
                             info="Zone code",
                             )
        self.owner = IdxParam(model='Owner',
                              default=None,
                              info="Owner code",
                              )


class Bus(Model, BusData):
    """
    AC Bus model.

    Power balance equation have the form of ``load - injection = 0``.
    Namely, load is positively summed, while injections are negative.
    """

    def __init__(self, system=None, config=None):
        BusData.__init__(self)
        Model.__init__(self, system=system, config=config)

        self.group = 'ACTopology'
        self.category = ['TransNode']
