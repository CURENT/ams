import logging

import numpy as np

from andes.models.bus import BusData

from ams.core.var import Algeb
from ams.core.model import Model

logger = logging.getLogger(__name__)


class Bus(BusData, Model):
    """
    AC Bus model.
    """

    def __init__(self, system, config):
        BusData.__init__(self)
        Model.__init__(self, system, config)

        self.group = 'ACTopology'
        # NOTE: in ANDES, self.zone is defined to trace a non-existing model "Region"
        # in AMS, model "Zone" is developed,
        # so we need to change the model name of IdxParam self.zone
        self.zone.model = 'Region'

        self.a = Algeb(name='a',
                       tex_name=r'\theta',
                       info='voltage angle',
                       unit='rad',
                       )
        self.v = Algeb(name='v',
                       tex_name='V',
                       info='voltage magnitude',
                       unit='p.u.',
                       )

        # island information
        self.n_islanded_buses = 0
        self.island_sets = list()
        self.islanded_buses = list()  # list of lists containing bus uid of islands
        self.islands = list()         # same as the above
        self.islanded_a = np.array([])
        self.islanded_v = np.array([])
