import logging
from collections import OrderedDict

import numpy as np

from andes.models.Bus import BusData  # NOQA

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
    
        # island information
        self.n_islanded_buses = 0
        self.island_sets = list()
        self.islanded_buses = list()  # list of lists containing bus uid of islands
        self.islands = list()         # same as the above
        self.islanded_a = np.array([])
        self.islanded_v = np.array([])
