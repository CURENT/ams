import logging

from andes.core.model import ModelData
from andes.utils.tab import Tab
from andes.core.service import BackRef

from ams.core.model import Model

logger = logging.getLogger(__name__)


class Zone(ModelData, Model):
    """
    Zone model for zonal items.

    An ``area`` can have multiple zones.

    Notes
    -----
    1. Zone is a collection of buses.
    2. Model ``Zone`` is not actually defined in ANDES.

    """

    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)

        self.group = 'Collection'

        self.Bus = BackRef()
        self.ACTopology = BackRef()

    def bus_table(self):
        """
        Return a formatted table with area idx and bus idx correspondence

        Returns
        -------
        str
            Formatted table

        """
        if self.n:
            header = ['Zone ID', 'Bus ID']
            rows = [(i, j) for i, j in zip(self.idx.v, self.Bus.v)]
            return Tab(header=header, data=rows).draw()
        else:
            return ''
