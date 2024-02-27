import logging

from andes.core.model import ModelData
from andes.utils.tab import Tab
from ams.core.model import Model
from ams.core.service import BackRef

logger = logging.getLogger(__name__)


class RegionData(ModelData):
    def __init__(self):
        super().__init__()


class Region(RegionData, Model):
    """
    Region model for zonal vars.

    Notes
    -----
    1. Region is a collection of buses.
    2. Model ``Region`` is not actually defined in ANDES.

    """

    def __init__(self, system, config):
        RegionData.__init__(self)
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
