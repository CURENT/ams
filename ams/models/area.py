import logging

from andes.models.area import AreaData
from andes.utils.tab import Tab
from ams.core.model import Model
from ams.core.service import BackRef

logger = logging.getLogger(__name__)


class Area(AreaData, Model):
    """
    Area model.
    """
    def __init__(self, system, config):
        AreaData.__init__(self)
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
            header = ['Area ID', 'Bus ID']
            rows = [(i, j) for i, j in zip(self.idx.v, self.Bus.v)]
            return Tab(header=header, data=rows).draw()
        else:
            return ''
