"""
Power flow routines.
"""
import logging
from collections import OrderedDict

import numpy as np

from andes.shared import deg2rad

from ams.solver.pypower.runpf import runpf, rundcpf

from ams.io.pypower import system2ppc
from ams.core.param import RParam
from ams.core.var import RAlgeb

from ams.routines.dcpf import DCPFlowData, DCPFlowBase

logger = logging.getLogger(__name__)


class PFlowData(DCPFlowData):
    """
    AC Power Flow routine.
    """

    def __init__(self):
        DCPFlowData.__init__(self)

    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        res, success = runpf(ppc, **kwargs)
        return ppc, success

class PFlowBase(DCPFlowBase):
    """
    Base class for AC Power Flow model.
    """

    def __init__(self, system, config):
        DCPFlowBase.__init__(self, system, config)


class PFlow(PFlowData, PFlowBase):
    """
    AC Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        PFlowData.__init__(self)
        PFlowBase.__init__(self, system, config)
