"""
Power flow routines.
"""
import logging

from andes.shared import deg2rad
from andes.utils.misc import elapsed

from ams.opt import Var
from ams.pypower import runpf
from ams.pypower.core import ppoption
from ams.routines.dcopf import DCOPF

from ams.io.pypower import system2ppc
from ams.core.param import RParam

logger = logging.getLogger(__name__)


class DCPF2(DCOPF):
    """
    DC power flow, overload the ``solve``, ``unpack``, and ``run`` methods.

    Notes
    -----
    1. DCPF is solved with PYPOWER ``runpf`` function.
    2. DCPF formulation is not complete yet, but this does not affect the
       results because the data are passed to PYPOWER for solving.
    """

    def __init__(self, system, config):
        DCOPF.__init__(self, system, config)
        self.info = 'DC Power Flow'
        self.type = 'PF'

        self.obj.e_str = '0'

    def summary(self, **kwargs):
        """
        # TODO: Print power flow summary.
        """
        raise NotImplementedError

    def enable(self, name):
        raise NotImplementedError

    def disable(self, name):
        raise NotImplementedError

    def dc2ac(self, name):
        raise NotImplementedError
