"""
Continuous power flow routine.
"""
import logging

from ams.pypower import runcpf

from ams.io.pypower import system2ppc
from ams.pypower.core import ppoption

from ams.routines.pflow import PFlow

logger = logging.getLogger(__name__)


class CPF(PFlow):
    """
    Continuous power flow.

    Still under development, not ready for use.
    """

    def __init__(self, system, config):
        PFlow.__init__(self, system, config)
        self.info = 'AC continuous power flow'
        self.type = 'PF'

    def solve(self, method=None, **kwargs):
        """
        Solve the CPF using PYPOWER.
        """
        ppc = system2ppc(self.system)
        ppopt = ppoption()
        res, success, sstats = runcpf(casedata=ppc, ppopt=ppopt, **kwargs)
        return res, success, sstats

    # FIXME: unpack results?

    def run(self, force_init=False, no_code=True,
            method='newton', **kwargs):
        """
        Run continuous power flow using PYPOWER.

        Examples
        --------
        >>> ss = ams.load(ams.get_case('matpower/case14.m'))
        >>> ss.CPF.run()

        Parameters
        ----------
        force_init : bool
            Force initialization.
        no_code : bool
            Disable showing code.
        method : str
            Method for solving the power flow.

        Returns
        -------
        exit_code : int
            Exit code of the routine.
        """
        super().run(force_init=force_init,
                    no_code=no_code, method=method,
                    **kwargs, )
