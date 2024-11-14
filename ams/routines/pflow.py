"""
Power flow routines.
"""
import logging
from collections import OrderedDict

from ams.pypower import runpf

from ams.io.pypower import system2ppc
from ams.pypower.core import ppoption
from ams.core.param import RParam

from ams.routines.dcpf import DCPF
from ams.opt.omodel import Var

logger = logging.getLogger(__name__)


class PFlow(DCPF):
    """
    AC Power Flow routine.

    Notes
    -----
    1. AC pwoer flow is solved with PYPOWER ``runpf`` function.
    2. AC power flow formulation in AMS style is NOT DONE YET,
       but this does not affect the results
       because the data are passed to PYPOWER for solving.
    """

    def __init__(self, system, config):
        DCPF.__init__(self, system, config)
        self.info = "AC Power Flow"
        self.type = "PF"

        self.config.add(OrderedDict((('qlim', 0),
                                     )))
        self.config.add_extra("_help",
                              qlim="Enforce generator q limits",
                              )
        self.config.add_extra("_alt",
                              qlim=(0, 1, 2),
                              )

        self.qd = RParam(info="reactive power load in system base",
                         name="qd", tex_name=r"q_{d}",
                         unit="p.u.",
                         model="StaticLoad", src="q0",)

        # --- bus ---
        self.vBus = Var(info="bus voltage magnitude",
                        unit="p.u.",
                        name="vBus", tex_name=r"v_{Bus}",
                        model="Bus", src="v",)
        # --- gen ---
        self.qg = Var(info="reactive power generation",
                      unit="p.u.",
                      name="qg", tex_name=r"q_{g}",
                      model="StaticGen", src="q",)
        # NOTE: omit AC power flow formulation here

    def solve(self, method="newton", **kwargs):
        """
        Solve the AC power flow using PYPOWER.
        """
        ppc = system2ppc(self.system)

        method_map = dict(newton=1, fdxb=2, fdbx=3, gauss=4)
        alg = method_map.get(method)
        if alg == 4:
            msg = "Gauss method is not fully tested yet, not recommended!"
            logger.warning(msg)
        if alg is None:
            msg = f"Invalid method `{method}` for PFlow."
            raise ValueError(msg)
        ppopt = ppoption(PF_ALG=alg, ENFORCE_Q_LIMS=self.config.qlim, **kwargs)

        res, sstats = runpf(casedata=ppc, ppopt=ppopt)
        return res, sstats

    def run(self, **kwargs):
        """
        Run AC power flow using PYPOWER.

        Currently, four methods are supported: 'newton', 'fdxb', 'fdbx', 'gauss',
        for Newton's method, fast-decoupled, XB, fast-decoupled, BX, and Gauss-Seidel,
        respectively.

        Note that gauss method is not recommended because it seems to be much
        more slower than the other three methods and not fully tested yet.

        Examples
        --------
        >>> ss = ams.load(ams.get_case('matpower/case14.m'))
        >>> ss.PFlow.run()

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
        return super().run(**kwargs,)
