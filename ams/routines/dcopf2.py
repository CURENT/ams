"""
DCOPF routines using PYPOWER.
"""
import logging

import numpy as np

from andes.utils.misc import elapsed

from ams.core.param import RParam
from ams.core.service import NumOp, NumOpDual

from ams.pypower import runopf  # NOQA
from ams.pypower.core import ppoption  # NOQA

from ams.io.pypower import system2ppc

from ams.routines.dcpf import DCPF
from ams.routines.dcopf import DCOPF


logger = logging.getLogger(__name__)


class DCOPF2(DCOPF, DCPF):
    """
    DCOPF using PYPOWER, used for benchmark.
    """

    def __init__(self, system, config):
        DCOPF.__init__(self, system, config)

    def unpack(self, res):
        """
        Unpack results from PYPOWER.
        """
        DCPF.unpack(self, res)

        # --- Objective ---
        self.obj.v = res['f']

        self.system.recent = self.system.routines[self.class_name]
        return True

    def solve(self, method=None, **kwargs):
        """
        Solve DCOPF using PYPOWER.
        """
        ppc = system2ppc(self.system)
        ppopt = ppoption(PF_DC=True)

        res, sstats = runopf(casedata=ppc, ppopt=ppopt, **kwargs)
        return res, sstats

    def run(self, force_init=False, no_code=True,
            method=None, **kwargs) -> bool:
        """
        Run DCOPF using PYPOWER.
        """
        if not self.initialized:
            self.init(force=force_init, no_code=no_code)
        t0, _ = elapsed()
        res, sstats = self.solve(method=method)
        self.exit_code = 0 if res['success'] else 1
        _, s = elapsed(t0)
        self.exec_time = float(s.split(' ')[0])
        self.unpack(res)
        n_iter = int(sstats['num_iters'])
        n_iter_str = f"{n_iter} iterations " if n_iter > 1 else f"{n_iter} iteration "
        if self.exit_code == 0:
            msg = f"{self.class_name} solved in {s}, converged after "
            msg += n_iter_str + f"using solver {sstats['solver_name']}."
            logger.info(msg)
            return True
        else:
            msg = f"{self.class_name} failed after "
            msg += f"{int(sstats['num_iters'])} iterations using solver "
            msg += f"{sstats['solver_name']}!"
            logger.warning(msg)
            return False
