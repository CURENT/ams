"""
DC power flow routines using PYPOWER.
"""
import logging

from andes.shared import deg2rad
from andes.utils.misc import elapsed

from ams.routines.routine import RoutineBase
from ams.opt import Var
from ams.pypower import runpf
from ams.pypower.core import ppoption

from ams.io.pypower import system2ppc
from ams.core.param import RParam

logger = logging.getLogger(__name__)


class DCPF0(RoutineBase):
    """
    DC power flow using PYPOWER.

    This class is deprecated as of version 0.9.12 and will be removed in 1.1.0.

    Notes
    -----
    1. DCPF is solved with PYPOWER ``runpf`` function.
    2. DCPF formulation is not complete yet, but this does not affect the
       results because the data are passed to PYPOWER for solving.
    """

    def __init__(self, system, config):
        RoutineBase.__init__(self, system, config)
        self.info = 'DC Power Flow'
        self.type = 'PF'

        self.ug = RParam(info='Gen connection status',
                         name='ug', tex_name=r'u_{g}',
                         model='StaticGen', src='u',
                         no_parse=True)

        # --- routine data ---
        self.x = RParam(info="line reactance",
                        name='x', tex_name='x',
                        unit='p.u.',
                        model='Line', src='x',)
        self.tap = RParam(info="transformer branch tap ratio",
                          name='tap', tex_name=r't_{ap}',
                          model='Line', src='tap',
                          unit='float',)
        self.phi = RParam(info="transformer branch phase shift in rad",
                          name='phi', tex_name=r'\phi',
                          model='Line', src='phi',
                          unit='radian',)

        # --- load ---
        self.pd = RParam(info='active deman',
                         name='pd', tex_name=r'p_{d}',
                         unit='p.u.',
                         model='StaticLoad', src='p0')
        # --- gen ---
        self.pg = Var(info='Gen active power',
                      unit='p.u.',
                      name='pg', tex_name=r'p_{g}',
                      model='StaticGen', src='p',)

        # --- bus ---
        self.aBus = Var(info='bus voltage angle',
                        unit='rad',
                        name='aBus', tex_name=r'a_{Bus}',
                        model='Bus', src='a',)

        # --- line flow ---
        self.plf = Var(info='Line flow',
                       unit='p.u.',
                       name='plf', tex_name=r'p_{lf}',
                       model='Line',)

    def unpack(self, res):
        """
        Unpack results from PYPOWER.
        """
        system = self.system
        mva = res['baseMVA']

        # --- copy results from ppc into system algeb ---
        # --- Bus ---
        system.Bus.v.v = res['bus'][:, 7]               # voltage magnitude
        system.Bus.a.v = res['bus'][:, 8] * deg2rad     # voltage angle

        # --- PV ---
        system.PV.p.v = res['gen'][system.Slack.n:, 1] / mva        # active power
        system.PV.q.v = res['gen'][system.Slack.n:, 2] / mva        # reactive power

        # --- Slack ---
        system.Slack.p.v = res['gen'][:system.Slack.n, 1] / mva     # active power
        system.Slack.q.v = res['gen'][:system.Slack.n, 2] / mva     # reactive power

        # --- Line ---
        self.plf.optz.value = res['branch'][:, 13] / mva  # line flow

        # --- copy results from system algeb into routine algeb ---
        for vname, var in self.vars.items():
            owner = getattr(system, var.model)  # instance of owner, Model or Group
            if var.src is None:          # skip if no source variable is specified
                continue
            elif hasattr(owner, 'group'):   # if owner is a Model instance
                grp = getattr(system, owner.group)
                idx = grp.get_all_idxes()
            elif hasattr(owner, 'get_idx'):  # if owner is a Group instance
                idx = owner.get_all_idxes()
            else:
                msg = f"Failed to find valid source variable `{owner.class_name}.{var.src}` for "
                msg += f"{self.class_name}.{vname}, skip unpacking."
                logger.warning(msg)
                continue
            try:
                logger.debug(f"Unpacking {vname} into {owner.class_name}.{var.src}.")
                var.optz.value = owner.get(src=var.src, attr='v', idx=idx)
            except AttributeError:
                logger.debug(f"Failed to unpack {vname} into {owner.class_name}.{var.src}.")
                continue
        self.system.recent = self.system.routines[self.class_name]
        return True

    def solve(self, method=None):
        """
        Solve DC power flow using PYPOWER.
        """
        ppc = system2ppc(self.system)
        ppopt = ppoption(PF_DC=True)

        res, sstats = runpf(casedata=ppc, ppopt=ppopt)
        return res, sstats

    def run(self, **kwargs):
        """
        Run DC pwoer flow.
        *args and **kwargs go to `self.solve()`, which are not used yet.

        Examples
        --------
        >>> ss = ams.load(ams.get_case('matpower/case14.m'))
        >>> ss.DCPF.run()

        Parameters
        ----------
        method : str
            Placeholder for future use.

        Returns
        -------
        exit_code : int
            Exit code of the routine.
        """
        if not self.initialized:
            self.init()
        t0, _ = elapsed()

        res, sstats = self.solve(**kwargs)
        self.converged = res['success']
        self.exit_code = 0 if res['success'] else 1
        _, s = elapsed(t0)
        self.exec_time = float(s.split(' ')[0])
        n_iter = int(sstats['num_iters'])
        n_iter_str = f"{n_iter} iterations " if n_iter > 1 else f"{n_iter} iteration "
        if self.exit_code == 0:
            msg = f"<{self.class_name}> solved in {s}, converged in "
            msg += n_iter_str + f"with {sstats['solver_name']}."
            logger.warning(msg)
            try:
                self.unpack(res)
            except Exception as e:
                logger.error(f"Failed to unpack results from {self.class_name}.\n{e}")
                return False
            self.system.report()
            return True
        else:
            msg = f"{self.class_name} failed in "
            msg += f"{int(sstats['num_iters'])} iterations with "
            msg += f"{sstats['solver_name']}!"
            logger.warning(msg)
            return False

    def summary(self, **kwargs):
        """
        # TODO: Print power flow summary.
        """
        raise NotImplementedError

    def enable(self, name):
        raise NotImplementedError

    def disable(self, name):
        raise NotImplementedError
