"""
Power flow routines.
"""
import logging

from andes.shared import deg2rad
from andes.utils.misc import elapsed

from ams.routines.routine import RoutineBase
from ams.opt.omodel import Var
from ams.pypower import runpf
from ams.pypower.core import ppoption

from ams.io.pypower import system2ppc
from ams.core.param import RParam

logger = logging.getLogger(__name__)


class DCPFlowBase(RoutineBase):
    """
    Base class for power flow.

    Overload the ``solve``, ``unpack``, and ``run`` methods.
    """

    def __init__(self, system, config):
        RoutineBase.__init__(self, system, config)
        self.info = 'DC Power Flow'
        self.type = 'PF'

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

    def unpack(self, res):
        """
        Unpack results from PYPOWER.
        """
        system = self.system
        mva = res['baseMVA']

        # --- copy results from routine algeb into system algeb ---
        # --- Bus ---
        system.Bus.v.v = res['bus'][:, 7]               # voltage magnitude
        system.Bus.a.v = res['bus'][:, 8] * deg2rad     # voltage angle

        # --- PV ---
        system.PV.p.v = res['gen'][system.Slack.n:, 1] / mva        # active power
        system.PV.q.v = res['gen'][system.Slack.n:, 2] / mva        # reactive power

        # --- Slack ---
        system.Slack.p.v = res['gen'][:system.Slack.n, 1] / mva     # active power
        system.Slack.q.v = res['gen'][:system.Slack.n, 2] / mva     # reactive power

        # --- copy results from system algeb into routine algeb ---
        for vname, var in self.vars.items():
            owner = getattr(system, var.model)  # instance of owner, Model or Group
            if var.src is None:          # skip if no source variable is specified
                continue
            elif hasattr(owner, 'group'):   # if owner is a Model instance
                grp = getattr(system, owner.group)
                idx = grp.get_idx()
            elif hasattr(owner, 'get_idx'):  # if owner is a Group instance
                idx = owner.get_idx()
            else:
                msg = f"Failed to find valid source variable `{owner.class_name}.{var.src}` for "
                msg += f"{self.class_name}.{vname}, skip unpacking."
                logger.warning(msg)
                continue
            try:
                logger.debug(f"Unpacking {vname} into {owner.class_name}.{var.src}.")
                var.v = owner.get(src=var.src, attr='v', idx=idx)
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

        res, success, sstats = runpf(casedata=ppc, ppopt=ppopt)
        self.converged = bool(success)
        return res, self.converged, sstats

    def run(self, force_init=False, no_code=True,
            method=None, **kwargs):
        """
        Run DC pwoer flow.

        Examples
        --------
        >>> ss = ams.load(ams.get_case('matpower/case14.m'))
        >>> ss.DCPF.run()

        Parameters
        ----------
        force_init : bool
            Force initialization.
        no_code : bool
            Disable showing code.
        method : str
            Placeholder for future use.

        Returns
        -------
        exit_code : int
            Exit code of the routine.
        """
        if not self.initialized:
            self.init(force=force_init, no_code=no_code)
        t0, _ = elapsed()
        res, success, sstats = self.solve(method=method)
        self.exit_code = 0 if success else 1
        _, s = elapsed(t0)
        self.exec_time = float(s.split(' ')[0])
        n_iter = int(sstats['num_iters'])
        n_iter_str = f"{n_iter} iterations " if n_iter > 1 else f"{n_iter} iteration "
        if self.exit_code == 0:
            msg = f"{self.class_name} solved in {s}, converged after "
            msg += n_iter_str + f"using solver {sstats['solver_name']}."
            logger.info(msg)
            self.unpack(res)
            return True
        else:
            msg = f"{self.class_name} failed after "
            msg += f"{int(sstats['num_iters'])} iterations using solver "
            msg += f"{sstats['solver_name']}!"
            logger.warning(msg)
            return False

    def summary(self, **kwargs):
        """
        # TODO: Print power flow summary.
        """
        pass

    def enable(self, name):
        raise NotImplementedError

    def disable(self, name):
        raise NotImplementedError


class DCPF(DCPFlowBase):
    """
    DC power flow.

    Notes
    -----
    1. DCPF is solved with PYPOWER ``runpf`` function.
    2. DCPF formulation is not complete yet, but this does not affect the
       results because the data are passed to PYPOWER for solving.
    """

    def __init__(self, system, config):
        DCPFlowBase.__init__(self, system, config)
        self.info = 'DC Power Flow'

        # --- bus ---
        self.aBus = Var(info='bus voltage angle',
                        unit='rad',
                        name='aBus', tex_name=r'a_{Bus}',
                        model='Bus', src='a',)
        # --- gen ---
        self.pg = Var(info='actual active power generation',
                      unit='p.u.',
                      name='pg', tex_name=r'p_{g}',
                      model='StaticGen', src='p',)
