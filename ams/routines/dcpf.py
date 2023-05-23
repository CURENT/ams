"""
Power flow routines.
"""
import logging
from collections import OrderedDict

import numpy as np

from andes.shared import deg2rad
from andes.utils.misc import elapsed

from ams.routines.routine import RoutineData, RoutineModel
from ams.opt.omodel import Constraint, Objective
from ams.solver.pypower.runpf import runpf, rundcpf

from ams.io.pypower import system2ppc
from ams.core.param import RParam
from ams.core.var import RAlgeb

logger = logging.getLogger(__name__)


class DCPFlowData(RoutineData):
    """
    Data class for power flow routines.
    """

    def __init__(self):
        RoutineData.__init__(self)
        # --- line ---
        self.x = RParam(info="line reactance",
                        name='x',
                        tex_name='x',
                        src='x',
                        unit='p.u.',
                        owner_name='Line',
                        )
        self.tap = RParam(info="transformer branch tap ratio",
                          name='tap',
                          src='tap',
                          tex_name='t_{ap}',
                          unit='float',
                          owner_name='Line',
                          )
        self.phi = RParam(info="transformer branch phase shift in rad",
                          name='phi',
                          src='phi',
                          tex_name=r'\phi',
                          unit='radian',
                          owner_name='Line',
                          )

        # --- load ---
        self.pd = RParam(info='active power load in system base',
                         name='pd',
                         src='p0',
                         tex_name=r'p_{d}',
                         unit='p.u.',
                         owner_name='PQ',
                         )


class DCPFlowBase(RoutineModel):
    """
    Base class for Power Flow model.

    Overload the ``solve``, ``unpack``, and ``run`` methods.
    """

    def __init__(self, system, config):
        RoutineModel.__init__(self, system, config)
        self.info = 'DC Power Flow'

        # --- bus ---
        self.aBus = RAlgeb(info='bus voltage angle',
                           unit='rad',
                           name='aBus',
                           src='a',
                           tex_name=r'a_{Bus}',
                           owner_name='Bus',
                           )
        # --- gen ---
        self.pg = RAlgeb(info='actual active power generation',
                         unit='p.u.',
                         name='pg',
                         src='p',
                         tex_name=r'p_{g}',
                         owner_name='StaticGen',
                         )
        # --- constraints ---
        self.pb = Constraint(name='pb',
                             info='power balance',
                             e_str='sum(pd) - sum(pg)',
                             type='eq',
                             )

    def solve(self, **kwargs):
        """
        Solve the DC Power Flow with PYPOWER.
        """
        ppc = system2ppc(self.system)
        res, success = rundcpf(ppc, **kwargs)
        return res, success

    def unpack(self, res):
        """
        Unpack results from PYPOWER.
        """
        system = self.system
        mva = res['baseMVA']
        # mva = self.system.config.mva

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
        for raname, ralgeb in self.ralgebs.items():
            owner = getattr(system, ralgeb.owner_name)  # instance of owner, Model or Group
            if ralgeb.src is None:          # skip if no source variable is specified
                continue
            elif hasattr(owner, 'group'):   # if owner is a Model instance
                grp = getattr(system, owner.group)
                idx=grp.get_idx()
            elif hasattr(owner, 'get_idx'): # if owner is a Group instance
                idx=owner.get_idx()
            else:
                msg = f"Failed to find valid source variable `{owner.class_name}.{ralgeb.src}` for "
                msg += f"{self.class_name}.{raname}, skip unpacking."
                logger.warning(msg)
                continue
            ralgeb.v = owner.get(src=ralgeb.src, attr='v', idx=idx)
        return True

    def run(self, **kwargs):
        """
        Run the DC Power Flow.

        Examples
        --------
        >>> ss = ams.load(ams.get_case('matpower/case14.m'))
        >>> ss.DCOPF.run()

        Other Parameters
        ----------------
        ppopt : dict
            PYPOWER options.

        Returns
        -------
        exit_code : int
            Exit code of the routine.

        # TODO: fix the kwargs input.
        """
        if not self.is_setup:
            logger.info(f"Setup model for {self.class_name}")
            self.setup()
        t0, _ = elapsed()
        res, success = self.solve(**kwargs)
        self.exit_code = 0 if success else 1
        _, s = elapsed(t0)
        self.exec_time = float(s.split(' ')[0])
        self.unpack(res)
        info = f"{self.class_name} completed in {s} with exit code {self.exit_code}."
        logger.info(info)
        return self.exit_code

    def summary(self, **kwargs):
        """
        # TODO: Print power flow summary.
        """
        pass

    def report(self, **kwargs):
        """
        Print power flow report.
        """
        pass


class DCPF(DCPFlowData, DCPFlowBase):
    """
    DC Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        DCPFlowData.__init__(self)
        DCPFlowBase.__init__(self, system, config)
