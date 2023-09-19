"""
Power flow routines.
"""
import logging
from collections import OrderedDict

import numpy as np

from andes.shared import deg2rad
from andes.utils.misc import elapsed

from ams.solver.pypower.runopf import runopf

from ams.io.pypower import system2ppc
from ams.core.param import RParam

from ams.routines.pflow import PFlowData, PFlowModel
from ams.routines.dcopf import DCOPFData
from ams.routines.routine import RoutineModel
from ams.opt.omodel import Var, Constraint, Objective

logger = logging.getLogger(__name__)


class ACOPFData(DCOPFData):
    """
    AC Power Flow routine.
    """

    def __init__(self):
        DCOPFData.__init__(self)
        self.qd = RParam(info='reactive power demand (system base)',
                         name='qd',
                         src='q0',
                         tex_name=r'q_{d}',
                         unit='p.u.',
                         model='PQ',
                         )


class ACOPFBase(RoutineModel):
    """
    Base class for AC Power Flow model.
    """

    def __init__(self, system, config):
        RoutineModel.__init__(self, system, config)
        # NOTE: ACOPF does not receive data from dynamic
        self.map1 = OrderedDict()
        self.map2 = OrderedDict([
            ('Bus', {
                'vBus': 'v0',
            }),
            # NOTE: separating PV and Slack rather than using StaticGen
            # might introduce error when sync with ANDES dynamic
            ('StaticGen', {
                'pg': 'p0',
            }),
        ])

    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        res = runopf(ppc, **kwargs)
        return res

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
        for raname, var in self.vars.items():
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
                msg += f"{self.class_name}.{raname}, skip unpacking."
                logger.warning(msg)
                continue
            var.v = owner.get(src=var.src, attr='v', idx=idx)

        # --- Objective ---
        self.obj.v = res['f']  # TODO: check unit

        self.system.recent = self.system.routines[self.class_name]
        return True

    def run(self, **kwargs):
        """
        Run the routine.
        """
        if not self.initialized:
            logger.info(f"Initialize routine {self.class_name}")
            self.init()
        t0, _ = elapsed()
        res = self.solve(**kwargs)
        self.exit_code = int(1 - res['success'])
        _, s = elapsed(t0)
        self.exec_time = float(s.split(' ')[0])
        self.unpack(res)
        if self.exit_code == 0:
            info = f"{self.class_name} completed in {s} with exit code {self.exit_code}."
            logger.info(info)
            return True
        else:
            info = f"{self.class_name} failed!"
            logger.error(info)
            return False


class ACOPFModel(ACOPFBase):
    """
    ACOPF model.
    """

    def __init__(self, system, config):
        ACOPFBase.__init__(self, system, config)
        self.info = 'AC Optimal Power Flow'
        self.type = 'ACED'
        # --- bus ---
        self.aBus = Var(info='Bus voltage angle',
                        unit='rad',
                        name='aBus',
                        src='a',
                        tex_name=r'a_{Bus}',
                        model='Bus',
                        )
        self.vBus = Var(info='Bus voltage magnitude',
                        unit='p.u.',
                        name='vBus',
                        src='v',
                        tex_name=r'v_{Bus}',
                        model='Bus',
                        )
        # --- gen ---
        self.pg = Var(info='Gen active power',
                      unit='p.u.',
                      name='pg',
                      src='p',
                      tex_name=r'p_{g}',
                      model='StaticGen',
                      )
        self.qg = Var(info='Gen reactive power',
                      unit='p.u.',
                      name='qg',
                      src='q',
                      tex_name=r'q_{g}',
                      model='StaticGen',
                      )
        # --- constraints ---
        self.pb = Constraint(name='pb',
                             info='power balance',
                             e_str='sum(pd) - sum(pg)',
                             type='eq',
                             )
        # TODO: ACOPF formulation
        # --- objective ---
        self.obj = Objective(name='tc',
                             info='total cost',
                             e_str='sum(c2 * pg**2 + c1 * pg + c0)',
                             sense='min',)


class ACOPF(ACOPFData, ACOPFModel):
    """
    Standard AC optimal power flow.

    Notes
    -----
    1. ACOPF is solved with PYPOWER ``runopf`` function.
    2. ACOPF formulation in AMS style is NOT DONE YET,
       but this does not affect the results
       because the data are passed to PYPOWER for solving.
    """

    def __init__(self, system=None, config=None):
        ACOPFData.__init__(self)
        ACOPFModel.__init__(self, system, config)
