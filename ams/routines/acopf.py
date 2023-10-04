"""
Power flow routines.
"""
import logging  # NOQA
from collections import OrderedDict  # NOQA

from ams.solver.pypower.runopf import runopf  # NOQA

from ams.io.pypower import system2ppc  # NOQA
from ams.core.param import RParam  # NOQA

from ams.routines.dcopf import DCOPFData  # NOQA
from ams.routines.dcpf import DCPFlowBase  # NOQA
from ams.opt.omodel import Var, Constraint, Objective  # NOQA

logger = logging.getLogger(__name__)


class ACOPFData(DCOPFData):
    """
    AC Power Flow routine.
    """

    def __init__(self):
        DCOPFData.__init__(self)
        self.ql = RParam(info='reactive power demand (system base)',
                         name='ql', tex_name=r'q_{l}',
                         model='mats', src='ql',
                         unit='p.u.',)


class ACOPFBase(DCPFlowBase):
    """
    Base class for AC Power Flow model.
    """

    def __init__(self, system, config):
        DCPFlowBase.__init__(self, system, config)
        self.info = 'AC Optimal Power Flow'
        self.type = 'ACED'
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
        success = res['success']
        return res, success

    def unpack(self, res):
        """
        Unpack results from PYPOWER.
        """
        super().unpack(res)

        # --- Objective ---
        self.obj.v = res['f']  # TODO: check unit

        self.system.recent = self.system.routines[self.class_name]
        return True


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
                        name='aBus', tex_name=r'a_{Bus}',
                        model='Bus', src='a',)
        self.vBus = Var(info='Bus voltage magnitude',
                        unit='p.u.',
                        name='vBus', tex_name=r'v_{Bus}',
                        src='v', model='Bus',)
        # --- gen ---
        self.pg = Var(info='Gen active power',
                      unit='p.u.',
                      name='pg', tex_name=r'p_{g}',
                      model='StaticGen', src='p',)
        self.qg = Var(info='Gen reactive power',
                      unit='p.u.',
                      name='qg', tex_name=r'q_{g}',
                      model='StaticGen', src='q',)
        # --- constraints ---
        self.pb = Constraint(name='pb',
                             info='power balance',
                             e_str='sum(pl) - sum(pg)',
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
