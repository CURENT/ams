"""
Distributional optimal power flow (DOPF).
"""
import numpy as np

from ams.core.param import RParam
from ams.core.service import NumOp

from ams.routines.dcopf import DCOPF

from ams.opt.omodel import Var, Constraint, Objective


class LDOPF(DCOPF):
    """
    Linearzied distribution OPF, where power loss are ignored.

    Reference:

    [1] L. Bai, J. Wang, C. Wang, C. Chen, and F. Li, “Distribution Locational Marginal Pricing (DLMP)
    for Congestion Management and Voltage Support,” IEEE Trans. Power Syst., vol. 33, no. 4,
    pp. 4061–4073, Jul. 2018, doi: 10.1109/TPWRS.2017.2767632.
    """

    def __init__(self, system, config):
        DCOPF.__init__(self, system, config)
        self.info = 'Linearzied distribution OPF'
        self.type = 'DED'

        # --- params ---
        self.ql = RParam(info='reactive power demand connected to Bus (system base)',
                         name='ql', tex_name=r'q_{l}', unit='p.u.',
                         model='mats', src='ql',)
        self.vmax = RParam(info="Bus voltage upper limit",
                           name='vmax', tex_name=r'v_{max}', unit='p.u.',
                           model='Bus', src='vmax', no_parse=True,
                           )
        self.vmin = RParam(info="Bus voltage lower limit",
                           name='vmin', tex_name=r'v_{min}', unit='p.u.',
                           model='Bus', src='vmin', no_parse=True,)
        self.r = RParam(info='line resistance',
                        name='r', tex_name='r', unit='p.u.',
                        model='Line', src='r')
        self.x = RParam(info='line reactance',
                        name='x', tex_name='x', unit='p.u.',
                        model='Line', src='x', )
        self.qmax = RParam(info='generator maximum reactive power (system base)',
                           name='qmax', tex_name=r'q_{max}', unit='p.u.',
                           model='StaticGen', src='qmax',)
        self.qmin = RParam(info='generator minimum reactive power (system base)',
                           name='qmin', tex_name=r'q_{min}', unit='p.u.',
                           model='StaticGen', src='qmin',)
        # --- vars ---
        self.qg = Var(info='Gen reactive power (system base)',
                      name='qg', tex_name=r'q_{g}', unit='p.u.',
                      model='StaticGen', src='q',)
        self.qglb = Constraint(name='qglb', type='uq',
                               info='Gen reactive power lower limit',
                               e_str='-qg + qmin',)
        self.qgub = Constraint(name='qgub', type='uq',
                               info='Gen reactive power upper limit',
                               e_str='qg - qmax',)
        # TODO: add qg limit,  lb=self.qmin, ub=self.qmax,
        self.qn = Var(info='Bus reactive power',
                      name='qn', tex_name=r'q_{n}', unit='p.u.',
                      model='Bus',)

        self.vsq = Var(info='square of Bus voltage',
                       name='vsq', tex_name=r'v^{2}', unit='p.u.',
                       model='Bus',)
        self.vu = Constraint(name='vu',
                             info='Voltage upper limit',
                             e_str='vsq - vmax**2',
                             type='uq',)
        self.vl = Constraint(name='vl',
                             info='Voltage lower limit',
                             e_str='-vsq + vmin**2',
                             type='uq',)

        self.qlf = Var(info='line reactive power',
                       name='qlf', tex_name=r'q_{lf}', unit='p.u.',
                       model='Line',)

        # --- constraints ---
        self.pinj.e_str = 'CftT@plf - pl - pn'
        self.qinj = Constraint(name='qinj',
                               info='node reactive power injection',
                               e_str='CftT@qlf - ql - qn',
                               type='eq',)

        self.qb = Constraint(name='qb', info='reactive power balance',
                             e_str='sum(ql) - sum(qg)',
                             type='eq',)

        self.lvd = Constraint(name='lvd',
                              info='line voltage drop',
                              e_str='Cft@vsq - (r * plf + x * qlf)',
                              type='eq',)

        # --- objective ---
        # NOTE: no need to revise objective function

    def unpack(self, **kwargs):
        super().unpack(**kwargs)
        vBus = np.sqrt(self.vsq.v)
        self.system.Bus.set(src='v', attr='v', value=vBus, idx=self.vsq.get_idx())


class LDOPF2(LDOPF):
    """
    Linearzied distribution OPF with variables for virtual inertia and damping from from REGCV1,
    where power loss are ignored.

    ERROR: the formulation is problematic, check later.

    Reference:

    [1] L. Bai, J. Wang, C. Wang, C. Chen, and F. Li, “Distribution Locational Marginal Pricing (DLMP)
    for Congestion Management and Voltage Support,” IEEE Trans. Power Syst., vol. 33, no. 4,
    pp. 4061–4073, Jul. 2018, doi: 10.1109/TPWRS.2017.2767632.
    """

    def __init__(self, system, config):
        LDOPF.__init__(self, system, config)

        # --- params ---
        self.cm = RParam(info='Virtual inertia cost',
                         name='cm', src='cm',
                         tex_name=r'c_{m}', unit=r'$/s',
                         model='REGCV1Cost',
                         indexer='reg', imodel='REGCV1')
        self.cd = RParam(info='Virtual damping cost',
                         name='cd', src='cd',
                         tex_name=r'c_{d}', unit=r'$/(p.u.)',
                         model='REGCV1Cost',
                         indexer='reg', imodel='REGCV1',)
        # --- vars ---
        self.M = Var(info='Emulated startup time constant (M=2H) from REGCV1',
                     name='M', tex_name=r'M', unit='s',
                     model='REGCV1',)
        self.D = Var(info='Emulated damping coefficient from REGCV1',
                     name='D', tex_name=r'D', unit='p.u.',
                     model='REGCV1',)
        self.obj = Objective(name='tc',
                             info='total cost', unit='$',
                             e_str='sum(c2 * pg**2 + c1 * pg + ug * c0 + cm * M + cd * D)',
                             sense='min',)
