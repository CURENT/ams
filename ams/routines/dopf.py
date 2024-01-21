"""
Distributional optimal power flow (DOPF).
"""
import numpy as np

from ams.core.param import RParam

from ams.routines.dcopf import DCOPF

from ams.opt.omodel import Var, Constraint, Objective


class DOPF(DCOPF):
    """
    Linearzied distribution OPF, where power loss are ignored.

    UNDER DEVELOPMENT!

    Reference:

    [1] L. Bai, J. Wang, C. Wang, C. Chen, and F. Li, “Distribution Locational Marginal Pricing (DLMP)
    for Congestion Management and Voltage Support,” IEEE Trans. Power Syst., vol. 33, no. 4,
    pp. 4061–4073, Jul. 2018, doi: 10.1109/TPWRS.2017.2767632.
    """

    def __init__(self, system, config):
        DCOPF.__init__(self, system, config)
        self.info = 'Linearzied distribution OPF'
        self.type = 'DED'

        # -- Data Section --
        # ---  generator ---
        self.qmax = RParam(info='generator maximum reactive power',
                           name='qmax', tex_name=r'q_{max}', unit='p.u.',
                           model='StaticGen', src='qmax',)
        self.qmin = RParam(info='generator minimum reactive power',
                           name='qmin', tex_name=r'q_{min}', unit='p.u.',
                           model='StaticGen', src='qmin',)
        # --- load ---
        self.qd = RParam(info='reactive demand',
                         name='qd', tex_name=r'q_{d}', unit='p.u.',
                         model='StaticLoad', src='q0',)
        # --- bus ---
        self.vmax = RParam(info="Bus voltage upper limit",
                           name='vmax', tex_name=r'v_{max}',
                           unit='p.u.',
                           model='Bus', src='vmax', no_parse=True,
                           )
        self.vmin = RParam(info="Bus voltage lower limit",
                           name='vmin', tex_name=r'v_{min}',
                           unit='p.u.',
                           model='Bus', src='vmin', no_parse=True,)
        # --- line ---
        self.r = RParam(info='line resistance',
                        name='r', tex_name='r', unit='p.u.',
                        model='Line', src='r',
                        no_parse=True,)
        self.x = RParam(info='line reactance',
                        name='x', tex_name='x', unit='p.u.',
                        model='Line', src='x',
                        no_parse=True,)
        # --- Model Section ---
        # --- generator ---
        self.qg = Var(info='Gen reactive power',
                      name='qg', tex_name=r'q_{g}', unit='p.u.',
                      model='StaticGen', src='q',)
        self.qglb = Constraint(name='qglb', type='uq',
                               info='qg min',
                               e_str='-qg + mul(ug, qmin)',)
        self.qgub = Constraint(name='qgub', type='uq',
                               info='qg max',
                               e_str='qg - mul(ug, qmax)',)
        # --- bus ---
        self.v = Var(info='Bus voltage',
                     name='v', tex_name=r'v',
                     unit='p.u.',
                     model='Bus', src='v')
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
        # --- line ---
        self.qlf = Var(info='line reactive power',
                       name='qlf', tex_name=r'q_{lf}',
                       unit='p.u.', model='Line',)
        self.lvd = Constraint(info='line voltage drop',
                              name='lvd', type='eq',
                              e_str='CftT@vsq - (r * plf + x * qlf)',)
        # --- power balance ---
        # NOTE: following Eqn seems to be wrong, double check
        # g_Q(\Theta, V, Q_g) = B_{bus}V\Theta + Q_{bus,shift} + Q_d + B_{sh} - C_gQ_g = 0
        self.qb = Constraint(info='reactive power balance',
                             name='qb', type='eq',
                             e_str='sum(qd) - sum(qg)',)

        # --- objective ---
        # NOTE: no need to revise objective function

    def _post_solve(self):
        self.v.optz.value = np.sqrt(self.vsq.optz.value)
        return super()._post_solve()


class DOPFVIS(DOPF):
    """
    Linearzied distribution OPF with variables for virtual inertia and damping from from REGCV1,
    where power loss are ignored.

    UNDER DEVELOPMENT!

    Reference:

    [1] L. Bai, J. Wang, C. Wang, C. Chen, and F. Li, “Distribution Locational Marginal Pricing (DLMP)
    for Congestion Management and Voltage Support,” IEEE Trans. Power Syst., vol. 33, no. 4,
    pp. 4061–4073, Jul. 2018, doi: 10.1109/TPWRS.2017.2767632.
    """

    def __init__(self, system, config):
        DOPF.__init__(self, system, config)

        # --- params ---
        self.cm = RParam(info='Virtual inertia cost',
                         name='cm', src='cm',
                         tex_name=r'c_{m}', unit=r'$/s',
                         model='VSGCost',
                         indexer='reg', imodel='VSG')
        self.cd = RParam(info='Virtual damping cost',
                         name='cd', src='cd',
                         tex_name=r'c_{d}', unit=r'$/(p.u.)',
                         model='VSGCost',
                         indexer='reg', imodel='VSG',)
        # --- vars ---
        self.M = Var(info='Emulated startup time constant (M=2H) from REGCV1',
                     name='M', tex_name=r'M', unit='s',
                     model='VSG',)
        self.D = Var(info='Emulated damping coefficient from REGCV1',
                     name='D', tex_name=r'D', unit='p.u.',
                     model='VSG',)
        obj = 'sum(c2 * pg**2 + c1 * pg + ug * c0 + cm * M + cd * D)'
        self.obj = Objective(name='tc',
                             info='total cost', unit='$',
                             e_str=obj,
                             sense='min',)
