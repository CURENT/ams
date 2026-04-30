"""
Distributional optimal power flow (DOPF).
"""
import numpy as np

import cvxpy as cp

from ams.core.param import RParam

from ams.routines.dcopf import DCOPF

from ams.opt import Var, Constraint, Objective


# --- e_fn callables (Phase 4.3 migration) ---

def _qglb(r):
    return -r.qg + cp.multiply(r.ug, r.qmin) <= 0


def _qgub(r):
    return r.qg - cp.multiply(r.ug, r.qmax) <= 0


def _vu(r):
    return r.vsq - r.vmax ** 2 <= 0


def _vl(r):
    return -r.vsq + r.vmin ** 2 <= 0


def _lvd(r):
    return r.CftT @ r.vsq - (cp.multiply(r.r, r.plf) + cp.multiply(r.x, r.qlf)) == 0


def _qb(r):
    return cp.sum(r.qd) - cp.sum(r.qg) == 0


def _dopfvis_obj(r):
    # Original e_str used cvxpy's deprecated `*` which is matrix-multiply
    # for vectors. c2/c1/c0/ug/pg are size 5 (StaticGen); cm/cd/M/D are
    # size 4 (VSG). Each term is a scalar dot product; preserve those
    # semantics with `@`.
    return (r.c2 @ r.pg ** 2 + r.c1 @ r.pg + r.ug @ r.c0
            + r.cm @ r.M + r.cd @ r.D)


class DOPF(DCOPF):
    """
    Linearzied distribution OPF, where power loss are ignored.

    UNDER DEVELOPMENT!

    References
    ----------
    1. L. Bai, J. Wang, C. Wang, C. Chen, and F. Li, “Distribution Locational Marginal Pricing (DLMP)
       for Congestion Management and Voltage Support,” IEEE Trans. Power Syst., vol. 33, no. 4,
       pp. 4061-4073, Jul. 2018, doi: 10.1109/TPWRS.2017.2767632.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)
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
        self.qglb = Constraint(name='qglb', info='qg min', e_fn=_qglb,)
        self.qgub = Constraint(name='qgub', info='qg max', e_fn=_qgub,)
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
                             e_fn=_vu,)
        self.vl = Constraint(name='vl',
                             info='Voltage lower limit',
                             e_fn=_vl,)
        # --- line ---
        self.qlf = Var(info='line reactive power',
                       name='qlf', tex_name=r'q_{lf}',
                       unit='p.u.', model='Line',)
        self.lvd = Constraint(info='line voltage drop',
                              name='lvd', e_fn=_lvd,)
        # --- power balance ---
        # NOTE: following Eqn seems to be wrong, double check
        # g_Q(\Theta, V, Q_g) = B_{bus}V\Theta + Q_{bus,shift} + Q_d + B_{sh} - C_gQ_g = 0
        self.qb = Constraint(info='reactive power balance',
                             name='qb', e_fn=_qb,)

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

    References
    ----------
    1. L. Bai, J. Wang, C. Wang, C. Chen, and F. Li, “Distribution Locational Marginal Pricing (DLMP)
       for Congestion Management and Voltage Support,” IEEE Trans. Power Syst., vol. 33, no. 4,
       pp. 4061-4073, Jul. 2018, doi: 10.1109/TPWRS.2017.2767632.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

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
        self.obj = Objective(name='tc',
                             info='total cost', unit='$',
                             e_fn=_dopfvis_obj,
                             sense='min',)
