"""
Distributional optimal power flow (DOPF).
"""
from collections import OrderedDict
import numpy as np
from scipy.optimize import linprog

from ams.core.param import RParam
from ams.core.service import NumOp

from ams.routines.routine import RoutineData, RoutineModel
from ams.routines.dcopf import DCOPFData, DCOPFBase, DCOPFModel

from ams.opt.omodel import Var, Constraint, Objective


class DOPFData(DCOPFData):
    """
    Data for DOPF.
    """

    def __init__(self):
        DCOPFData.__init__(self)
        self.qd = RParam(info='reactive power demand connected to Bus (system base)',
                         name='qd', tex_name=r'q_{d}', unit='p.u.')
        self.vmax = RParam(info="Bus voltage upper limit",
                           name='vmax', tex_name=r'v_{max}', unit='p.u.',
                           model='Bus', src='vmax',
                           )
        self.vmin = RParam(info="Bus voltage lower limit",
                           name='vmin', tex_name=r'v_{min}', unit='p.u.',
                           model='Bus', src='vmin', )
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


class DOPFBase(DCOPFBase):
    """
    Base class for DOPF.

    Overload the ``solve``, ``unpack``, and ``run`` methods.
    """

    def __init__(self, system, config):
        DCOPFBase.__init__(self, system, config)


class LDOPFModel(DOPFBase):
    """
    Linearzied distribution OPF model.
    """

    def __init__(self, system, config):
        DOPFBase.__init__(self, system, config)
        self.info = 'Linearzied distribution OPF'
        self.type = 'DED'
        # --- vars ---
        # --- generation ---
        self.pg = Var(info='active power generation (system base)',
                      name='pg', tex_name=r'p_{g}', unit='p.u.',
                      model='StaticGen', src='p',
                      lb=self.pmin, ub=self.pmax, )
        self.qg = Var(info='active power generation (system base)',
                      name='qg', tex_name=r'q_{g}', unit='p.u.',
                      model='StaticGen', src='q',
                      lb=self.qmin, ub=self.qmax, )
        # --- bus voltage and power injection ---
        self.vsq = Var(info='square of Bus voltage',
                       name='vsq', tex_name=r'v^{2}', unit='p.u.',
                       model='Bus',)
        self.pn = Var(info='Bus active power',
                      name='pn', tex_name=r'p_{n}', unit='p.u.',
                      model='Bus',)
        self.qn = Var(info='Bus reactive power',
                      name='qn', tex_name=r'q_{n}', unit='p.u.',
                      model='Bus',)

        # --- line flow ---
        self.pl = Var(info='line active power',
                      name='pl', tex_name=r'p_{l}', unit='p.u.',
                      model='Line',)
        self.ql = Var(info='line reactive power',
                      name='ql', tex_name=r'q_{l}', unit='p.u.',
                      model='Line',)

        self.isq = Var(info='square of line current',
                       name='isq', tex_name=r'i^{2}', unit='p.u.',
                       model='Line',)

        # --- constraints ---
        # --- node injection ---
        self.CftT = NumOp(u=self.Cft, fun=np.transpose,

                          name='CftT', tex_name=r'C_{ft}^{T}',

                          info='transpose of connection matrix',)
        self.pinj = Constraint(name='pinj',
                               info='node active power injection',
                               e_str='CftT@(pl - r * isq ) - pd - pn',
                               type='eq',)
        # TODO: is the e_str correct? Not sure if the sign is negative
        self.qinj = Constraint(name='qinj',
                               info='node reactive power injection',
                               e_str='CftT@(ql - x * isq ) - qd - qn',
                               type='eq',)
        # --- bus voltage ---
        self.vu = Constraint(name='vu',
                             info='Voltage upper limit',
                             e_str='vsq - vmax**2',
                             type='uq',)
        self.vl = Constraint(name='vl',
                             info='Voltage lower limit',
                             e_str='-vsq + vmin**2',
                             type='uq',)

        # --- branch voltage drop ---
        self.lvd = Constraint(name='lvd',
                              info='line voltage drop',
                              e_str='Cft@vsq - 2*(r * pl + x * ql) + (r**2 + x**2) @ isq',
                              type='eq',)

        # --- branch current ---
        self.lc = Constraint(name='lc',
                             info='line current using SOCP',
                             e_str='norm(vstack([2*(pl), 2*(ql), isq - Cft@vsq]), 2) - isq - Cft@vsq',
                             type='uq',)
        self.lub = Constraint(name='lub',
                              info='line limits upper bound',
                              e_str='isq - rate_a**2',
                              type='uq',)
        self.llb = Constraint(name='llb',
                              info='line limits lower bound',
                              e_str='-isq + rate_a**2',
                              type='uq',)

        # --- generation ---
        self.pgb = Constraint(name='pgb',
                              info='active power generation',
                              e_str='Cg@pn - pg',
                              type='eq',)
        self.qgb = Constraint(name='qgb',
                              info='reactive power generation',
                              e_str='Cg@qn - qg',
                              type='eq',)

        # --- objective ---
        self.obj = Objective(name='tc',
                             info='total cost [experiment]',
                             unit='$',
                             e_str='sum(c2 * pg**2 + c1 * pg + ug * c0)',
                             sense='min',)


class LDOPF(DOPFData, LDOPFModel):
    """
    Linearzied distribution OPF, where power loss are ignored.
    """

    def __init__(self, system, config):
        DOPFData.__init__(self)
        LDOPFModel.__init__(self, system, config)
