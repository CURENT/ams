"""
Distributional optimal power flow (DOPF) routine.
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
    Data for distribution network OPF.
    """

    def __init__(self):
        DCOPFData.__init__(self)


class DOPFBase(DCOPFBase):
    """
    Base class for DOPF.

    Overload the ``solve``, ``unpack``, and ``run`` methods.
    """

    def __init__(self, system, config):
        DCOPFBase.__init__(self, system, config)
        self.pd = RParam(info='active power demand connected to Bus (system base)',
                         name='pd',
                         tex_name=r'p_{d}',
                         unit='p.u.',
                         )
        self.qd = RParam(info='reactive power demand connected to Bus (system base)',
                         name='qd',
                         tex_name=r'q_{d}',
                         unit='p.u.',
                         )
        self.vmax = RParam(info="Bus voltage upper limit",
                           name='vmax',
                           tex_name=r'v_{max}',
                           unit='p.u.',
                           model='Bus',
                           src='vmax',
                           )
        self.vmin = RParam(info="Bus voltage lower limit",
                           name='vmin',
                           tex_name=r'v_{min}',
                           unit='p.u.',
                           model='Bus',
                           src='vmin',
                           )
        self.r = RParam(info='line resistance',
                        name='r',
                        tex_name='r',
                        unit='p.u.',
                        model='Line',
                        )
        self.x = RParam(info='line reactance',
                        name='x',
                        tex_name='x',
                        unit='p.u.',
                        model='Line',
                        )
        self.qmax = RParam(info='generator maximum reactive power (system base)',
                           name='qmax',
                           tex_name=r'q_{max}',
                           unit='p.u.',
                           model='StaticGen',
                           )
        self.qmin = RParam(info='generator minimum reactive power (system base)',
                           name='qmin',
                           tex_name=r'q_{min}',
                           unit='p.u.',
                           model='StaticGen',
                           )
        self.Cft = RParam(info='connection matrix for line to bus',
                          name='Cft',
                          tex_name=r'C_{ft}',
                          )
        self.Cg = RParam(info='connection matrix for gen to bus',
                         name='Cg',
                         tex_name=r'C_{g}',
                         )


class RDOPFModel(RDOPFBase):
    """
    DOPF model.
    """

    def __init__(self, system, config):
        RDOPFBase.__init__(self, system, config)
        self.info = 'Radial distribution network OPF'
        self.type = 'DED'
        # --- vars ---
        # --- generation ---
        self.pg = Var(info='active power generation (system base)',
                      unit='p.u.',
                      name='pg',
                      src='p',
                      tex_name=r'p_{g}',
                      model='StaticGen',
                      lb=self.pmin,
                      ub=self.pmax,
                      )
        self.qg = Var(info='active power generation (system base)',
                      unit='p.u.',
                      name='qg',
                      src='q',
                      tex_name=r'q_{g}',
                      model='StaticGen',
                      lb=self.qmin,
                      ub=self.qmax,
                      )
        # --- bus voltage and power injection ---
        self.vsq = Var(info='square of Bus voltage',
                       unit='p.u.',
                       name='vsq',
                       tex_name=r'v^{2}',
                       model='Bus',
                       )
        self.pn = Var(info='Bus active power',
                      unit='p.u.',
                      name='pn',
                      tex_name=r'p_{n}',
                      model='Bus',
                      )
        self.qn = Var(info='Bus reactive power',
                      unit='p.u.',
                      name='qn',
                      tex_name=r'q_{n}',
                      model='Bus',
                      )

        # --- line flow ---
        self.pl = Var(info='line active power',
                      unit='p.u.',
                      name='pl',
                      tex_name=r'p_{l}',
                      model='Line',
                      )
        self.ql = Var(info='line reactive power',
                      unit='p.u.',
                      name='ql',
                      tex_name=r'q_{l}',
                      model='Line',
                      )
        self.isq = Var(info='square of line current',
                       unit='p.u.',
                       name='isq',
                       tex_name=r'i^{2}',
                       model='Line',
                       )

        # --- constraints ---
        # --- node injection ---
        self.CftT = NumOp(u=self.Cft,
                          fun=np.transpose,
                          name='CftT',
                          tex_name=r'C_{ft}^{T}',
                          info='transpose of connection matrix',)
        self.pinj = Constraint(name='pinj',
                               info='node active power injection',
                               e_str='CftT@(pl - r * isq ) - pd - pn',
                               type='eq',
                               )
        # TODO: is the e_str correct? Not sure if the sign is negative
        self.qinj = Constraint(name='qinj',
                               info='node reactive power injection',
                               e_str='CftT@(ql - x * isq ) - qd - qn',
                               type='eq',
                               )
        # --- bus voltage ---
        self.vu = Constraint(name='vu',
                             info='Voltage upper limit',
                             e_str='vsq - vmax**2',
                             type='uq',
                             )
        self.vl = Constraint(name='vl',
                             info='Voltage lower limit',
                             e_str='-vsq + vmin**2',
                             type='uq',
                             )

        # --- branch voltage drop ---
        self.lvd = Constraint(name='lvd',
                              info='line voltage drop',
                              e_str='Cft@vsq - 2*(r * pl + x * ql) + (r**2 + x**2) @ isq',
                              type='eq',
                              )

        # --- branch current ---
        self.lc = Constraint(name='lc',
                             info='line current using SOCP',
                             e_str='norm(vstack([2*(pl), 2*(ql), isq - Cft@vsq]), 2) - isq - Cft@vsq',
                             type='uq',
                             )
        self.lub = Constraint(name='lub',
                              info='line limits upper bound',
                              e_str='isq - rate_a**2',
                              type='uq',
                              )
        self.llb = Constraint(name='llb',
                              info='line limits lower bound',
                              e_str='-isq + rate_a**2',
                              type='uq',
                              )

        # --- generation ---
        self.pgb = Constraint(name='pgb',
                              info='active power generation',
                              e_str='Cg@pn - pg',
                              type='eq',
                              )
        self.qgb = Constraint(name='qgb',
                              info='reactive power generation',
                              e_str='Cg@qn - qg',
                              type='eq',
                              )

        # --- objective ---
        self.obj = Objective(name='tc',
                             info='total cost [experiment]',
                             unit='$',
                             e_str='sum(c2 * pg**2 + c1 * pg + ug * c0)',
                             sense='min',)


class RDOPF(RDOPFData, RDOPFModel):
    """
    Radial distributiona network OPF (RDOPF) routine.
    """

    def __init__(self, system, config):
        RDOPFData.__init__(self)
        RDOPFModel.__init__(self, system, config)
