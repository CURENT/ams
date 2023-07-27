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
    DOPF data.
    """

    def __init__(self):
        DCOPFData.__init__(self)


class DOPFBase(DCOPFBase):
    """
    Base class for DOPF dispatch model.

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
        self.Cft = RParam(info='connection matrix',
                          name='Cft',
                          tex_name=r'C_{ft}',
                          )


class DOPFModel(DOPFBase):
    """
    DOPF model.
    """

    def __init__(self, system, config):
        DOPFBase.__init__(self, system, config)
        self.info = 'Distributional Optimal Power Flow'
        self.type = 'DED'
        # --- vars ---
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

        # --- constraints ---
        # --- node injection ---
        self.CftT = NumOp(u=self.Cft,
                          fun=np.transpose,
                          name='CftT',
                          tex_name=r'C_{ft}^{T}',
                          info='transpose of connection matrix',
                          is_sparse=True,)
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
        # # --- branch voltage drop ---
        # self.lvd = Constraint(name='lvd',
        #                       info='branch voltage drop',
        #                       e_str='Cft',
        #                       type='eq',
        #                       )

        # --- objective ---
        # TODO: need a conversion from pn to pg
        self.obj = Objective(name='tc',
                             info='total cost [experiment]',
                             unit='$',
                             e_str='sum(pn)',
                             sense='min',)


class DOPF(DOPFData, DOPFModel):
    """
    Distributional optimal power flow (DOPF) routine.
    """

    def __init__(self, system, config):
        DOPFData.__init__(self)
        DOPFModel.__init__(self, system, config)
