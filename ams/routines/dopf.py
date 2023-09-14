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
        self.Cft = RParam(info='connection matrix for Line and Bus',
                          name='Cft', tex_name=r'C_{ft}',)


class LDOPFModel(DCOPFModel):
    """
    Linearzied distribution OPF model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)
        self.info = 'Linearzied distribution OPF'
        self.type = 'DED'
        # --- vars ---
        self.qg = Var(info='Gen reactive power (system base)',
                      name='qg', tex_name=r'q_{g}', unit='p.u.',
                      model='StaticGen', src='q',
                      lb=self.qmin, ub=self.qmax,)
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

        self.pl = Var(info='line active power',
                      name='pl', tex_name=r'p_{l}', unit='p.u.',
                      model='Line',)
        self.ql = Var(info='line reactive power',
                      name='ql', tex_name=r'q_{l}', unit='p.u.',
                      model='Line',)

        # --- constraints ---
        self.CftT = NumOp(u=self.Cft,
                          fun=np.transpose,
                          name='CftT',
                          tex_name=r'C_{ft}^{T}',
                          info='transpose of connection matrix',)
        self.pinj.e_str = 'CftT@pl - pd - pn'
        self.qinj = Constraint(name='qinj',
                               info='node reactive power injection',
                               e_str='CftT@ql - qd - qn',
                               type='eq',)

        self.qb = Constraint(name='qb', info='reactive power balance',
                             e_str='sum(qd) - sum(qg)',
                             type='eq',)

        self.lvd = Constraint(name='lvd',
                              info='line voltage drop',
                              e_str='Cft@vsq - (r * pl + x * ql)',
                              type='eq',)

        # --- objective ---
        # NOTE: no need to revise objective function

    def unpack(self, **kwargs):
        super().unpack(**kwargs)
        vBus = np.sqrt(self.vsq.v)
        self.system.Bus.set(src='v', attr='v', value=vBus, idx=self.vsq.get_idx())


class LDOPF(DOPFData, LDOPFModel):
    """
    Linearzied distribution OPF, where power loss are ignored.

    Reference:

    [1] L. Bai, J. Wang, C. Wang, C. Chen, and F. Li, “Distribution Locational Marginal Pricing (DLMP)
    for Congestion Management and Voltage Support,” IEEE Trans. Power Syst., vol. 33, no. 4,
    pp. 4061–4073, Jul. 2018, doi: 10.1109/TPWRS.2017.2767632.
    """

    def __init__(self, system, config):
        DOPFData.__init__(self)
        LDOPFModel.__init__(self, system, config)


class DOPF2Data(DOPFData):
    """
    Data for DOPF with PFRCost for virtual inertia and damping from REGCV1.
    """

    def __init__(self):
        DOPFData.__init__(self)
        self.cm = RParam(info='RegUp reserve coefficient',
                         name='cm', src='cm',
                         tex_name=r'c_{m}', unit=r'$/s',
                         model='REGCV1Cost',
                         indexer='reg', imodel='REGCV1')
        self.cd = RParam(info='RegDown reserve coefficient',
                         name='cd', src='cd',
                         tex_name=r'c_{d}', unit=r'$/(p.u.)',
                         model='REGCV1Cost',
                         indexer='reg', imodel='REGCV1',)


class LDOPF2Model(LDOPFModel):
    """
    Linearzied distribution OPF model with VSG.
    """

    def __init__(self, system, config):
        LDOPFModel.__init__(self, system, config)
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


class LDOPF2(DOPF2Data, LDOPF2Model):
    """
    Linearzied distribution OPF with variables for virtual inertia and damping from from REGCV1,
    where power loss are ignored.

    Reference:

    [1] L. Bai, J. Wang, C. Wang, C. Chen, and F. Li, “Distribution Locational Marginal Pricing (DLMP)
    for Congestion Management and Voltage Support,” IEEE Trans. Power Syst., vol. 33, no. 4,
    pp. 4061–4073, Jul. 2018, doi: 10.1109/TPWRS.2017.2767632.
    """

    def __init__(self, system, config):
        DOPF2Data.__init__(self)
        LDOPF2Model.__init__(self, system, config)
