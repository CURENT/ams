"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import NumOp
from ams.routines.ed import EDData, EDModel

from ams.opt.omodel import Var, Constraint, Objective

logger = logging.getLogger(__name__)


class UCData(EDData):
    """
    UC data.
    """

    def __init__(self):
        EDData.__init__(self)
        self.ug0 = RParam(info='initial gen connection status',
                          name='ug0', tex_name=r'u_{g,0}',
                          model='StaticGen', src='u',)
        self.csu = RParam(info='startup cost',
                          name='csu', tex_name=r'c_{su}',
                          model='GCost', src='csu',
                          unit='$',)
        self.csd = RParam(info='shutdown cost',
                          name='csd', tex_name=r'c_{sd}',
                          model='GCost', src='csd',
                          unit='$',)
        self.td1 = RParam(info='minimum ON duration',
                          name='td1', tex_name=r't_{d1}',
                          model='StaticGen', src='td1',
                          unit='min',)
        self.td2 = RParam(info='minimum OFF duration',
                          name='td2', tex_name=r't_{d2}',
                          model='StaticGen', src='td2',
                          unit='min',)

        self.sd.info = 'zonal load factor for UC'
        self.sd.model = 'UCTSlot'

        self.timeslot.info = 'Time slot for multi-period UC'
        self.timeslot.model = 'UCTSlot'

        self.dt = RParam(info='UC interval',
                         name='dt', tex_name=r't{d}',
                         model='UCTSlot', src='dt',
                         unit='min',)

        self.dsr = RParam(info='spinning reserve requirement',
                          name='dsr', tex_name=r'd_{sr}',
                          model='SR', src='demand',
                          unit='%',)

        self.dnsr = RParam(info='non-spinning reserve requirement',
                           name='dnsr', tex_name=r'd_{nsr}',
                           model='NSR', src='demand',
                           unit='%',)


class UCModel(EDModel):
    """
    UC model.
    """

    def __init__(self, system, config):
        EDModel.__init__(self, system, config)
        self.info = 'unit commitment'
        self.type = 'DCUC'
        # --- vars ---
        self.ugd = Var(info='commitment decision',
                       horizon=self.timeslot,
                       name='ugd', tex_name=r'u_{g,d}',
                       model='StaticGen', src='u',
                       boolean=True,)
        self.vgd = Var(info='startup action',
                       horizon=self.timeslot,
                       name='vgd', tex_name=r'v_{g,d}',
                       model='StaticGen', src='u',
                       boolean=True,)
        self.wgd = Var(info='shutdown action',
                       horizon=self.timeslot,
                       name='wgd', tex_name=r'w_{g,d}',
                       model='StaticGen', src='u',
                       boolean=True,)

        self.zug = Var(info='Aux var for ugd',
                       horizon=self.timeslot,
                       name='zug', tex_name=r'z_{ug}',
                       model='StaticGen', pos=True,)

        self.psr = Var(info='spinning reserve (system base)',
                       unit='p.u.', name='psr', tex_name=r'p_{sr}',
                       model='StaticGen', nonneg=True,)
        self.pnsr = Var(info='non-spinning reserve (system base)',
                        unit='p.u.', name='pnsr', tex_name=r'p_{nsr}',
                        model='StaticGen', nonneg=True,)

        # NOTE: add action constraints by two parts
        self.actv = Constraint(name='actv', info='startup action',
                               e_str='ugd @ Mr - vgd[:, 1:]',
                               type='uq',)
        self.actv0 = Constraint(name='actv0', info='initial startup action',
                                e_str='ugd[:, 0] - ug0  - vgd[:, 0]',
                                type='uq',)
        self.actw = Constraint(name='actw', info='shutdown action',
                               e_str='-ugd @ Mr - wgd[:, 1:]',
                               type='uq',)
        self.actw0 = Constraint(name='actw0', info='initial shutdown action',
                                e_str='-ugd[:, 0] + ug0 - wgd[:, 0]',
                                type='uq',)

        # --- constraints ---
        self.pb.e_str = 'pds - Spg @ zug'  # power balance

        self.Mzug = NumOp(info='10 times of max of pmax as big M for zug',
                        name='Mzug', tex_name=r'M_{zug}',
                        u=self.pmax, fun=np.max,
                        rfun=np.dot, rargs=dict(b=10),)

        self.zuglb = Constraint(name='zuglb', info='zug lower bound',
                                type='uq', e_str='- zug + pg')
        self.zugub = Constraint(name='zugub', info='zug upper bound',
                                type='uq', e_str='zug - pg - Mzug[0] * (1 - ugd)')
        self.zugub2 = Constraint(name='zugub2', info='zug upper bound',
                                    type='uq', e_str='zug - Mzug[0] * ugd')

        # TODO: add reserve balance, 3% or 5%
        # TODO: constrs: minimum ON/OFF time for conventional units
        # TODO: add data prameters: minimum ON/OFF time for conventional units

        # TODO: constrs: unserved energy constraint

        # TODO: Energy storage?

        # self.rgu = Constraint(name='rgu',
        #                       info='ramp up limit of generator output',
        #                       e_str='pg - pg0 - R10',
        #                       type='uq',
        #                       )
        # self.rgd = Constraint(name='rgd',
        #                       info='ramp down limit of generator output',
        #                       e_str='-pg + pg0 - R10',
        #                       type='uq',
        #                       )
        # --- objective ---
        # NOTE: havn't adjust time duration
        gcost = 'sum(c2 * zug**2 + c1 * zug + c0 * ugd + csu * vgd + csd * wgd)'
        rcost = ''
        self.obj.e_str = gcost + rcost


class UC(UCData, UCModel):
    """
    DC-based unit commitment (UC), wherew.

    References
    ----------
    1. Huang, Y., Pardalos, P. M., & Zheng, Q. P. (2017). Electrical power unit commitment: deterministic and
    two-stage stochastic programming models and algorithms. Springer.

    2. D. A. Tejada-Arango, S. Lumbreras, P. Sánchez-Martín and A. Ramos, "Which Unit-Commitment Formulation
    is Best? A Comparison Framework," in IEEE Transactions on Power Systems, vol. 35, no. 4, pp. 2926-2936,
    July 2020, doi: 10.1109/TPWRS.2019.2962024.
    """

    def __init__(self, system, config):
        UCData.__init__(self)
        UCModel.__init__(self, system, config)
