"""
Real-time economic dispatch.
"""
import numpy as np
from ams.core.param import RParam
from ams.routines.dcopf import DCOPFData, DCOPFModel

from ams.opt.omodel import Var, Constraint, Objective


class RTEDData(DCOPFData):
    """
    RTED parameters and variables.
    """

    def __init__(self):
        DCOPFData.__init__(self)
        # 1. reserve
        # 1.1. reserve cost
        self.cru = RParam(info='RegUp reserve coefficient',
                          name='cru',
                          src='cru',
                          tex_name=r'c_{r,u}',
                          unit=r'$/(p.u.^2)',
                          owner_name='RCost',
                          )
        self.crd = RParam(info='RegDown reserve coefficient',
                          name='crd',
                          src='crd',
                          tex_name=r'c_{r,d}',
                          unit=r'$/(p.u.^2)',
                          owner_name='RCost',
                          )
        # 1.2. reserve requirement
        self.du = RParam(info='RegUp reserve requirement',
                         name='du',
                         src='du',
                         tex_name=r'd_{u}',
                         unit='p.u.',
                         owner_name='AGCR',
                         )
        self.dd = RParam(info='RegDown reserve requirement',
                         name='dd',
                         src='dd',
                         tex_name=r'd_{d}',
                         unit='p.u.',
                         owner_name='AGCR',
                         )
        # FIXME: not generalized
        all_zone = np.array(self.system.Zone.idx.v)
        bus_idx = self.system.StaticGen.get(src='bus', attr='v',
                                            idx=self.pg.get_idx())
        zone_idx = self.system.Bus.get(src='zone', attr='v', idx=bus_idx)
        zone_list = np.array(len(all_zone) * [zone_idx])
        bool_array = (zone_list == all_zone[:, np.newaxis])
        prus_v = np.array(bool_array, dtype=int)
        self.prus = RParam(info='coefficient vector for RegUp reserve',
                           name='prus',
                           tex_name=r'p_{r,u,s}',
                           owner_name='StaticGen',
                           v=prus_v,
                           )
        # 1.3 reserve ramp rate
        # FIXME: seems not used
        self.Ragc = RParam(info='AGC ramp rate',
                           name='Ragc',
                           src='Ragc',
                           tex_name=r'R_{agc}',
                           unit='p.u./min',
                           owner_name='StaticGen',
                           )
        # 2. generator
        self.pg0 = RParam(info='generator active power start point',
                          name='pg0',
                          src='pg0',
                          tex_name=r'p_{g0}',
                          unit='p.u.',
                          owner_name='StaticGen',
                          )
        self.R10 = RParam(info='10-min ramp rate',
                          name='R10',
                          src='R10',
                          tex_name=r'R_{10}',
                          unit='p.u./min',
                          owner_name='StaticGen',
                          )


class RTEDModel(DCOPFModel):
    """
    RTED model.
    """

    def __init__(self, system, config):
        DCOPFModel.__init__(self, system, config)
        self.info = 'Real-time economic dispatch'
        self.type = 'DCED'
        # --- vars ---
        self.pru = Var(info='RegUp reserve',
                       unit='p.u.',
                       name='pru',
                       tex_name=r'p_{r,u}',
                       owner_name='StaticGen',
                       )
        self.prd = Var(info='RegDn reserve',
                       unit='p.u.',
                       name='prd',
                       tex_name=r'p_{r,d}',
                       owner_name='StaticGen',
                       )
        # --- constraints ---
        # self.rbu = Constraint(name='rbu',
        #                      info='RegUp reserve balance',
        #                      e_str='prus @ pru - du',
        #                      type='eq',
        #                      )
        # self.rbd = Constraint(name='rbu',
        #                      info='RegDn reserve balance',
        #                      e_str='prds @ prd - dd',
        #                      type='eq',
        #                      )
        self.rpru = Constraint(name='ru',
                               info='RegUp reserve ramp',
                               e_str='pg + pru - pmax',
                               type='uq',
                               )
        self.rprd = Constraint(name='rd',
                               info='RegDn reserve ramp',
                               e_str='-pg + prd - pmin',
                               type='uq',
                               )
        self.rpgu = Constraint(name='rampu',
                               info='GEN ramp up',
                               e_str='pg - pg0 - R10',
                               type='uq',
                               )
        self.rpgd = Constraint(name='rampd',
                               info='GEN ramp down',
                               e_str='-pg + pg0 - R10',
                               type='uq',
                               )
        # --- objective ---
        self.obj = Objective(name='tc',
                             info='total generation and reserve cost',
                             e_str='sum(c2 * pg**2 + c1 * pg + c0 + cru * pru + crd * prd)',
                             sense='min',
                             )


class RTED(RTEDData, RTEDModel):
    """
    DCOPF dispatch routine.
    """

    def __init__(self, system, config):
        RTEDData.__init__(self)
        RTEDModel.__init__(self, system, config)
