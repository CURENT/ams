"""
Real-time economic dispatch.
"""
from ams.core.param import RParam
from ams.routines.dcopf import DCOPFData, DCOPFModel

from ams.opt.omodel import Var, Constraint, Objective


class RTEDData(DCOPFData):
    """
    RTED parameters and variables.
    """

    def __init__(self):
        DCOPFData.__init__(self)
        # --- reserve cost ---
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
        # --- generator ---
        self.pg0 = RParam(info='generator active power start point',
                          name='pg0',
                          src='pg0',
                          tex_name=r'p_{g0}',
                          unit='p.u.',
                          owner_name='StaticGen',
                          )
        self.Ragc = RParam(info='AGC ramp rate',
                           name='Ragc',
                           src='Ragc',
                           tex_name=r'R_{agc}',
                           unit='p.u./min',
                           owner_name='StaticGen',
                           )
        self.R10 = RParam(info='10-min ramp rate',
                          name='R10',
                          src='R10',
                          tex_name=r'R_{10}',
                          unit='p.u./min',
                          owner_name='StaticGen',
                          )
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
        # self.rb = Constraint(name='rb',
        #                      info='reserve bound',
        #                      e_str='sum(pr) - ',
        #                      type='eq',
        #                      )
        self.ru = Constraint(name='ru',
                             info='RegUp reserve',
                             e_str='pg + pru - pmax',
                             type='uq',
                             )
        self.rd = Constraint(name='rd',
                             info='RegDn reserve',
                             e_str='-pg + prd - pmin',
                             type='uq',
                             )
        self.rampu = Constraint(name='rampu',
                                info='generator ramp up',
                                e_str='pg - pg0 - R10',
                                type='uq',
                                )
        self.rampd = Constraint(name='rampd',
                                info='generator ramp down',
                                e_str='-pg + pg0 - R10',
                                type='uq',
                                )
        # --- objective ---
        # TODO: add reserve cost
        self.obj = Objective(name='tc',
                             info='total generation cost and reserve',
                             e_str='sum(c2 * pg**2 + c1 * pg + c0 + cru * pru + crd * prd)',
                             sense='min',)


class RTED(RTEDData, RTEDModel):
    """
    DCOPF dispatch routine.
    """

    def __init__(self, system, config):
        RTEDData.__init__(self)
        RTEDModel.__init__(self, system, config)
