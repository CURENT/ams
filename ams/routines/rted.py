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
        self.cr = RParam(info='Reserve cost coefficient',
                         name='cr',
                         tex_name=r'c_{r}',
                         unit=r'$/(p.u.^2)',
                         owner_name='RCost',
                         )
        # --- generator output ---
        self.pg0 = RParam(info='generator active power start point',
                          name='pg0',
                          src='pg0',
                          tex_name=r'p_{g0}',
                          unit='p.u.',
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
        self.pr = Var(info='active power reserve',
                      unit='p.u.',
                      name='pr',
                      tex_name=r'p_{r}',
                      owner_name='StaticGen',
                      )
        # --- constraints ---
        # TODO: ramp constraint

        # --- objective ---
        # TODO: add reserve cost
        self.obj = Objective(name='tc',
                             info='total generation cost and reserve',
                             e_str='sum(c2 * pg**2 + c1 * pg + c0 + cr * pr)',
                             sense='min',)


class RTED(RTEDData, RTEDModel):
    """
    DCOPF dispatch routine.
    """

    def __init__(self, system, config):
        RTEDData.__init__(self)
        RTEDModel.__init__(self, system, config)
