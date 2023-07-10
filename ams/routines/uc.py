"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.routines.ed import EDData, EDModel

from ams.opt.omodel import Var, Constraint, Objective

logger = logging.getLogger(__name__)


class UCData(EDData):
    """
    UC data.
    """

    def __init__(self):
        EDData.__init__(self)
        self.csu = RParam(info='startup cost',
                          name='csu',
                          src='csu',
                          tex_name=r'c_{su}',
                          unit='$',
                          model='GCost',
                          )
        self.csd = RParam(info='shutdown cost',
                          name='csd',
                          src='csd',
                          tex_name=r'c_{sd}',
                          unit='$',
                          model='GCost',
                          )


class UCModel(EDModel):
    """
    UC model.
    """

    def __init__(self, system, config):
        EDModel.__init__(self, system, config)
        self.info = 'unit commitment'
        self.type = 'DCUC'
        # --- vars ---
        self.ugd = Var(info='gen on decision',
                       name='ugd',
                       horizon=self.horizon,
                       tex_name=r'u_{g,d}',
                       model='StaticGen',
                       bool=True,
                       src='u',
                       )
        # TODO: add variable "off"

        # TODO: add variable "reserve"
        # NOTE: spinning reserve or non-spinning reserve?
        # NOTE: is spinning reserve and AGC reserve the same? seems True
        # --- constraints ---
        self.pb.e_str = 'Rpd - Spg @ multiply(ugd, pg)'  # power balance
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
        # self.obj.e_str = 'sum(ugd * c2 * pg**2 + ugd * c1 * pg + c0 * ugd + csu * ugd + csd * (1 - ud))'
        # self.obj.e_str = 'sum(c2 * multiply(ugd, pg**2) +  c1 * multiply(ugd * pg) + c0 * ugd + csu * ugd)'  # DEBUG only
        # self.obj.e_str = 'sum(c2 * multiply(ugd, pg**2))'  # DEBUG only
        # self.obj = Objective(name='tc',
        #                      info='total generation and reserve cost',
        #                      e_str='sum(pg**2 * ug + c1 * pg * ug+ c0 * ug + csu * ug + csd * (1 - ug))',
        #                      sense='min',
        #                      )


class UC(UCData, UCModel):
    """
    DC-based unit commitment (UC).

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
