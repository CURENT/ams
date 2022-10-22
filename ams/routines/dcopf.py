"""Standard DCOPF"""
from ams.routines.base import BaseRT
from ams.core.param import NumParam, IdxParam
from ams.core.optz import Var, Obj, Constr


class dcopf(BaseRT):
    """DCOPF model"""

    def __init__(self) -> None:
        self.pg = Var()
        self.pb = Constr(expr='sum(pg) == sum(pd)',
                         tex_name='PowerBalance',
                         info='Power balance')
        self.obj = Obj(expr='pg * cost', sense='min')
