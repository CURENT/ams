"""
Cost model.
"""

from collections import OrderedDict

from andes.core import (ModelData, IdxParam, NumParam, Model,
                        ExtAlgeb, ExtService, ConstService, Limiter)


class GCostData(ModelData):
    def __init__(self):
        super().__init__()
        self.gen = IdxParam(info="static generator index",
                            model='StaticGen',
                            mandatory=True,
                            )
        self.type = NumParam(default=2,
                           info='Cost model type. 1 for piecewise linear, 2 for polynomial',
                           power=False,
                           tex_name=r'type',
                           vrange=(1, 2),
                           )
        self.startup = NumParam(default=0,
                           info='startup cost in US dollars',
                           power=False,
                           tex_name=r'c_{su}',
                           unit='USD',
                           )
        self.shutdown = NumParam(default=0,
                           info='shutdown cost in US dollars',
                           power=False,
                           tex_name=r'c_{sd}',
                           unit='USD',
                           )
        self.c2 = NumParam(default=0,
                           info='coefficient 2',
                           power=False,
                           tex_name=r'c_{2}',
                           unit=r'$/MW (MVar)',
                           )
        self.c1 = NumParam(default=0,
                           info='coefficient 1',
                           power=False,
                           tex_name=r'c_{1}',
                           unit=r'$/MW (MVar)',
                           )
        self.c0 = NumParam(default=0,
                           info='coefficient 0',
                           power=False,
                           tex_name=r'c_{0}',
                           unit=r'$',
                           )


# TODO: improve documentation
class GCost(GCostData, Model):
    """
    Generator cost model.
    """
    
    def __init__(self, system, config):
        GCostData.__init__(self)
        self.group = 'Cost'
        Model.__init__(self, system, config)
