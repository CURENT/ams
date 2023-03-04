"""
Cost model.
"""

from andes.core import (ModelData, IdxParam, NumParam)

from ams.core.model import Model


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


# TODO: double check the picewise linear part documentation
class GCost(GCostData, Model):
    """
    Generator cost model, similar to MATPOWER ``gencost`` format.

    ``type`` is the cost model type. 1 for piecewise linear, 2 for polynomial.

    In piecewise linear cost model, cost function f(p) is defined by a set of points:
    (p0, c0), (p1, c1), (p2, c2), where p0 < p1 < p2.

    In quadratic cost model, cost function f(p) is defined by a set of coefficients:
    f(p) = c2 * p^2 + c1 * p + c0.
    """

    def __init__(self, system, config):
        GCostData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Cost'
