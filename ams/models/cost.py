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
                             tex_name=r't_{ype}',
                             vrange=(1, 2),
                             )
        self.csu = NumParam(default=0,
                            info='startup cost in US dollars',
                            power=False,
                            tex_name=r'c_{su}',
                            unit='$',
                            )
        self.csd = NumParam(default=0,
                            info='shutdown cost in US dollars',
                            power=False,
                            tex_name=r'c_{sd}',
                            unit='$',
                            )
        self.c2 = NumParam(default=0,
                           info='coefficient 2',
                           power=False,
                           tex_name=r'c_{2}',
                           unit=r'$/(p.u.*h)^2',
                           )
        self.c1 = NumParam(default=0,
                           info='coefficient 1',
                           power=False,
                           tex_name=r'c_{1}',
                           unit=r'$/p.u.*h',
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


class SFRCostData(ModelData):
    def __init__(self):
        super().__init__()
        self.gen = IdxParam(info="static generator index",
                            model='StaticGen',
                            mandatory=True,
                            )
        self.cru = NumParam(default=0,
                            info='coefficient for RegUp reserve',
                            power=False,
                            tex_name=r'c_{r}',
                            unit=r'$/p.u.*h',
                            )
        self.crd = NumParam(default=0,
                            info='coefficient for RegDn reserve',
                            power=False,
                            tex_name=r'c_{r}',
                            unit=r'$/p.u.*h',
                            )


class SFRCost(SFRCostData, Model):
    """
    Linear SFR cost model.
    """

    def __init__(self, system, config):
        SFRCostData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Cost'


class REGCV1CostData(ModelData):
    def __init__(self):
        super().__init__()
        self.reg = IdxParam(info="Renewable generator idx",
                            model='RenGen',
                            mandatory=True,
                            )
        self.cm = NumParam(default=0,
                           info='cost for emulated inertia (M)',
                           power=False,
                           tex_name=r'c_{r}',
                           unit=r'$/s',
                           )
        self.cd = NumParam(default=0,
                           info='cost for emulated damping (D)',
                           power=False,
                           tex_name=r'c_{r}',
                           unit=r'$/p.u.',
                           )


class REGCV1Cost(REGCV1CostData, Model):
    """
    Linear cost model for :ref:`REGCV1` emulated inertia (M) and damping (D).
    """

    def __init__(self, system, config):
        REGCV1CostData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Cost'
