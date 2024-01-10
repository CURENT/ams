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


class SFRCost(ModelData, Model):
    """
    Linear SFR cost model.
    """

    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Cost'
        self.gen = IdxParam(info="static generator index",
                            model='StaticGen',
                            mandatory=True,)
        self.cru = NumParam(default=0,
                            tex_name=r'c_{r}', unit=r'$/(p.u.*h)',
                            info='cost for RegUp reserve',)
        self.crd = NumParam(default=0,
                            tex_name=r'c_{r}', unit=r'$/(p.u.*h)',
                            info='cost for RegDn reserve',)


class SRCost(ModelData, Model):
    """
    Linear spinning reserve cost model.
    """

    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.gen = IdxParam(info="static generator index",
                            model='StaticGen',
                            mandatory=True,)
        self.csr = NumParam(default=0,
                            tex_name=r'c_{sr}', unit=r'$/(p.u.*h)',
                            info='cost for spinning reserve',)


class NSRCost(ModelData, Model):
    """
    Linear non-spinning reserve cost model.
    """

    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.gen = IdxParam(info="static generator index",
                            model='StaticGen',
                            mandatory=True,)
        self.cnsr = NumParam(default=0,
                             tex_name=r'c_{nsr}', unit=r'$/(p.u.*h)',
                             info='cost for non-spinning reserve',)


class DCost(ModelData, Model):
    """
    Linear cost model for dispatchable loads.
    """
    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Cost'
        self.pq = IdxParam(info="static load index",
                           mandatory=True,)
        self.cdp = NumParam(default=999,
                            tex_name=r'c_{d,p}', unit=r'$/(p.u.*h)',
                            info='cost for unserve load penalty',)


class VSGCostData(ModelData):
    def __init__(self):
        super().__init__()
        self.reg = IdxParam(info="Renewable generator idx",
                            model='RenGen',
                            mandatory=True,
                            )
        self.cm = NumParam(default=0,
                           info='cost for emulated inertia (M)',
                           tex_name=r'c_{r}',
                           unit=r'$/s',
                           )
        self.cd = NumParam(default=0,
                           info='cost for emulated damping (D)',
                           tex_name=r'c_{r}',
                           unit=r'$/p.u.',
                           )


class VSGCost(VSGCostData, Model):
    """
    Linear cost model for VSG emulated inertia (M) and damping (D).
    """

    def __init__(self, system, config):
        VSGCostData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Cost'
