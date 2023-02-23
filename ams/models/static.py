from collections import OrderedDict  # NOQA

from andes.models.static.pq import PQData  # NOQA
from andes.models.static.pv import PVData  # NOQA
from andes.models.static.slack import SlackData  # NOQA

from ams.core.model import Model


class PQ(PQData, Model):
    """
    PQ load model.

    TODO: implement type conversion in config
    """

    def __init__(self, system, config):
        PQData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'StaticLoad'
        self.config.add(OrderedDict((('pq2z', 1),
                                     ('p2p', 0.0),
                                     ('p2i', 0.0),
                                     ('p2z', 1.0),
                                     ('q2q', 0.0),
                                     ('q2i', 0.0),
                                     ('q2z', 1.0),
                                     )))
        self.config.add_extra("_help",
                              pq2z="pq2z conversion if out of voltage limits",
                              p2p="P constant power percentage for TDS. Must have (p2p+p2i+p2z)=1",
                              p2i="P constant current percentage",
                              p2z="P constant impedance percentage",
                              q2q="Q constant power percentage for TDS. Must have (q2q+q2i+q2z)=1",
                              q2i="Q constant current percentage",
                              q2z="Q constant impedance percentage",
                              )
        self.config.add_extra("_alt",
                              pq2z="(0, 1)",
                              p2p="float",
                              p2i="float",
                              p2z="float",
                              q2q="float",
                              q2i="float",
                              q2z="float",
                              )
        self.config.add_extra("_tex",
                              pq2z="z_{pq2z}",
                              p2p=r"\gamma_{p2p}",
                              p2i=r"\gamma_{p2i}",
                              p2z=r"\gamma_{p2z}",
                              q2q=r"\gamma_{q2q}",
                              q2i=r"\gamma_{q2i}",
                              q2z=r"\gamma_{q2z}",
                              )


class PV(PVData, Model):
    """
    PV generator model.

    TODO: implement type conversion in config
    """

    def __init__(self, system, config):
        PVData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'StaticGen'

        self.config.add(OrderedDict((('pv2pq', 0),
                                     ('npv2pq', 0),
                                     ('min_iter', 2),
                                     ('err_tol', 0.01),
                                     ('abs_violation', 1),
                                     )))
        self.config.add_extra("_help",
                              pv2pq="convert PV to PQ in PFlow at Q limits",
                              npv2pq="max. # of conversion each iteration, 0 - auto",
                              min_iter="iteration number starting from which to enable switching",
                              err_tol="iteration error below which to enable switching",
                              abs_violation='use absolute (1) or relative (0) limit violation',
                              )

        self.config.add_extra("_alt",
                              pv2pq=(0, 1),
                              npv2pq=">=0",
                              min_iter='int',
                              err_tol='float',
                              abs_violation=(0, 1),
                              )
        self.config.add_extra("_tex",
                              pv2pq="z_{pv2pq}",
                              npv2pq="n_{pv2pq}",
                              min_iter="sw_{iter}",
                              err_tol=r"\epsilon_{tol}"
                              )


class Slack(SlackData, Model):
    """
    Slack generator.
    """

    def __init__(self, system=None, config=None):
        SlackData.__init__(self)
        Model.__init__(self, system, config)

        self.config.add(OrderedDict((('av2pv', 0),
                                     )))
        self.config.add_extra("_help",
                              av2pv="convert Slack to PV in PFlow at P limits",
                              )
        self.config.add_extra("_alt",
                              av2pv=(0, 1),
                              )
        self.config.add_extra("_tex",
                              av2pv="z_{av2pv}",
                              )
