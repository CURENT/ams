from collections import OrderedDict

from andes.core.param import NumParam, ExtParam
from andes.models.static.pq import PQData

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

        self.zone = ExtParam(model='Bus', src='zone', indexer=self.bus, export=False,
                             info='Retrieved zone idx', vtype=str, default=None,
                             )
        self.ctrl = NumParam(default=1,
                             info="load controllability",
                             tex_name=r'c_{trl}',)
