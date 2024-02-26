from collections import OrderedDict

from andes.core.param import NumParam, ExtParam
from andes.models.static.pv import PVData
from andes.models.static.slack import SlackData

from ams.core.model import Model
from ams.core.var import Algeb


class GenParam:
    """
    Additional parameters for static generators.
    """

    def __init__(self) -> None:
        self.ctrl = NumParam(default=1,
                             info="generator controllability",
                             tex_name=r'c_{trl}',)
        self.Pc1 = NumParam(default=0.0,
                            info="lower real power output of PQ capability curve",
                            tex_name=r'P_{c1}',
                            unit='p.u.')
        self.Pc2 = NumParam(default=0.0,
                            info="upper real power output of PQ capability curve",
                            tex_name=r'P_{c2}',
                            unit='p.u.')
        self.Qc1min = NumParam(default=0.0,
                               info="minimum reactive power output at Pc1",
                               tex_name=r'Q_{c1min}',
                               unit='p.u.')
        self.Qc1max = NumParam(default=0.0,
                               info="maximum reactive power output at Pc1",
                               tex_name=r'Q_{c1max}',
                               unit='p.u.')
        self.Qc2min = NumParam(default=0.0,
                               info="minimum reactive power output at Pc2",
                               tex_name=r'Q_{c2min}',
                               unit='p.u.')
        self.Qc2max = NumParam(default=0.0,
                               info="maximum reactive power output at Pc2",
                               tex_name=r'Q_{c2max}',
                               unit='p.u.')
        self.Ragc = NumParam(default=999.0,
                             info="ramp rate for load following/AGC",
                             tex_name=r'R_{agc}',
                             unit='p.u./h')
        self.R10 = NumParam(default=999.0,
                            info="ramp rate for 10 minute reserves",
                            tex_name=r'R_{10}',
                            unit='p.u./h')
        self.R30 = NumParam(default=999.0,
                            info="30 minute ramp rate",
                            tex_name=r'R_{30}',
                            unit='p.u./h')
        self.Rq = NumParam(default=999.0,
                           info="ramp rate for reactive power (2 sec timescale)",
                           tex_name=r'R_{q}',
                           unit='p.u./h')
        self.apf = NumParam(default=0.0,
                            info="area participation factor",
                            tex_name=r'a_{pf}')
        self.pg0 = NumParam(default=0.0,
                            info='active power start point (system base)',
                            tex_name=r'p_{g0}',
                            unit='p.u.',
                            )
        self.td1 = NumParam(default=0,
                            info='minimum ON duration',
                            tex_name=r't_{d1}',
                            unit='h',
                            )
        self.td2 = NumParam(default=0,
                            info='minimum OFF duration',
                            tex_name=r't_{d2}',
                            unit='h',
                            )


class PVModel(Model):
    """
    PV generator model (power flow) with q limit and PV-PQ conversion.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
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

        self.zone = ExtParam(model='Bus', src='zone', indexer=self.bus, export=False,
                             info='Retrieved zone idx', vtype=str, default=None,
                             )

        self.ud = Algeb(info='connection status decision',
                        unit='bool',
                        tex_name=r'u_d',
                        name='ud',
                        )
        self.p = Algeb(info='active power generation',
                       unit='p.u.',
                       tex_name='p',
                       name='p',
                       )
        self.q = Algeb(info='reactive power generation',
                       unit='p.u.',
                       tex_name='q',
                       name='q',
                       )


class PV(PVData, GenParam, PVModel):
    """
    PV generator model.

    TODO: implement type conversion in config
    """

    def __init__(self, system, config):
        PVData.__init__(self)
        GenParam.__init__(self)
        PVModel.__init__(self, system, config)


class Slack(SlackData, GenParam, PVModel):
    """
    Slack generator model.
    """

    def __init__(self, system=None, config=None):
        SlackData.__init__(self)
        GenParam.__init__(self)
        PVModel.__init__(self, system, config)
