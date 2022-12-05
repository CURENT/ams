"""
AC transmission line and two-winding transformer line.
"""

from ams.core import (ModelData, IdxParam, NumParam, ExtParam, DataParam)


class LineData(ModelData):
    """
    Line data.
    """

    def __init__(self):
        super().__init__()

        self.bus1 = IdxParam(model='Bus', info="idx of from bus")
        self.bus2 = IdxParam(model='Bus', info="idx of to bus")

        self.Sn = NumParam(default=100.0,
                           info="Power rating",
                           non_zero=True,
                           tex_name=r'S_n',
                           unit='MW',
                           )
        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f',
                           unit='Hz',
                           )
        self.Vn1 = NumParam(default=110.0,
                            info="AC voltage rating",
                            non_zero=True,
                            tex_name=r'V_{n1}',
                            unit='kV',
                            )
        self.Vn2 = NumParam(default=110.0,
                            info="rated voltage of bus2",
                            non_zero=True,
                            tex_name=r'V_{n2}',
                            unit='kV',
                            )

        self.r = NumParam(default=1e-8,
                          info="line resistance",
                          tex_name='r',
                          z=True,
                          unit='p.u.',
                          )
        self.x = NumParam(default=1e-8,
                          info="line reactance",
                          tex_name='x',
                          z=True,
                          unit='p.u.',
                          non_zero=True,
                          )
        self.b = NumParam(default=0.0,
                          info="shared shunt susceptance",
                          y=True,
                          unit='p.u.',
                          )
        self.g = NumParam(default=0.0,
                          info="shared shunt conductance",
                          y=True,
                          unit='p.u.',
                          )
        self.b1 = NumParam(default=0.0,
                           info="from-side susceptance",
                           y=True,
                           tex_name='b_1',
                           unit='p.u.',
                           )
        self.g1 = NumParam(default=0.0,
                           info="from-side conductance",
                           y=True,
                           tex_name='g_1',
                           unit='p.u.',
                           )
        self.b2 = NumParam(default=0.0,
                           info="to-side susceptance",
                           y=True,
                           tex_name='b_2',
                           unit='p.u.',
                           )
        self.g2 = NumParam(default=0.0,
                           info="to-side conductance",
                           y=True,
                           tex_name='g_2',
                           unit='p.u.',
                           )

        self.trans = NumParam(default=0,
                              info="transformer branch flag",
                              unit='bool',
                              )
        self.tap = NumParam(default=1.0,
                            info="transformer branch tap ratio",
                            tex_name='t_{ap}',
                            non_negative=True,
                            unit='float',
                            )
        self.phi = NumParam(default=0.0,
                            info="transformer branch phase shift in rad",
                            tex_name=r'\phi',
                            unit='radian',
                            )

        self.rate_a = NumParam(default=0.0,
                               info="long-term flow limit (placeholder)",
                               tex_name='R_{ATEA}',
                               unit='MVA',
                               )

        self.rate_b = NumParam(default=0.0,
                               info="short-term flow limit (placeholder)",
                               tex_name='R_{ATEB}',
                               unit='MVA',
                               )

        self.rate_c = NumParam(default=0.0,
                               info="emergency flow limit (placeholder)",
                               tex_name='R_{ATEC}',
                               unit='MVA',
                               )

        self.owner = IdxParam(model='Owner', info="owner code")

        self.xcoord = DataParam(info="x coordinates")
        self.ycoord = DataParam(info="y coordinates")
