"""
OPF routines.
"""
from collections import OrderedDict
from ams.routines.base import BaseRoutine
from ams.utils import timer
import numpy as np
from scipy.optimize import linprog


# NOTE: in the future, there might be a ``OPFBase`` as refactorization

class DCOPF(BaseRoutine):
    """
    DC Optimal Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "DC Optimal Power Flow"
        self.rtn_models = OrderedDict([
            ('Bus', ['vmax', 'vmin']),
            ('Slack', ['pmax', 'pmin', 'qmax', 'qmin']),
            ('PV', ['pmax', 'pmin', 'qmax', 'qmin']),
        ])

    @timer
    def solve(self):
        """
        Overload solve function.
        """
        # res = linprog(self.c,
        #               A_ub=self.Aub, b_ub=self.bub,
        #               A_eq=self.Aeq, b_eq=self.beq,
        #               bounds=[self.lb, self.ub])
        return None

    def unpack(self, **kwargs):
        """
        Overload the unpack function.
        """
        return None

    def setup_om(self):
        # --- optimization modeling ---

        # --- debug ---
        # self.om.AddOVars(OAlgeb=self.pGen,
        #                  lb=self.system.Gen.pmin,
        #                  ub=self.system.Gen.pmax,)

        # --- decision variables ---
        self.om.AddOVars(OAlgeb=self.pPV,
                         lb=self.system.PV.pmin,
                         ub=self.system.PV.pmax,)
        self.om.AddOVars(OAlgeb=self.pSlack,
                         lb=self.system.Slack.pmin,
                         ub=self.system.Slack.pmax,)

        # --- constraints ---
        self.om.AddConstrs(name='pb',
                           n=self.system.Bus.n,
                           expr='Sum(pg) - Sum(p0PQ)',
                           type='eq',
                           info='power balance',
                           )
        self.om.AddConstrs(name='lub',
                           n=self.system.Line.n,
                           expr='GSF * (G - D) - rate_aLine',  # TODO; fix eqn
                           type='uq',
                           info='line limits upper bound',
                           )
        self.om.AddConstrs(name='llb',
                           n=self.system.Line.n,
                           expr='- GSF * (G - D) - rate_aLine',  # TODO; fix eqn
                           type='lq',
                           info='line limits lower bound',
                           )

        # --- objective ---
        # TODO: how to handle cost coefficients by generator?
        self.om.obj.set(expr='c2GCost * pg^2 + c1GCost * pg + c0GCost',
                        sense='min',)


class OPF(DCOPF):
    """
    AC Optimal Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system=system, config=config)
        self.info = "AC Optimal Power Flow"

    def setup_om(self):
        pass
