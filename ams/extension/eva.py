"""
EV Aggregator.
"""

from collections import OrderedDict

import scipy.stats as stats

from andes.core import Config
from andes.core.param import NumParam
from andes.core.model import ModelData
from andes.shared import pd

from ams.core.model import Model


class EVA(ModelData, Model):
    """
    State space modeling based EV aggregation model.

    In the EVA, each single EV is recorded as a device with its own parameters.
    The parameters are generated from given statistical distributions.

    Reference:
    [1] J. Wang et al., "Electric Vehicles Charging Time Constrained Deliverable Provision of Secondary
    Frequency Regulation," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2024.3356948.
    [2] M. Wang, Y. Mu, Q. Shi, H. Jia and F. Li, "Electric Vehicle Aggregator Modeling and Control for
    Frequency Regulation Considering Progressive State Recovery," in IEEE Transactions on Smart Grid,
    vol. 11, no. 5, pp. 4176-4189, Sept. 2020, doi: 10.1109/TSG.2020.2981843.
    """

    def __init__(self, N=10000, Ns=20, Tagc=4, SOCf=0.2, r=0.5,
                 seed=None,):
        """
        Initialize the EV aggregation model.

        Parameters
        ----------
        N: int, optional
            Number of related EVs, default is 10000.
        Ns : int, optional
            Number of SOC intervals, default is 20.
        Tagc : int, optional
            AGC time intervals in seconds, default is 4.
        SOCf : float, optional
            Force charge SOC level between 0 and 1, default is 0.2.
        r : float, optional
            Ratio of time range 1 to time range 2 between 0 and 1, default is 0.5.
        """
        # inherit attributes and methods from ANDES `ModelData` and AMS `Model`
        ModelData.__init__(self)
        Model.__init__(self, system=None, config=None)

        # manually set config as EVA is not processed by the system
        self.config = Config(self.__class__.__name__)
        self.config.add(OrderedDict((('ns', Ns),
                                     ('tagc', Tagc),
                                     ('socf', SOCf),
                                     ('r', r),
                                     ('socl', 0),
                                     ('socu', 1),
                                     ('seed', seed),
                                     )))
        self.config.add_extra("_help",
                              ns="SOC intervals",
                              tagc="AGC time intervals in seconds",
                              socf="Force charge SOC level",
                              r="ratio of time range 1 to time range 2",
                              socl="lowest SOC limit",
                              socu="highest SOC limit",
                              seed='seed (or None) for random number generator',
                              )
        self.config.add_extra("_tex",
                              ns='N_s',
                              tagc='T_{agc}',
                              socf='SOC_f',
                              r='r',
                              socl='SOC_{l}',
                              socu='SOC_{u}',
                              seed='seed',
                              )
        self.config.add_extra("_alt",
                              ns="int",
                              tagc="float",
                              socf="float",
                              r="float",
                              socl="float",
                              socu="float",
                              seed='int or None',
                              )

        # manually set attributes as EVA is not processed by the system
        self.n = int(N)
        self.idx.v = ['SEV_' + str(i+1) for i in range(self.n)]
        self.uid = {self.idx.v[i]: i for i in range(self.n)}

    def setup(self):
        """
        Setup the EV aggregation model.

        Populate itself with generated EV devices based on the given parameters.
        """
        # --- parameters ---
        self.namax = NumParam(default=0,
                              info='maximum number of action')
        self.ts = NumParam(default=0,
                           info='arrive time, in 24 hours')
        self.tf = NumParam(default=0,
                           info='departure time, in 24 hours')
        self.tt = NumParam(default=0,
                           info='Tolerance of increased charging time')
        self.soci = NumParam(default=0,
                             info='initial SOC')
        self.socd = NumParam(default=0,
                             info='demand SOC')
        self.Pc = NumParam(default=0,
                           info='rated charging power, in kW')
        self.Pd = NumParam(default=0,
                           info='rated discharging power, in kW')
        self.nc = NumParam(default=0,
                           info='charging efficiency')
        self.nd = NumParam(default=0,
                           info='discharging efficiency')
        self.Q = NumParam(default=0,
                          info='rated capacity')

        # --- initialization ---
        # NOTE: following definition comes from ref[2]
        # normal distribution parameters
        ndist = {'soci': {'mu': 0.3, 'var': 0.05, 'lb': 0.2, 'ub': 0.4},
                 'socd': {'mu': 0.8, 'var': 0.03, 'lb': 0.7, 'ub': 0.9},
                 'ts1': {'mu': -6.5, 'var': 3.4, 'lb': 0.0, 'ub': 5.5},
                 'ts2': {'mu': 17.5, 'var': 3.4, 'lb': 5.5, 'ub': 24.0},
                 'tf1': {'mu': 8.9, 'var': 3.4, 'lb': 0.0, 'ub': 20.9},
                 'tf2': {'mu': 32.9, 'var': 3.4, 'lb': 20.9, 'ub': 24.0},
                 'tt': {'mu': 0.5, 'var': 0.02, 'lb': 0, 'ub': 1}}
        # uniform distribution parameters
        udist = {'Pc': {'lb': 5.0, 'ub': 7.0},
                 'Pd': {'lb': 5.0, 'ub': 7.0},
                 'nc': {'lb': 0.88, 'ub': 0.95},
                 'nd': {'lb': 0.88, 'ub': 0.95},
                 'Q': {'lb': 20.0, 'ub': 30.0}}

        # --- set soci, socd ---
        self.soci.v = build_truncnorm(ndist['soci']['mu'], ndist['soci']['var'],
                                      ndist['soci']['lb'], ndist['soci']['ub'],
                                      self.n, self.config.seed)
        self.socd.v = build_truncnorm(ndist['socd']['mu'], ndist['socd']['var'],
                                      ndist['socd']['lb'], ndist['socd']['ub'],
                                      self.n, self.config.seed)

        # --- set ts, tf ---
        tdf = pd.DataFrame({
            col: build_truncnorm(ndist[col]['mu'], ndist[col]['var'],
                                 ndist[col]['lb'], ndist[col]['ub'],
                                 self.n, self.config.seed)
            for col in ['ts1', 'ts2', 'tf1', 'tf2']
        })

        nev_t1 = int(self.n * self.config.r)  # number of EVs in time range 1
        tp1 = tdf[['ts1', 'tf1']].sample(n=nev_t1, random_state=self.config.seed)
        tp2 = tdf[['ts2', 'tf2']].sample(n=self.n-nev_t1, random_state=self.config.seed)
        tp = pd.concat([tp1, tp2], axis=0).reset_index(drop=True).fillna(0)
        tp['ts'] = tp['ts1'] + tp['ts2']
        tp['tf'] = tp['tf1'] + tp['tf2']
        # Swap ts and tf if ts > tf
        check = tp['ts'] > tp['tf']
        tp.loc[check, ['ts', 'tf']] = tp.loc[check, ['tf', 'ts']].values

        self.ts.v = tp['ts'].values
        self.tf.v = tp['tf'].values

        # --- variables ---
        # self.soc0 = Algeb(info='previous SOC')
        # self.u0 = Algeb(info='previous online status')
        # self.na0 = Algeb(info='previous action number')
        # self.soc = Algeb(info='SOC')
        # self.u = Algeb(info='online status')
        # self.na = Algeb(info='action number')


def build_truncnorm(mu, var, lb, ub, n, seed):
    """
    Helper function to generate truncated normal distribution
    using scipy.stats.

    Parameters
    ----------
    mu : float
        Mean of the normal distribution.
    var : float
        Variance of the normal distribution.
    lb : float
        Lower bound of the truncated distribution.
    ub : float
        Upper bound of the truncated distribution.
    n : int
        Number of samples to generate.
    seed : int
        Random seed to use.
    """
    a = (lb - mu) / var
    b = (ub - mu) / var
    distribution = stats.truncnorm(a, b, loc=mu, scale=var)
    return distribution.rvs(n, random_state=seed)
