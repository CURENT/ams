"""
EV Aggregator module.

EVD is the generated datasets, and EVA is the aggregator model.

Reference:
[1] J. Wang et al., "Electric Vehicles Charging Time Constrained Deliverable Provision of Secondary
Frequency Regulation," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2024.3356948.
[2] M. Wang, Y. Mu, Q. Shi, H. Jia and F. Li, "Electric Vehicle Aggregator Modeling and Control for
Frequency Regulation Considering Progressive State Recovery," in IEEE Transactions on Smart Grid,
vol. 11, no. 5, pp. 4176-4189, Sept. 2020, doi: 10.1109/TSG.2020.2981843.
"""

import logging
import itertools
from collections import OrderedDict

import scipy.stats as stats

from andes.core.param import NumParam
from andes.core.model import ModelData
from andes.shared import np, pd
from andes.utils.misc import elapsed

from ams.core.model import Model
from ams.utils.paths import ams_root
from ams.core import Config

logger = logging.getLogger(__name__)


# NOTE: following definition comes from ref[2], except `tt` that is assumed by ref[1]
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


class EVD(ModelData, Model):
    """
    In the EVD, each single EV is recorded as a device with its own parameters.
    The parameters are generated from given statistical distributions.
    """

    def __init__(self, N=10000, Ns=20, Tagc=4, SOCf=0.2, r=0.5,
                 t=18, seed=None, name='EVA', A_csv=None):
        """
        Initialize the EV aggregation model.

        Parameters
        ----------
        N : int, optional
            Number of related EVs, default is 10000.
        Ns : int, optional
            Number of SOC intervals, default is 20.
        Tagc : int, optional
            AGC time intervals in seconds, default is 4.
        SOCf : float, optional
            Force charge SOC level between 0 and 1, default is 0.2.
        r : float, optional
            Ratio of time range 1 to time range 2 between 0 and 1, default is 0.5.
        seed : int or None, optional
            Seed for random number generator, default is None.
        t : int, optional
            Current time in 24 hours, default is 18.
        name : str, optional
            Name of the EVA, default is 'EVA'.
        A_csv : str, optional
            Path to the CSV file containing the state space matrix A, default is None.
        """
        # inherit attributes and methods from ANDES `ModelData` and AMS `Model`
        ModelData.__init__(self)
        Model.__init__(self, system=None, config=None)

        self.evdname = name

        # internal flags
        self.is_setup = False        # if EVA has been setup

        self.t = np.array(t, dtype=float)  # time in 24 hours
        self.eva = None  # EV Aggregator
        self.A_csv = A_csv  # path to the A matrix

        # manually set config as EVA is not processed by the system
        self.config = Config(self.__class__.__name__)
        self.config.add(OrderedDict((('n', int(N)),
                                     ('ns', Ns),
                                     ('tagc', Tagc),
                                     ('socf', SOCf),
                                     ('r', r),
                                     ('socl', 0),
                                     ('socu', 1),
                                     ('tf', self.t),
                                     ('prumax', 0),
                                     ('prdmax', 0),
                                     ('seed', seed),
                                     )))
        self.config.add_extra("_help",
                              n="Number of related EVs",
                              ns="SOC intervals",
                              tagc="AGC time intervals in seconds",
                              socf="Force charge SOC level",
                              r="ratio of time range 1 to time range 2",
                              socl="lowest SOC limit",
                              socu="highest SOC limit",
                              tf="EVA running end time in 24 hours",
                              prumax="maximum power of regulation up, in MW",
                              prdmax="maximum power of regulation down, in MW",
                              seed='seed (or None) for random number generator',
                              )
        self.config.add_extra("_tex",
                              n='N_{ev}',
                              ns='N_s',
                              tagc='T_{agc}',
                              socf='SOC_f',
                              r='r',
                              socl='SOC_{l}',
                              socu='SOC_{u}',
                              tf='T_f',
                              prumax='P_{ru,max}',
                              prdmax='P_{rd,max}',
                              seed='seed',
                              )
        self.config.add_extra("_alt",
                              n='int',
                              ns="int",
                              tagc="float",
                              socf="float",
                              r="float",
                              socl="float",
                              socu="float",
                              tf="float",
                              prumax="float",
                              prdmax="float",
                              seed='int or None',
                              )

        unit = self.config.socu / self.config.ns
        self.soc_intv = OrderedDict({
            i: (np.around(i * unit, 2), np.around((i + 1) * unit, 2))
            for i in range(self.config.ns)
        })

        # NOTE: the parameters and variables are declared here and populated in `setup()`
        # param `idx`, `name`, and `u` are already included in `ModelData`
        # variables here are actually declared as parameters for memory saving
        # because ams.core.var.Var has more overhead

        # --- parameters ---
        self.namax = NumParam(default=0,
                              info='maximum number of action')
        self.ts = NumParam(default=0, vrange=(0, 24),
                           info='arrive time, in 24 hours')
        self.tf = NumParam(default=0, vrange=(0, 24),
                           info='departure time, in 24 hours')
        self.tt = NumParam(default=0,
                           info='Tolerance of increased charging time, in hours')
        self.soci = NumParam(default=0,
                             info='initial SOC')
        self.socd = NumParam(default=0,
                             info='demand SOC')
        self.Pc = NumParam(default=0,
                           info='rated charging power, in kW')
        self.Pd = NumParam(default=0,
                           info='rated discharging power, in kW')
        self.nc = NumParam(default=0,
                           info='charging efficiency',
                           vrange=(0, 1))
        self.nd = NumParam(default=0,
                           info='discharging efficiency',
                           vrange=(0, 1))
        self.Q = NumParam(default=0,
                          info='rated capacity, in kWh')

        # --- variables ---
        self.soc0 = NumParam(default=0,
                             info='previous SOC')
        self.u0 = NumParam(default=0,
                           info='previous online status')
        self.na0 = NumParam(default=0,
                            info='previous action number')
        self.soc = NumParam(default=0,
                            info='SOC')
        self.na = NumParam(default=0,
                           info='action number')

    def setup(self, ndist=ndist, udist=udist):
        """
        Setup the EV aggregation model.

        Parameters
        ----------
        ndist : dict, optional
            Normal distribution parameters, default by built-in `ndist`.
        udist : dict, optional
            Uniform distribution parameters, default by built-in `udist`.

        Returns
        -------
        is_setup : bool
            If the setup is successful.
        """
        if self.is_setup:
            logger.warning(f'{self.evdname} aggregator has been setup, setup twice is not allowed.')
            return False

        t0, _ = elapsed()

        # manually set attributes as EVA is not processed by the system
        self.n = self.config.n
        self.idx.v = ['SEV_' + str(i+1) for i in range(self.config.n)]
        self.name.v = ['SEV ' + str(i+1) for i in range(self.config.n)]
        self.u.v = np.array(self.u.v, dtype=int)
        self.uid = {self.idx.v[i]: i for i in range(self.config.n)}

        # --- populate parameters' value ---
        # set `soci`, `socd`, `tt`
        self.soci.v = build_truncnorm(ndist['soci']['mu'], ndist['soci']['var'],
                                      ndist['soci']['lb'], ndist['soci']['ub'],
                                      self.config.n, self.config.seed)
        self.socd.v = build_truncnorm(ndist['socd']['mu'], ndist['socd']['var'],
                                      ndist['socd']['lb'], ndist['socd']['ub'],
                                      self.config.n, self.config.seed)
        self.tt.v = build_truncnorm(ndist['tt']['mu'], ndist['tt']['var'],
                                    ndist['tt']['lb'], ndist['tt']['ub'],
                                    self.config.n, self.config.seed)
        # set `ts`, `tf`
        tdf = pd.DataFrame({
            col: build_truncnorm(ndist[col]['mu'], ndist[col]['var'],
                                 ndist[col]['lb'], ndist[col]['ub'],
                                 self.config.n, self.config.seed)
            for col in ['ts1', 'ts2', 'tf1', 'tf2']
        })

        nev_t1 = int(self.config.n * self.config.r)  # number of EVs in time range 1
        tp1 = tdf[['ts1', 'tf1']].sample(n=nev_t1, random_state=self.config.seed)
        tp2 = tdf[['ts2', 'tf2']].sample(n=self.config.n-nev_t1, random_state=self.config.seed)
        tp = pd.concat([tp1, tp2], axis=0).reset_index(drop=True).fillna(0)
        tp['ts'] = tp['ts1'] + tp['ts2']
        tp['tf'] = tp['tf1'] + tp['tf2']
        # Swap ts and tf if ts > tf
        check = tp['ts'] > tp['tf']
        tp.loc[check, ['ts', 'tf']] = tp.loc[check, ['tf', 'ts']].values

        self.ts.v = tp['ts'].values
        self.tf.v = tp['tf'].values

        # set `Pc`, `Pd`, `nc`, `nd`, `Q`
        # NOTE: here it assumes (1) Pc == Pd, (2) nc == nd given by ref[2]
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        self.Pc.v = np.random.uniform(udist['Pc']['lb'], udist['Pc']['ub'], self.config.n)
        self.Pd.v = self.Pc.v
        self.nc.v = np.random.uniform(udist['nc']['lb'], udist['nc']['ub'], self.config.n)
        self.nd.v = self.nc.v
        self.Q.v = np.random.uniform(udist['Q']['lb'], udist['Q']['ub'], self.config.n)

        # --- adjust variables given current time ---
        self.g_u()  # update online status
        # adjust SOC considering random behavior
        # NOTE: here we ignore the AGC participation before the current time `self.t`

        # stayed time for the EVs arrived before t, reset negative time to 0
        tc = np.maximum(self.t - self.ts.v, 0)
        self.soc.v = self.soci.v + tc * self.Pc.v * self.nc.v / self.Q.v  # charge them

        tr = (self.socd.v - self.soci.v) * self.Q.v / self.Pc.v / self.nc.v  # time needed to charge to socd

        # ratio of stay/required time, stay less than required time reset to 1
        kt = np.maximum(tc / tr, 1)
        socp = self.socd.v + np.log(kt) * (1 - self.socd.v)  # log scale higher than socd
        mask = kt > 1
        self.soc.v[mask] = socp[mask]  # Update soc

        # clip soc to min/max
        self.soc.v = np.clip(self.soc.v, self.config.socl, self.config.socu)

        self.soc0.v = self.soc.v.copy()
        self.u0.v = self.u.v.copy()

        self.evd = EVA(evd=self, A_csv=self.A_csv)

        self.is_setup = True

        _, s = elapsed(t0)
        msg = f'{self.evdname} aggregator setup in {s}, and the current time is {self.t} H.\n'
        msg += f'It has {self.config.n} EVs in total and {self.u.v.sum()} EVs online.'
        logger.info(msg)

        return self.is_setup

    def g_u(self):
        """
        Update online status of EVs based on current time.
        """
        self.u.v = ((self.ts.v <= self.t) & (self.t <= self.tf.v)).astype(int)

        return True


class EVA:
    """
    State space modeling based EV aggregation model.
    """

    def __init__(self, evd, A_csv=None):
        """
        Parameters
        ----------
        EVD : ams.extension.eva.EVD
            EV Aggregator model.
        A_csv : str, optional
            Path to the CSV file containing the state space matrix A, default is None.
        """
        self.parent = evd

        # states of EV, intersection of charging status and SOC intervals
        # C: charging, I: idle, D: discharging
        states = list(itertools.product(['C', 'I', 'D'], self.parent.soc_intv.keys()))
        self.state = OrderedDict(((''.join(str(i) for i in s), 0.0) for s in states))

        # NOTE: 3*ns comes from the intersection of charging status and SOC intervals
        ns = self.parent.config.ns
        # NOTE: x, A will be updated in `setup()`
        self.x = np.zeros(3*ns)

        # A matrix
        default_A_csv = ams_root() + '/extension/Aest.csv'
        if A_csv:
            try:
                self.A = pd.read_csv(A_csv).values
                logger.debug(f'Loaded A matrix from {A_csv}.')
            except FileNotFoundError:
                self.A = pd.read_csv(default_A_csv).values
                logger.debug(f'File {A_csv} not found, using default A matrix.')
        else:
            self.A = pd.read_csv(default_A_csv).values
            logger.debug('No A matrix provided, using default A matrix.')

        mate = np.eye(ns)
        mat0 = np.zeros((ns, ns))
        self.B = np.vstack((-mate, mate, mat0))
        self.C = np.vstack((mat0, -mate, mate))

        # SSM variables
        kde = stats.gaussian_kde(self.parent.Pc.v)
        step = 0.01
        Pl_values = np.arange(self.parent.Pc.v.min(), self.parent.Pc.v.max(), step)
        self.Pave = 1e-3 * np.sum([Pl * kde.integrate_box(Pl, Pl + step) for Pl in Pl_values])  # kw to MW

        # NOTE: D, Da, Db, Dc, Dd will be scaled by Pave later in `setup()`
        vec1 = np.ones((1, ns))
        vec0 = np.zeros((1, ns))
        self.D = self.Pave * np.hstack((-vec1, vec0, vec0))
        self.Da = self.Pave * np.hstack((vec0, vec0, vec1))
        self.Db = self.Pave * np.hstack((vec1, vec1, vec1))
        self.Db[0, ns] = 0  # low charged EVs don't DC
        self.Dc = self.Pave * np.hstack((-vec1, vec0, vec0))
        self.Dd = self.Pave * np.hstack((-vec1, -vec1, -vec1))
        self.Dd[0, 2*ns-1] = 0  # overcharged EVs don't C


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

    Returns
    -------
    samples : ndarray
        Generated samples.
    """
    a = (lb - mu) / var
    b = (ub - mu) / var
    distribution = stats.truncnorm(a, b, loc=mu, scale=var)
    return distribution.rvs(n, random_state=seed)
