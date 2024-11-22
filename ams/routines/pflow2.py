"""
Power flow routines independent from PYPOWER.
"""
import logging
from collections import OrderedDict

from andes.utils.misc import elapsed

from ams.routines.routine import RoutineBase
from ams.interface import _to_andes_pflow

logger = logging.getLogger(__name__)


class PFlow2(RoutineBase):
    """
    Power flow analysis using ANDES PFlow routine.

    More PFlow settings can be changed via `PFlow2._adsys.config` and `PFlow2.om.config`.

    Reference
    ---------

    [1] ANDES Documentation - Simulation and Plot, [Online],

    Available:

    https://docs.andes.app/en/latest/_examples/ex1.html
    """

    def __init__(self, system, config):
        RoutineBase.__init__(self, system, config)
        self.info = 'AC Power Flow'
        self.type = 'PF'
        self._adsys = None

        self.config.add(OrderedDict((('tol', 1e-6),
                                     ('max_iter', 25),
                                     ('method', 'NR'),
                                     ('check_conn', 1),
                                     ('n_factorize', 4),
                                     )))
        self.config.add_extra("_help",
                              tol="convergence tolerance",
                              max_iter="max. number of iterations",
                              method="calculation method",
                              check_conn='check connectivity before power flow',
                              n_factorize="first N iterations to factorize Jacobian in dishonest method",
                              )
        self.config.add_extra("_alt",
                              tol="float",
                              method=("NR", "dishonest", "NK"),
                              check_conn=(0, 1),
                              max_iter=">=10",
                              n_factorize=">0",
                              )

    def init(self, **kwargs):
        """
        Initialize the ANDES PFlow routine.

        kwargs go to andes.system.System().
        """
        self._adsys = _to_andes_pflow(self.system,
                                      no_output=self.system.files.no_output,
                                      config=self.config.as_dict(),
                                      **kwargs)
        self._adsys.setup()
        self.om = self._adsys.PFlow
        self.initialized = True
        return self.initialized

    def solve(self, **kwargs):
        """
        Placeholder.
        """
        return True

    def run(self, **kwargs):
        """
        Run the routine.
        """
        # --- solve optimization ---
        t0, _ = elapsed()
        _ = self.om.run()
        self.exit_code = self._adsys.exit_code
        self.converged = self.exit_code == 0
        _, s = elapsed(t0)
        self.exec_time = float(s.split(" ")[0])
        self.unpack()
        return True

    def _post_solve(self):
        """
        Placeholder.
        """
        return True

    def unpack(self, **kwargs):
        """
        Unpack the results from ANDES PFlow routine.
        """
        # TODO: maybe also include the DC devices results
        bus_idx = self.system.Bus.idx.v
        self.system.Bus.set(src='v', attr='v', idx=bus_idx,
                            value=self._adsys.Bus.get(src='v', attr='v', idx=bus_idx))
        self.system.Bus.set(src='a', attr='v', idx=bus_idx,
                            value=self._adsys.Bus.get(src='a', attr='v', idx=bus_idx))
        pv_idx = self.system.PV.idx.v
        self.system.PV.set(src='p', attr='v', idx=pv_idx,
                           value=self.system.PV.get(src='p0', attr='v', idx=pv_idx))
        self.system.PV.set(src='q', attr='v', idx=pv_idx,
                           value=self._adsys.PV.get(src='q', attr='v', idx=pv_idx))
        slack_idx = self.system.Slack.idx.v
        self.system.Slack.set(src='p', attr='v', idx=slack_idx,
                              value=self._adsys.Slack.get(src='p', attr='v', idx=slack_idx))
        self.system.Slack.set(src='q', attr='v', idx=slack_idx,
                              value=self._adsys.Slack.get(src='q', attr='v', idx=slack_idx))
        return True

    def update(self, **kwargs):
        """
        Placeholder.
        """
        return True

    def enable(self, name):
        raise NotImplementedError

    def disable(self, name):
        raise NotImplementedError

    def addRParam(self, **kwargs):
        raise NotImplementedError

    def addService(self, **kwargs):
        raise NotImplementedError

    def addConstrs(self, **kwargs):
        raise NotImplementedError

    def addVars(self, **kwargs):
        raise NotImplementedError

    def export_csv(self, path=None):
        # TODO: customize the CSV output
        raise NotImplementedError