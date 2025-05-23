"""
Power flow routines using ANDES.
"""
import logging
from typing import Optional, Union, Type
from collections import OrderedDict

import numpy as np

from andes.utils.misc import elapsed

from ams.core.param import RParam
from ams.routines.routine import RoutineBase
from ams.opt import Var, Expression, Objective
from ams.interface import to_andes_pflow, sync_adsys

logger = logging.getLogger(__name__)


class PFlow(RoutineBase):
    """
    Power flow analysis using ANDES PFlow routine.

    More settings can be changed via ``PFlow2._adsys.config`` and ``PFlow2._adsys.PFlow.config``.

    All generator output powers, bus voltages, and angles are included in the variable definitions.
    However, not all of these are unknowns; the definitions are provided for easy access.

    References
    ----------
    1. M. L. Crow, Computational methods for electric power systems. 2015.
    2. ANDES Documentation - Simulation and Plot.
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

        self.Bf = RParam(info='Bf matrix',
                         name='Bf', tex_name=r'B_{f}',
                         model='mats', src='Bf',
                         no_parse=True, sparse=True,)
        self.Pfinj = RParam(info='Line power injection vector',
                            name='Pfinj', tex_name=r'P_{f}^{inj}',
                            model='mats', src='Pfinj',
                            no_parse=True,)

        self.pg = Var(info='Gen active power',
                      unit='p.u.',
                      name='pg', tex_name=r'p_g',
                      model='StaticGen', src='p')
        self.qg = Var(info='Gen reactive power',
                      unit='p.u.',
                      name='qg', tex_name=r'q_g',
                      model='StaticGen', src='q')
        self.aBus = Var(info='Bus voltage angle',
                        unit='rad',
                        name='aBus', tex_name=r'\theta_{bus}',
                        model='Bus', src='a',)
        self.vBus = Var(info='Bus voltage magnitude',
                        unit='p.u.',
                        name='vBus', tex_name=r'V_{bus}',
                        model='Bus', src='v',)
        self.plf = Expression(info='Line flow',
                              name='plf', tex_name=r'p_{lf}',
                              unit='p.u.',
                              e_str='Bf@aBus + Pfinj',
                              model='Line', src=None,)

        self.obj = Objective(name='obj',
                             info='place holder', unit='$',
                             sense='min', e_str='0',)

    def init(self, **kwargs):
        """
        Initialize the ANDES PFlow routine.

        kwargs go to andes.system.System().
        """
        self._adsys = to_andes_pflow(self.system,
                                     no_output=self.system.files.no_output,
                                     config=self.config.as_dict(),
                                     **kwargs)
        self._adsys.setup()
        self.om.init()
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
        if not self.initialized:
            self.init()

        t0, _ = elapsed()
        _ = self._adsys.PFlow.run()
        self.exit_code = self._adsys.exit_code
        self.converged = self.exit_code == 0
        _, s = elapsed(t0)
        self.exec_time = float(s.split(" ")[0])

        self.unpack(res=None)
        return self.converged

    def _post_solve(self):
        """
        Placeholder.
        """
        return True

    def unpack(self, res, **kwargs):
        """
        Unpack the results from ANDES PFlow routine.
        """
        # TODO: maybe also include the DC devices results
        sys = self.system
        # --- device results ---
        bus_idx = sys.Bus.idx.v
        sys.Bus.set(src='v', attr='v', idx=bus_idx,
                    value=self._adsys.Bus.get(src='v', attr='v', idx=bus_idx))
        sys.Bus.set(src='a', attr='v', idx=bus_idx,
                    value=self._adsys.Bus.get(src='a', attr='v', idx=bus_idx))
        pv_idx = sys.PV.idx.v
        pv_u = sys.PV.get(src='u', attr='v', idx=pv_idx)
        # NOTE: for p, we should consider the online status as p0 is a param
        sys.PV.set(src='p', attr='v', idx=pv_idx,
                   value=pv_u * sys.PV.get(src='p0', attr='v', idx=pv_idx))
        sys.PV.set(src='q', attr='v', idx=pv_idx,
                   value=self._adsys.PV.get(src='q', attr='v', idx=pv_idx))
        slack_idx = sys.Slack.idx.v
        sys.Slack.set(src='p', attr='v', idx=slack_idx,
                      value=self._adsys.Slack.get(src='p', attr='v', idx=slack_idx))
        sys.Slack.set(src='q', attr='v', idx=slack_idx,
                      value=self._adsys.Slack.get(src='q', attr='v', idx=slack_idx))
        # --- routine results ---
        self.pg.optz.value = sys.StaticGen.get(src='p', attr='v', idx=self.pg.get_all_idxes())
        self.qg.optz.value = sys.StaticGen.get(src='q', attr='v', idx=self.qg.get_all_idxes())
        self.aBus.optz.value = sys.Bus.get(src='a', attr='v', idx=self.aBus.get_all_idxes())
        self.vBus.optz.value = sys.Bus.get(src='v', attr='v', idx=self.vBus.get_all_idxes())
        return True

    def update(self, params=None, build_mats=False):
        """
        This method updates the parameters in the optimization model. In this routine,
        the `params` and `build_mats` arguments are not used because the parameters
        are updated directly to the ANDES system.
        """
        if not self.initialized:
            self.init()

        sync_adsys(self.system, self._adsys)

        return True

    def enable(self, name):
        raise NotImplementedError

    def disable(self, name):
        raise NotImplementedError

    def addRParam(self,
                  name: str,
                  tex_name: Optional[str] = None,
                  info: Optional[str] = None,
                  src: Optional[str] = None,
                  unit: Optional[str] = None,
                  model: Optional[str] = None,
                  v: Optional[np.ndarray] = None,
                  indexer: Optional[str] = None,
                  imodel: Optional[str] = None,):
        """
        Not supported!
        """
        raise NotImplementedError

    def addService(self,
                   name: str,
                   value: np.ndarray,
                   tex_name: str = None,
                   unit: str = None,
                   info: str = None,
                   vtype: Type = None,):
        """
        Not supported!
        """
        raise NotImplementedError

    def addConstrs(self,
                   name: str,
                   e_str: str,
                   info: Optional[str] = None,
                   is_eq: Optional[str] = False,):
        """
        Not supported!
        """
        raise NotImplementedError

    def addVars(self,
                name: str,
                model: Optional[str] = None,
                shape: Optional[Union[int, tuple]] = None,
                tex_name: Optional[str] = None,
                info: Optional[str] = None,
                src: Optional[str] = None,
                unit: Optional[str] = None,
                horizon: Optional[RParam] = None,
                nonneg: Optional[bool] = False,
                nonpos: Optional[bool] = False,
                cplx: Optional[bool] = False,
                imag: Optional[bool] = False,
                symmetric: Optional[bool] = False,
                diag: Optional[bool] = False,
                psd: Optional[bool] = False,
                nsd: Optional[bool] = False,
                hermitian: Optional[bool] = False,
                boolean: Optional[bool] = False,
                integer: Optional[bool] = False,
                pos: Optional[bool] = False,
                neg: Optional[bool] = False,):
        """
        Not supported!
        """
        raise NotImplementedError
