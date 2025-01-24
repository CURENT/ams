"""
Power flow routines using PyOptInterface.
"""
from pyoptinterface import nlfunc, ipopt
import pyoptinterface as poi
import math

import logging
from typing import Optional, Union, Type
from collections import OrderedDict

import numpy as np

from andes.utils.misc import elapsed

from ams.core.param import RParam
from ams.routines.routine import RoutineBase
from ams.opt import Var, Expression, Objective

logger = logging.getLogger(__name__)


class PFlow1(RoutineBase):
    """
    UNDER DEVELOPMENT!
    
    Power flow analysis using PyOptInterface.

    References
    ----------
    [1] Y. Yang, C. Lin, L. Xu, X. Yang, W. Wu and B. Wang, "Accelerating Optimal Power Flow with
    Structure-aware Automatic Differentiation and Code Generation," in IEEE Transactions on Power Systems

    [2] Y. Yang, C. Lin, L. Xu, and W. Wu, "PyOptInterface: Design and implementation of an efficient
    modeling language for mathematical optimization," arXiv, 2024.
    """

    def __init__(self, system, config):
        RoutineBase.__init__(self, system, config)
        self.info = 'AC Power Flow'
        self.type = 'PF'
        self._adsys = None

        self.config.add(OrderedDict((('tol', 1e-6),
                                     )))
        self.config.add_extra("_help",
                              tol="convergence tolerance",
                              )
        self.config.add_extra("_alt",
                              tol="float",
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
