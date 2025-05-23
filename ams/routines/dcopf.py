"""
DCOPF routines.
"""
import logging

import numpy as np
from ams.core.param import RParam
from ams.core.service import NumOp, NumOpDual

from ams.routines.dcpf import DCPFBase

from ams.opt import Constraint, Objective, ExpressionCalc, Expression


logger = logging.getLogger(__name__)


class DCOPF(DCPFBase):
    """
    DC optimal power flow (DCOPF).

    Notes
    -----
    1. The nodal price is calculated as ``pi`` in ``pic``.
    2. Devices online status of ``StaticGen``, ``StaticLoad``, and ``Shunt`` are considered in the connectivity
       matrices ``Cft``, ``Cg``, ``Cl``, and ``Csh``.

    References
    ----------
    1. R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, “MATPOWER: Steady-State
       Operations, Planning, and Analysis Tools for Power Systems Research and Education,” IEEE
       Trans. Power Syst., vol. 26, no. 1, pp. 12-19, Feb. 2011
    """

    def __init__(self, system, config):
        DCPFBase.__init__(self, system, config)
        self.info = 'DC Optimal Power Flow'
        self.type = 'DCED'

        # --- Mapping Section ---
        # --- from map ---
        self.map1.update({
            'ug': ('StaticGen', 'u'),
        })
        # --- to map ---
        self.map2.update({
            'vBus': ('Bus', 'v0'),
            'ug': ('StaticGen', 'u'),
            'pg': ('StaticGen', 'p0'),
        })

        # --- Data Section ---
        # --- generator cost ---
        self.c2 = RParam(info='Gen cost coefficient 2',
                         name='c2', tex_name=r'c_{2}',
                         unit=r'$/(p.u.^2)', model='GCost',
                         indexer='gen', imodel='StaticGen',
                         nonneg=True, no_parse=True)
        self.c1 = RParam(info='Gen cost coefficient 1',
                         name='c1', tex_name=r'c_{1}',
                         unit=r'$/(p.u.)', model='GCost',
                         indexer='gen', imodel='StaticGen',)
        self.c0 = RParam(info='Gen cost coefficient 0',
                         name='c0', tex_name=r'c_{0}',
                         unit=r'$', model='GCost',
                         indexer='gen', imodel='StaticGen',
                         no_parse=True)
        # --- generator ---
        self.ctrl = RParam(info='Gen controllability',
                           name='ctrl', tex_name=r'c_{trl}',
                           model='StaticGen', src='ctrl',
                           no_parse=True)
        self.ctrle = NumOpDual(info='Effective Gen controllability',
                               name='ctrle', tex_name=r'c_{trl, e}',
                               u=self.ctrl, u2=self.ug,
                               fun=np.multiply, no_parse=True)
        self.nctrl = NumOp(u=self.ctrl, fun=np.logical_not,
                           name='nctrl', tex_name=r'c_{trl,n}',
                           info='Effective Gen uncontrollability',
                           no_parse=True,)
        self.nctrle = NumOpDual(info='Effective Gen uncontrollability',
                                name='nctrle', tex_name=r'c_{trl,n,e}',
                                u=self.nctrl, u2=self.ug,
                                fun=np.multiply, no_parse=True)
        self.pmax = RParam(info='Gen maximum active power',
                           name='pmax', tex_name=r'p_{g, max}',
                           unit='p.u.', model='StaticGen',
                           no_parse=False,)
        self.pmin = RParam(info='Gen minimum active power',
                           name='pmin', tex_name=r'p_{g, min}',
                           unit='p.u.', model='StaticGen',
                           no_parse=False,)

        # --- line ---
        self.ul = RParam(info='Line connection status',
                         name='ul', tex_name=r'u_{l}',
                         model='Line', src='u',
                         no_parse=True,)
        self.rate_a = RParam(info='long-term flow limit',
                             name='rate_a', tex_name=r'R_{ATEA}',
                             unit='p.u.', model='Line',)
        # --- line angle difference ---
        self.amax = RParam(model='Line', src='amax',
                           name='amax', tex_name=r'\theta_{bus, max}',
                           info='max line angle difference',
                           no_parse=True,)
        self.amin = RParam(model='Line', src='amin',
                           name='amin', tex_name=r'\theta_{bus, min}',
                           info='min line angle difference',
                           no_parse=True,)

        # --- Model Section ---
        self.pmaxe = Expression(info='Effective pmax',
                                name='pmaxe', tex_name=r'p_{g, max, e}',
                                e_str='mul(nctrle, pg0) + mul(ctrle, pmax)',
                                model='StaticGen', src=None, unit='p.u.',)
        self.pmine = Expression(info='Effective pmin',
                                name='pmine', tex_name=r'p_{g, min, e}',
                                e_str='mul(nctrle, pg0) + mul(ctrle, pmin)',
                                model='StaticGen', src=None, unit='p.u.',)
        self.pglb = Constraint(name='pglb', info='pg min',
                               e_str='-pg + pmine', is_eq=False,)
        self.pgub = Constraint(name='pgub', info='pg max',
                               e_str='pg - pmaxe', is_eq=False,)

        # --- line flow ---
        self.plflb = Constraint(info='line flow lower bound',
                                name='plflb', is_eq=False,
                                e_str='-plf - mul(ul, rate_a)',)
        self.plfub = Constraint(info='line flow upper bound',
                                name='plfub', is_eq=False,
                                e_str='plf - mul(ul, rate_a)',)
        self.alflb = Constraint(info='line angle difference lower bound',
                                name='alflb', is_eq=False,
                                e_str='-CftT@aBus + amin',)
        self.alfub = Constraint(info='line angle difference upper bound',
                                name='alfub', is_eq=False,
                                e_str='CftT@aBus - amax',)
        # NOTE: in CVXPY, dual_variables returns a list
        self.pi = ExpressionCalc(info='LMP, dual of <pb>',
                                 name='pi', unit='$/p.u.',
                                 model='Bus', src=None,
                                 e_str='pb.dual_variables[0]')
        self.mu1 = ExpressionCalc(info='Lagrange multipliers, dual of <plflb>',
                                  name='mu1', unit='$/p.u.',
                                  model='Line', src=None,
                                  e_str='plflb.dual_variables[0]')
        self.mu2 = ExpressionCalc(info='Lagrange multipliers, dual of <plfub>',
                                  name='mu2', unit='$/p.u.',
                                  model='Line', src=None,
                                  e_str='plfub.dual_variables[0]')

        # --- objective ---
        obj = 'sum(mul(c2, pg**2))'
        obj += '+ sum(mul(c1, pg))'
        obj += '+ sum(mul(ug, c0))'
        self.obj = Objective(name='obj',
                             info='total cost', unit='$',
                             sense='min', e_str=obj,)

    def dc2ac(self, kloss=1.0, **kwargs):
        """
        Convert the results using ACOPF.

        Parameters
        ----------
        kloss : float, optional
            The loss factor for the conversion. Defaults to 1.0.
        """
        exec_time = self.exec_time
        if self.exec_time == 0 or self.exit_code != 0:
            logger.warning(f'{self.class_name} is not executed successfully, quit conversion.')
            return False

        # --- ACOPF ---
        # scale up load
        pq_idx = self.system.StaticLoad.get_all_idxes()
        pd0 = self.system.StaticLoad.get(src='p0', attr='v', idx=pq_idx).copy()
        qd0 = self.system.StaticLoad.get(src='q0', attr='v', idx=pq_idx).copy()
        self.system.StaticLoad.set(src='p0', idx=pq_idx, attr='v', value=pd0 * kloss)
        self.system.StaticLoad.set(src='q0', idx=pq_idx, attr='v', value=qd0 * kloss)
        # run ACOPF
        ACOPF = self.system.ACOPF
        ACOPF.run()
        # scale load back
        self.system.StaticLoad.set(src='p0', idx=pq_idx, attr='v', value=pd0)
        self.system.StaticLoad.set(src='q0', idx=pq_idx, attr='v', value=qd0)
        if not ACOPF.exit_code == 0:
            logger.warning('<ACOPF> did not converge, conversion failed.')
            # NOTE: mock results to fit interface with ANDES
            self.vBus = ACOPF.vBus
            self.vBus.optz.value = np.ones(self.system.Bus.n)
            self.aBus.optz.value = np.zeros(self.system.Bus.n)
            return False
        self.pg.optz.value = ACOPF.pg.v

        # NOTE: mock results to fit interface with ANDES
        self.vBus.optz.value = ACOPF.vBus.v
        self.aBus.optz.value = ACOPF.aBus.v
        self.exec_time = exec_time

        # --- set status ---
        self.system.recent = self
        self.converted = True
        logger.warning(f'<{self.class_name}> converted to AC.')
        return True

    def run(self, **kwargs):
        """
        Run the routine.

        Following kwargs go to `self.init()`: `force`, `force_mats`, `force_constr`, `force_om`.

        Following kwargs go to `self.solve()`: `solver`, `verbose`, `gp`, `qcp`, `requires_grad`,
        `enforce_dpp`, `ignore_dpp`, `method`, and all rest.

        Parameters
        ----------
        force : bool, optional
            If True, force re-initialization. Defaults to False.
        force_mats : bool, optional
            If True, force re-generating matrices. Defaults to False.
        force_constr : bool, optional
            Whether to turn on all constraints.
        force_om : bool, optional
            If True, force re-generating optimization model. Defaults to False.
        solver: str, optional
            The solver to use. For example, 'GUROBI', 'ECOS', 'SCS', or 'OSQP'.
        verbose : bool, optional
            Overrides the default of hiding solver output and prints logging
            information describing CVXPY's compilation process.
        gp : bool, optional
            If True, parses the problem as a disciplined geometric program
            instead of a disciplined convex program.
        qcp : bool, optional
            If True, parses the problem as a disciplined quasiconvex program
            instead of a disciplined convex program.
        requires_grad : bool, optional
            Makes it possible to compute gradients of a solution with respect to Parameters
            by calling problem.backward() after solving, or to compute perturbations to the variables
            given perturbations to Parameters by calling problem.derivative().
            Gradients are only supported for DCP and DGP problems, not quasiconvex problems.
            When computing gradients (i.e., when this argument is True), the problem must satisfy the DPP rules.
        enforce_dpp : bool, optional
            When True, a DPPError will be thrown when trying to solve a
            non-DPP problem (instead of just a warning).
            Only relevant for problems involving Parameters. Defaults to False.
        ignore_dpp : bool, optional
            When True, DPP problems will be treated as non-DPP, which may speed up compilation. Defaults to False.
        method : function, optional
            A custom solve method to use.
        """
        return super().run(**kwargs)
