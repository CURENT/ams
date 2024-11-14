"""
DCOPF routines.
"""
import logging

import numpy as np
from ams.core.param import RParam
from ams.core.service import NumOp, NumOpDual, VarSelect

from ams.routines.routine import RoutineBase

from ams.opt.omodel import Var, Constraint, Objective, ExpressionCalc


logger = logging.getLogger(__name__)


class DCOPF(RoutineBase):
    """
    DC optimal power flow (DCOPF).

    The nodal price is calculated as ``pi`` in ``pic``.

    References
    ----------
    1. R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, “MATPOWER: Steady-State Operations, Planning, and
    Analysis Tools for Power Systems Research and Education,” IEEE Trans. Power Syst., vol. 26, no. 1, pp. 12–19,
    Feb. 2011
    """

    def __init__(self, system, config):
        RoutineBase.__init__(self, system, config)
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
        self.ug = RParam(info='Gen connection status',
                         name='ug', tex_name=r'u_{g}',
                         model='StaticGen', src='u',
                         no_parse=True)
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
        self.pg0 = RParam(info='Gen initial active power',
                          name='p0', tex_name=r'p_{g, 0}',
                          unit='p.u.',
                          model='StaticGen', src='pg0')
        # --- bus ---
        self.buss = RParam(info='Bus slack',
                           name='buss', tex_name=r'B_{us,s}',
                           model='Slack', src='bus',
                           no_parse=True,)
        # --- load ---
        self.upq = RParam(info='Load connection status',
                          name='upq', tex_name=r'u_{PQ}',
                          model='StaticLoad', src='u',
                          no_parse=True,)
        self.pd = RParam(info='active demand',
                         name='pd', tex_name=r'p_{d}',
                         model='StaticLoad', src='p0',
                         unit='p.u.',)
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
        # --- shunt ---
        self.gsh = RParam(info='shunt conductance',
                          name='gsh', tex_name=r'g_{sh}',
                          model='Shunt', src='g',
                          no_parse=True,)
        # --- connection matrix ---
        self.Cg = RParam(info='Gen connection matrix',
                         name='Cg', tex_name=r'C_{g}',
                         model='mats', src='Cg',
                         no_parse=True, sparse=True,)
        self.Cl = RParam(info='Load connection matrix',
                         name='Cl', tex_name=r'C_{l}',
                         model='mats', src='Cl',
                         no_parse=True, sparse=True,)
        self.CftT = RParam(info='Transpose of line connection matrix',
                           name='CftT', tex_name=r'C_{ft}^T',
                           model='mats', src='CftT',
                           no_parse=True, sparse=True,)
        self.Csh = RParam(info='Shunt connection matrix',
                          name='Csh', tex_name=r'C_{sh}',
                          model='mats', src='Csh',
                          no_parse=True, sparse=True,)
        # --- system matrix ---
        self.Bbus = RParam(info='Bus admittance matrix',
                           name='Bbus', tex_name=r'B_{bus}',
                           model='mats', src='Bbus',
                           no_parse=True, sparse=True,)
        self.Bf = RParam(info='Bf matrix',
                         name='Bf', tex_name=r'B_{f}',
                         model='mats', src='Bf',
                         no_parse=True, sparse=True,)
        self.Pbusinj = RParam(info='Bus power injection vector',
                              name='Pbusinj', tex_name=r'P_{bus}^{inj}',
                              model='mats', src='Pbusinj',
                              no_parse=True,)
        self.Pfinj = RParam(info='Line power injection vector',
                            name='Pfinj', tex_name=r'P_{f}^{inj}',
                            model='mats', src='Pfinj',
                            no_parse=True,)

        # --- Model Section ---
        # --- generation ---
        self.pg = Var(info='Gen active power',
                      unit='p.u.',
                      name='pg', tex_name=r'p_g',
                      model='StaticGen', src='p',
                      v0=self.pg0)
        pglb = '-pg + mul(nctrle, pg0) + mul(ctrle, pmin)'
        self.pglb = Constraint(name='pglb', info='pg min',
                               e_str=pglb, is_eq=False,)
        pgub = 'pg - mul(nctrle, pg0) - mul(ctrle, pmax)'
        self.pgub = Constraint(name='pgub', info='pg max',
                               e_str=pgub, is_eq=False,)
        # --- bus ---
        self.aBus = Var(info='Bus voltage angle',
                        unit='rad',
                        name='aBus', tex_name=r'\theta_{bus}',
                        model='Bus', src='a',)
        self.csb = VarSelect(info='select slack bus',
                             name='csb', tex_name=r'c_{sb}',
                             u=self.aBus, indexer='buss',
                             no_parse=True,)
        self.sba = Constraint(info='align slack bus angle',
                              name='sbus', is_eq=True,
                              e_str='csb@aBus',)
        self.pi = Var(info='nodal price',
                      name='pi', tex_name=r'\pi',
                      unit='$/p.u.',
                      model='Bus',)
        # --- power balance ---
        pb = 'Bbus@aBus + Pbusinj + Cl@(mul(upq, pd)) + Csh@gsh - Cg@pg'
        self.pb = Constraint(name='pb', info='power balance',
                             e_str=pb, is_eq=True,)
        # --- line flow ---
        self.plf = Var(info='Line flow',
                       unit='p.u.',
                       name='plf', tex_name=r'p_{lf}',
                       model='Line',)
        self.plflb = Constraint(info='line flow lower bound',
                                name='plflb', is_eq=False,
                                e_str='-Bf@aBus - Pfinj - mul(ul, rate_a)',)
        self.plfub = Constraint(info='line flow upper bound',
                                name='plfub', is_eq=False,
                                e_str='Bf@aBus + Pfinj - mul(ul, rate_a)',)
        self.alflb = Constraint(info='line angle difference lower bound',
                                name='alflb', is_eq=False,
                                e_str='-CftT@aBus + amin',)
        self.alfub = Constraint(info='line angle difference upper bound',
                                name='alfub', is_eq=False,
                                e_str='CftT@aBus - amax',)
        self.plfc = ExpressionCalc(info='plf calculation',
                                   name='plfc', var='plf',
                                   e_str='Bf@aBus + Pfinj')
        # NOTE: in CVXPY, dual_variables returns a list
        self.pic = ExpressionCalc(info='dual of Constraint pb',
                                  name='pic', var='pi',
                                  e_str='pb.dual_variables[0]')

        # --- objective ---
        obj = 'sum(mul(c2, pg**2))'
        obj += '+ sum(mul(c1, pg))'
        obj += '+ sum(mul(ug, c0))'
        self.obj = Objective(name='obj',
                             info='total cost', unit='$',
                             sense='min', e_str=obj,)

    def solve(self, **kwargs):
        """
        Solve the routine optimization model.
        args and kwargs go to `self.om.prob.solve()` (`cvxpy.Problem.solve()`).
        """
        return self.om.prob.solve(**kwargs)

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

    def _post_solve(self):
        """
        Post-solve calculations.
        """
        for expr in self.exprs.values():
            try:
                var = getattr(self, expr.var)
                var.optz.value = expr.v
                logger.debug(f'Post solve: {var} = {expr.e_str}')
            except AttributeError:
                raise AttributeError(f'No such variable {expr.var}')
        return True

    def unpack(self, **kwargs):
        """
        Unpack the results from CVXPY model.
        """
        # --- solver results to routine algeb ---
        for _, var in self.vars.items():
            # --- copy results from routine algeb into system algeb ---
            if var.model is None:          # if no owner
                continue
            if var.src is None:            # if no source
                continue
            else:
                try:
                    idx = var.owner.get_idx()
                except AttributeError:
                    idx = var.owner.idx.v
                else:
                    pass
                # NOTE: only unpack the variables that are in the model or group
                try:
                    var.owner.set(src=var.src, idx=idx, attr='v', value=var.v)
                except (KeyError, TypeError):
                    logger.error(f'Failed to unpack <{var}> to <{var.owner.class_name}>.')
                    pass

        # label the most recent solved routine
        self.system.recent = self.system.routines[self.class_name]
        return True

    def dc2ac(self, kloss=1.0, **kwargs):
        """
        Convert the DCOPF results with ACOPF.

        Parameters
        ----------
        kloss : float, optional
            The loss factor for the conversion. Defaults to 1.2.
        """
        exec_time = self.exec_time
        if self.exec_time == 0 or self.exit_code != 0:
            logger.warning(f'{self.class_name} is not executed successfully, quit conversion.')
            return False

        # --- ACOPF ---
        # scale up load
        pq_idx = self.system.StaticLoad.get_idx()
        pd0 = self.system.StaticLoad.get(src='p0', attr='v', idx=pq_idx).copy()
        qd0 = self.system.StaticLoad.get(src='q0', attr='v', idx=pq_idx).copy()
        self.system.StaticLoad.set(src='p0', idx=pq_idx, attr='v', value=pd0 * kloss)
        self.system.StaticLoad.set(src='q0', idx=pq_idx, attr='v', value=qd0 * kloss)

        ACOPF = self.system.ACOPF
        # run ACOPF
        ACOPF.run()
        # self.exec_time += ACOPF.exec_time
        # scale load back
        self.system.StaticLoad.set(src='p0', idx=pq_idx, value=pd0)
        self.system.StaticLoad.set(src='q0', idx=pq_idx, value=qd0)
        if not ACOPF.exit_code == 0:
            logger.warning('<ACOPF> did not converge, conversion failed.')
            # NOTE: mock results to fit interface with ANDES
            self.vBus = ACOPF.vBus
            self.vBus.optz.value = np.ones(self.system.Bus.n)
            self.aBus.optz.value = np.zeros(self.system.Bus.n)
            return False
        self.pg.optz.value = ACOPF.pg.v

        # NOTE: mock results to fit interface with ANDES
        self.addVars(name='vBus',
                     info='Bus voltage', unit='p.u.',
                     model='Bus', src='v',)
        self.vBus.parse()
        self.vBus.optz.value = ACOPF.vBus.v
        self.aBus.optz.value = ACOPF.aBus.v
        self.exec_time = exec_time

        # --- set status ---
        self.system.recent = self
        self.converted = True
        logger.warning(f'<{self.class_name}> converted to AC.')
        return True
