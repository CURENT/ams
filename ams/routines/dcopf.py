"""
DCOPF routines.
"""
import logging

import numpy as np
from ams.core.param import RParam
from ams.core.service import NumOp, NumOpDual

from ams.routines.routine import RoutineBase

from ams.opt.omodel import Var, Constraint, Objective


logger = logging.getLogger(__name__)


class DCOPF(RoutineBase):
    """
    DC optimal power flow (DCOPF).

    Line flow variable `plf` is calculated as ``Bf@aBus + Pfinj``
    after solving the problem in ``_post_solve()`` .
    """

    def __init__(self, system, config):
        RoutineBase.__init__(self, system, config)
        self.info = 'DC Optimal Power Flow'
        self.type = 'DCED'
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
        # --- load ---
        self.pd = RParam(info='active demand',
                         name='pd', tex_name=r'p_{d}',
                         model='StaticLoad', src='p0',
                         unit='p.u.',)
        # --- line ---
        self.rate_a = RParam(info='long-term flow limit',
                             name='rate_a', tex_name=r'R_{ATEA}',
                             unit='MVA', model='Line',)
        # --- line angle difference ---
        self.amax = NumOp(u=self.rate_a, fun=np.ones_like,
                          rfun=np.dot, rargs=dict(b=np.pi),
                          name='amax', tex_name=r'\theta_{bus, max}',
                          info='max line angle difference',
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
                               e_str=pglb, type='uq',)
        pgub = 'pg - mul(nctrle, pg0) - mul(ctrle, pmax)'
        self.pgub = Constraint(name='pgub', info='pg max',
                               e_str=pgub, type='uq',)
        # --- bus ---
        self.aBus = Var(info='Bus voltage angle',
                        unit='rad',
                        name='aBus', tex_name=r'\theta_{bus}',
                        model='Bus', src='a',)
        # --- power balance ---
        pb = 'Bbus@aBus + Pbusinj + Cl@pd + Csh@gsh - Cg@pg'
        self.pb = Constraint(name='pb', info='power balance',
                             e_str=pb, type='eq',)
        # --- line flow ---
        self.plf = Var(info='Line flow',
                       unit='p.u.',
                       name='plf', tex_name=r'p_{lf}',
                       model='Line',)
        self.plflb = Constraint(info='line flow lower bound',
                                name='plflb', type='uq',
                                e_str='-Bf@aBus - Pfinj - rate_a',)
        self.plfub = Constraint(info='line flow upper bound',
                                name='plfub', type='uq',
                                e_str='Bf@aBus + Pfinj - rate_a',)
        self.alflb = Constraint(info='line angle difference lower bound',
                                name='alflb', type='uq',
                                e_str='-CftT@aBus - amax',)
        self.alfub = Constraint(info='line angle difference upper bound',
                                name='alfub', type='uq',
                                e_str='CftT@aBus - amax',)

        # --- objective ---
        obj = 'sum(mul(c2, power(pg, 2)))'
        obj += '+ sum(mul(c1, pg))'
        obj += '+ sum(mul(ug, c0))'
        self.obj = Objective(name='obj',
                             info='total cost', unit='$',
                             sense='min', e_str=obj,)

    def solve(self, **kwargs):
        """
        Solve the routine optimization model.
        """
        return self.om.prob.solve(**kwargs)

    def run(self, no_code=True, **kwargs):
        """
        Run the routine.

        Parameters
        ----------
        no_code : bool, optional
            If True, print the generated CVXPY code. Defaults to False.

        Other Parameters
        ----------------
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
        kwargs : keywords, optional
            Additional solver specific arguments. See CVXPY documentation for details.
        """
        return RoutineBase.run(self, no_code=no_code, **kwargs)

    def _post_solve(self):
        # --- post-solving calculations ---
        # line flow: Bf@aBus + Pfinj
        # using sparse matrix in MatProcessor is faster
        self.plf.optz.value = self.system.mats.Bf._v@self.aBus.v + self.Pfinj.v
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
                    var.owner.set(src=var.src, attr='v', idx=idx, value=var.v)
                # failed to find source var in the owner (model or group)
                except (KeyError, TypeError):
                    pass

        # label the most recent solved routine
        self.system.recent = self.system.routines[self.class_name]
        return True
