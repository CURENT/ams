"""
DCOPF routines.
"""
import logging

import numpy as np
from ams.core.param import RParam
from ams.core.service import NumOp, NumOpDual

from ams.routines.routine import RoutineModel

from ams.opt.omodel import Var, Constraint, Objective


logger = logging.getLogger(__name__)


class DCOPF(RoutineModel):
    """
    DC optimal power flow (DCOPF).

    Bus voltage ``vBus`` is fixed to 1.
    Bus angle ``aBus`` is estimated
    as ::math:``a_{Bus} = C_{ft}^{-1} \\times x \\times p_{L}``.
    """

    def __init__(self, system, config):
        RoutineModel.__init__(self, system, config)
        self.info = 'DC Optimal Power Flow'
        self.type = 'DCED'
        # --- Data Section ---
        # --- generator cost ---
        self.c2 = RParam(info='Gen cost coefficient 2',
                         name='c2', tex_name=r'c_{2}',
                         unit=r'$/(p.u.^2)', model='GCost',
                         indexer='gen', imodel='StaticGen',
                         nonneg=True)
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
                           name='pmax', tex_name=r'p_{G, max}',
                           unit='p.u.', model='StaticGen',
                           no_parse=False,)
        self.pmin = RParam(info='Gen minimum active power',
                           name='pmin', tex_name=r'p_{G, min}',
                           unit='p.u.', model='StaticGen',
                           no_parse=False,)
        self.pg0 = RParam(info='Gen initial active power',
                          name='p0', tex_name=r'p_{G,0}',
                          unit='p.u.', model='StaticGen',)
        # --- load ---
        self.pd = RParam(info='active demand',
                         name='pd', tex_name=r'p_{D}',
                         model='StaticLoad', src='p0',
                         unit='p.u.',)
        # --- line ---
        self.x = RParam(info='line reactance',
                        name='x', tex_name=r'x',
                        model='Line', src='x',
                        unit='p.u.', no_parse=True,)
        self.rate_a = RParam(info='long-term flow limit',
                             name='rate_a', tex_name=r'R_{ATEA}',
                             unit='MVA', model='Line',)
        # --- connection matrix ---
        self.Cg = RParam(info='Gen connection matrix',
                         name='Cg', tex_name=r'C_{G}',
                         model='mats', src='Cg',
                         no_parse=True, sparse=True,)
        self.Cl = RParam(info='Load connection matrix',
                         name='Cl', tex_name=r'C_{L}',
                         model='mats', src='Cl',
                         no_parse=True, sparse=True,)
        self.Cft = RParam(info='Line connection matrix',
                          name='Cft', tex_name=r'C_{ft}',
                          model='mats', src='Cft',
                          no_parse=True, sparse=True,)
        self.PTDF = RParam(info='Power Transfer Distribution Factor',
                           name='PTDF', tex_name=r'P_{TDF}',
                           model='mats', src='PTDF',
                           no_parse=True,)
        # --- Model Section ---
        # --- generation ---
        self.pg = Var(info='Gen active power',
                      unit='p.u.',
                      name='pg', tex_name=r'p_{G}',
                      model='StaticGen', src='p',
                      v0=self.pg0)
        # NOTE: Var bounds need to set separately
        pglb = '-pg + mul(nctrle, pg0) + mul(ctrle, pmin)'
        self.pglb = Constraint(name='pglb', info='pg min',
                               e_str=pglb, type='uq',)
        pgub = 'pg - mul(nctrle, pg0) - mul(ctrle, pmax)'
        self.pgub = Constraint(name='pgub', info='pg max',
                               e_str=pgub, type='uq',)
        # --- bus ---
        self.png = Var(info='Bus active power from gen',
                       unit='p.u.',
                       name='png', tex_name=r'p_{Bus,G}',
                       model='Bus',)
        self.pnd = Var(info='Bus active power from load',
                       unit='p.u.',
                       name='pnd', tex_name=r'p_{Bus,D}',
                       model='Bus',)
        self.pngb = Constraint(name='pngb', type='eq',
                               e_str='Cg@png - pg',
                               info='Bus active power from gen',)
        self.pndb = Constraint(name='pndb', type='eq',
                               e_str='Cl@pnd - pd',
                               info='Bus active power from load',)
        # --- line ---
        # NOTE: `ug*pmin` results in unexpected error
        self.plf = Var(info='Line active power',
                       name='plf', tex_name=r'p_{L}',
                       unit='p.u.', model='Line',)
        self.plflb = Constraint(name='plflb', info='Line power lower bound',
                                e_str='-plf - rate_a', type='uq',)
        self.plfub = Constraint(name='plfub', info='Line power upper bound',
                                e_str='plf - rate_a', type='uq',)
        # --- power balance ---
        self.pb = Constraint(name='pb', info='power balance',
                             e_str='sum(pd) - sum(pg)',
                             type='eq',)
        self.pnb = Constraint(name='pnb', type='eq',
                              info='nodal power injection',
                              e_str='PTDF@(png - pnd) - plf',)
        # --- objective ---
        obj = 'sum(mul(c2, power(pg, 2)))' \
            '+ sum(mul(c1, pg))' \
            '+ sum(mul(c0, ug))'
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
        return RoutineModel.run(self, no_code=no_code, **kwargs)

    def unpack(self, **kwargs):
        """
        Unpack the results from CVXPY model.
        """
        # --- copy results from solver into routine algeb ---
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
