"""
OPF routines.
"""
import logging  # NOQA

from ams.core.param import RParam  # NOQA

from ams.routines.routine import RoutineData, RoutineModel  # NOQA

from ams.opt.omodel import Var, Constraint, Objective  # NOQA


logger = logging.getLogger(__name__)


class DCOPFData(RoutineData):
    """
    DCOPF data.
    """

    def __init__(self):
        RoutineData.__init__(self)
        # --- generator cost ---
        self.ug = RParam(info='Gen connection status',
                         name='ug', tex_name=r'u_{g}',
                         model='StaticGen', src='u',)
        self.ctrl = RParam(info='Gen controllability',
                           name='ctrl', tex_name=r'c_{trl}',
                           model='StaticGen', src='ctrl',)
        self.c2 = RParam(info='Gen cost coefficient 2',
                         name='c2', tex_name=r'c_{2}',
                         unit=r'$/(p.u.^2)', model='GCost',
                         indexer='gen', imodel='StaticGen',)
        self.c1 = RParam(info='Gen cost coefficient 1',
                         name='c1', tex_name=r'c_{1}',
                         unit=r'$/(p.u.)', model='GCost',
                         indexer='gen', imodel='StaticGen',)
        self.c0 = RParam(info='Gen cost coefficient 0',
                         name='c0', tex_name=r'c_{0}',
                         unit=r'$', model='GCost',
                         indexer='gen', imodel='StaticGen',)
        # --- generator limit ---
        self.pmax = RParam(info='Gen maximum active power (system base)',
                           name='pmax', tex_name=r'p_{max}',
                           unit='p.u.', model='StaticGen',)
        self.pmin = RParam(info='Gen minimum active power (system base)',
                           name='pmin', tex_name=r'p_{min}',
                           unit='p.u.', model='StaticGen',)
        self.pg0 = RParam(info='Gen initial active power (system base)',
                          name='p0', tex_name=r'p_{g,0}',
                          unit='p.u.', model='StaticGen',)
        self.Cg = RParam(info='connection matrix for Gen and Bus',
                         name='Cg', tex_name=r'C_{g}',
                         model='mats', src='Cg',)
        # --- load ---
        self.pl = RParam(info='nodal active load (system base)',
                         name='pl', tex_name=r'p_{l}',
                         model='mats', src='pl',
                         unit='p.u.',)
        # --- line ---
        self.rate_a = RParam(info='long-term flow limit',
                             name='rate_a', tex_name=r'R_{ATEA}',
                             unit='MVA', model='Line',)
        self.PTDF = RParam(info='Power transfer distribution factor matrix',
                           name='PTDF', tex_name=r'P_{TDF}',
                           model='mats', src='PTDF',)


class DCOPFBase(RoutineModel):
    """
    Base class for DCOPF dispatch model.

    Overload the ``solve``, ``unpack``, and ``run`` methods.
    """

    def __init__(self, system, config):
        RoutineModel.__init__(self, system, config)

    def solve(self, **kwargs):
        """
        Solve the routine optimization model.
        """
        res = self.om.mdl.solve(**kwargs)
        return res

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
                except KeyError:  # failed to find source var in the owner (model or group)
                    pass
                except TypeError:  # failed to find source var in the owner (model or group)
                    pass
        self.system.recent = self.system.routines[self.class_name]
        return True


class DCOPFModel(DCOPFBase):
    """
    DCOPF model.
    """

    def __init__(self, system, config):
        DCOPFBase.__init__(self, system, config)
        self.info = 'DC Optimal Power Flow'
        self.type = 'DCED'
        # --- vars ---
        self.pg = Var(info='Gen active power (system base)',
                      unit='p.u.', name='pg', src='p',
                      tex_name=r'p_{g}',
                      model='StaticGen',
                      lb=self.pmin, ub=self.pmax,
                      ctrl=self.ctrl, v0=self.pg0)
        self.pn = Var(info='Bus active power injection (system base)',
                      unit='p.u.', name='pn', tex_name=r'p_{n}',
                      model='Bus',)
        # --- constraints ---
        self.pb = Constraint(name='pb', info='power balance',
                             e_str='sum(pl) - sum(pg)',
                             type='eq',)
        self.pinj = Constraint(name='pinj',
                               info='nodal power injection',
                               e_str='Cg@(pn - pl) - pg',
                               type='eq',)
        self.lub = Constraint(name='lub', info='Line limits upper bound',
                              e_str='PTDF @ (pn - pl) - rate_a',
                              type='uq',)
        self.llb = Constraint(name='llb', info='Line limits lower bound',
                              e_str='- PTDF @ (pn - pl) - rate_a',
                              type='uq',)
        # --- objective ---
        self.obj = Objective(name='tc',
                             info='total cost', unit='$',
                             e_str='sum(c2 * pg**2 + c1 * pg + ug * c0)',
                             sense='min',)


class DCOPF(DCOPFData, DCOPFModel):
    """
    Standard DC optimal power flow (DCOPF).

    In this model, the bus injected power ``pn`` is used as internal variable
    between generator output and load demand.
    """

    def __init__(self, system, config):
        DCOPFData.__init__(self)
        DCOPFModel.__init__(self, system, config)
