"""
Power flow routines independent from PYPOWER.
"""
import logging

import numpy as np  # noqa
from ams.core.param import RParam
from ams.core.service import NumOp, NumOpDual, VarSelect  # noqa

from ams.routines.routine import RoutineBase

from ams.opt import Var, Constraint, Objective, Expression, ExpressionCalc  # noqa


logger = logging.getLogger(__name__)


class PFlow2(RoutineBase):
    """
    Power flow routine.

    [In progress]

    References
    ----------
    1. R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, “MATPOWER: Steady-State Operations, Planning, and
    Analysis Tools for Power Systems Research and Education,” IEEE Trans. Power Syst., vol. 26, no. 1, pp. 12–19,
    Feb. 2011
    """

    def __init__(self, system, config):
        RoutineBase.__init__(self, system, config)
        self.info = 'AC Power Flow'
        self.type = 'PF'

        # --- Mapping Section ---
        # TODO: skip for now
        # # --- from map ---
        # self.map1.update({
        #     'ug': ('StaticGen', 'u'),
        # })
        # # --- to map ---
        # self.map2.update({
        #     'vBus': ('Bus', 'v0'),
        #     'ug': ('StaticGen', 'u'),
        #     'pg': ('StaticGen', 'p0'),
        # })

        # --- Data Section ---
        # --- generator ---
        self.ug = RParam(info='Gen connection status',
                         name='ug', tex_name=r'u_{g}',
                         model='StaticGen', src='u',
                         no_parse=True)
        # --- load ---
        self.pd = RParam(info='active demand',
                         name='pd', tex_name=r'p_{d}',
                         model='StaticLoad', src='p0',
                         unit='p.u.',)
        self.qd = RParam(info='reactive demand',
                         name='qd', tex_name=r'q_{d}',
                         model='StaticLoad', src='q0',
                         unit='p.u.',)
        # --- line ---
        self.ul = RParam(info='Line connection status',
                         name='ul', tex_name=r'u_{l}',
                         model='Line', src='u',
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
                      model='StaticGen', src='p',)
        self.qg = Var(info='Gen reactive power',
                      unit='p.u.',
                      name='qg', tex_name=r'q_g',
                      model='StaticGen', src='q',)
        # --- bus ---
        self.aBus = Var(info='Bus voltage angle',
                        unit='rad',
                        name='aBus', tex_name=r'\theta_{bus}',
                        model='Bus', src='a',)
        self.vBus = Var(info='Bus voltage magnitude',
                        unit='p.u.',
                        name='vBus', tex_name=r'v_{bus}',
                        model='Bus', src='v',)
        # --- power balance ---
        pb = 'Bbus@aBus + Pbusinj + Cl@pd + Csh@gsh - Cg@pg'
        self.pb = Constraint(name='pb', info='power balance',
                             e_str=pb, is_eq=True,
                             )
        self.Vc = ExpressionCalc(name='Vc',
                                 info='power balance',
                                 e_str='vBus dot exp(1j * aBus)',
                                 )
        # --- objective ---
        # self.obj = Objective(name='obj',
        #                      info='No objective', unit='$',
        #                      sense='min', e_str='(0)',)

    # TODO: we might also need to override self.om.init()?

    # TODO: seems we might need override the following methods
    # def init(self, **kwargs):
    #     """
    #     Initialize the routine optimization model.
    #     args and kwargs go to `self.om.init()` (`cvxpy.Problem.__init__()`).
    #     """
    #     raise NotImplementedError('Not implemented yet.')

    def solve(self, **kwargs):
        """
        Solve the routine optimization model.
        args and kwargs go to `self.om.prob.solve()` (`cvxpy.Problem.solve()`).
        """
        raise NotImplementedError('Not implemented yet.')

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
        raise NotImplementedError('Not implemented yet.')

    def _post_solve(self):
        """
        Post-solve calculations.
        """
        return True

    def unpack(self, **kwargs):
        """
        Unpack the results from CVXPY model.
        """
        raise NotImplementedError('Not implemented yet.')
