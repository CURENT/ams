"""
Power flow routines independent from PYPOWER.
"""
import logging

import numpy as np
from ams.core.param import RParam
from ams.core.service import NumOp, NumOpDual, VarSelect

from ams.routines.routine import RoutineBase

from ams.opt import Var, Constraint, Objective, Expression


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
                          name='pg0', tex_name=r'p_{g, 0}',
                          unit='p.u.',
                          model='StaticGen', src='p0')
        self.qmax = RParam(info='Gen maximum reactive power',
                           name='qmax', tex_name=r'q_{g, max}',
                           unit='p.u.', model='StaticGen',
                           no_parse=False,)
        self.qmin = RParam(info='Gen minimum reactive power',
                           name='qmin', tex_name=r'q_{g, min}',
                           unit='p.u.', model='StaticGen',
                           no_parse=False,)
        self.qg0 = RParam(info='Gen initial reactive power',
                          name='qg0', tex_name=r'q_{g, 0}',
                          unit='p.u.',
                          model='StaticGen', src='q0')
        # --- bus ---
        self.buss = RParam(info='Bus slack',
                           name='buss', tex_name=r'B_{us,s}',
                           model='Slack', src='bus',
                           no_parse=True,)
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
        self.qg = Var(info='Gen reactive power',
                      unit='p.u.',
                      name='qg', tex_name=r'q_g',
                      model='StaticGen', src='q',)
        qglb = '-qg + mul(nctrle, qg0) + mul(ctrle, qmin)'
        self.qglb = Constraint(name='qglb', info='qg min',
                               e_str=qglb, is_eq=False,)
        qgub = 'qg - mul(nctrle, qg0) - mul(ctrle, qmax)'
        self.qgub = Constraint(name='qgub', info='qg max',
                               e_str=qgub, is_eq=False,)

        # --- bus ---
        self.aBus = Var(info='Bus voltage angle',
                        unit='rad',
                        name='aBus', tex_name=r'\theta_{bus}',
                        model='Bus', src='a',)
        self.vBus = Var(info='Bus voltage magnitude',
                        unit='p.u.',
                        name='vBus', tex_name=r'v_{bus}',
                        model='Bus', src='v',)
        self.csb = VarSelect(info='select slack bus',
                             name='csb', tex_name=r'c_{sb}',
                             u=self.aBus, indexer='buss',
                             no_parse=True,)
        self.sba = Constraint(info='align slack bus angle',
                              name='sbus', is_eq=True,
                              e_str='csb@aBus',)
        # --- power balance ---
        # NOTE: CVXPY does not support exp complex number
        self.Vc = Expression(info='Complex bus voltage',
                             name='Vc', tex_name=r'V_{bus}',
                             unit='p.u.', no_parse=True,
                             e_str='vBus dot exp(1j dot aBus)',
                             model='Bus', src=None,
                             vtype=complex,)
        self.pbin = Expression(info='Bus power in',
                               name='pbin', tex_name=r'p_{bus}^{in}',
                               unit='p.u.', no_parse=True,
                               e_str='-Cl@pd - Csh@gsh + Cg@pg',
                               model='Bus', src=None,)
        self.pbout = Expression(info='Bus power out',
                                name='pbout', tex_name=r'p_{bus}^{out}',
                                unit='p.u.', no_parse=True,
                                e_str='Vc @ Bbus @ conj(Vc)',
                                model='Bus', src=None)
        pb = 'Bbus@aBus + Pbusinj + Cl@pd + Csh@gsh - Cg@pg'
        self.pb = Constraint(name='pb', info='power balance',
                             e_str='pbin - pbout', is_eq=True,)
        # # --- line flow ---
        # self.plf = Var(info='Line flow',
        #                unit='p.u.',
        #                name='plf', tex_name=r'p_{lf}',
        #                model='Line',)
        # self.plflb = Constraint(info='line flow lower bound',
        #                         name='plflb', is_eq=False,
        #                         e_str='-Bf@aBus - Pfinj - mul(ul, rate_a)',)
        # self.plfub = Constraint(info='line flow upper bound',
        #                         name='plfub', is_eq=False,
        #                         e_str='Bf@aBus + Pfinj - mul(ul, rate_a)',)
        # self.alflb = Constraint(info='line angle difference lower bound',
        #                         name='alflb', is_eq=False,
        #                         e_str='-CftT@aBus + amin',)
        # self.alfub = Constraint(info='line angle difference upper bound',
        #                         name='alfub', is_eq=False,
        #                         e_str='CftT@aBus - amax',)

        # --- objective ---
        self.obj = Objective(name='obj',
                             info='No objective', unit='$',
                             sense='min', e_str='0',)

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