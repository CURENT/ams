"""
Power flow routines.
"""
import logging

from ams.opt import Var, Constraint, Expression, Objective
from ams.routines.routine import RoutineBase

from ams.core.param import RParam
from ams.core.service import VarSelect

logger = logging.getLogger(__name__)


class DCPFBase(RoutineBase):
    """
    Base class for DC power flow.
    """

    def __init__(self, system, config):
        RoutineBase.__init__(self, system, config)

        self.ug = RParam(info='Gen connection status',
                         name='ug', tex_name=r'u_{g}',
                         model='StaticGen', src='u',
                         no_parse=True)
        self.pg0 = RParam(info='Gen initial active power',
                          name='pg0', tex_name=r'p_{g, 0}',
                          unit='p.u.', model='StaticGen',
                          src='p0', no_parse=False,)
        # --- shunt ---
        self.gsh = RParam(info='shunt conductance',
                          name='gsh', tex_name=r'g_{sh}',
                          model='Shunt', src='g',
                          no_parse=True,)

        self.buss = RParam(info='Bus slack',
                           name='buss', tex_name=r'B_{us,s}',
                           model='Slack', src='bus',
                           no_parse=True,)
        # --- load ---
        self.pd = RParam(info='active demand',
                         name='pd', tex_name=r'p_{d}',
                         model='StaticLoad', src='p0',
                         unit='p.u.',)

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

        # --- generation ---
        self.pg = Var(info='Gen active power',
                      unit='p.u.',
                      name='pg', tex_name=r'p_g',
                      model='StaticGen', src='p',
                      v0=self.pg0)

        # --- bus ---
        self.vBus = Var(info='Bus voltage magnitude, placeholder',
                        unit='p.u.',
                        name='vBus', tex_name=r'v_{Bus}',
                        src='v', model='Bus',)
        self.aBus = Var(info='Bus voltage angle',
                        unit='rad',
                        name='aBus', tex_name=r'\theta_{bus}',
                        model='Bus', src='a',)

        # --- power balance ---
        pb = 'Bbus@aBus + Pbusinj + Cl@pd + Csh@gsh - Cg@pg'
        self.pb = Constraint(name='pb', info='power balance',
                             e_str=pb, is_eq=True,)

        # --- bus ---
        self.csb = VarSelect(info='select slack bus',
                             name='csb', tex_name=r'c_{sb}',
                             u=self.aBus, indexer='buss',
                             no_parse=True,)
        self.sba = Constraint(info='align slack bus angle',
                              name='sbus', is_eq=True,
                              e_str='csb@aBus',)

        # --- line flow ---
        self.plf = Expression(info='Line flow',
                              name='plf', tex_name=r'p_{lf}',
                              unit='p.u.',
                              e_str='Bf@aBus + Pfinj',
                              model='Line', src=None,)

    def solve(self, **kwargs):
        """
        Solve the routine optimization model.
        args and kwargs go to `self.om.prob.solve()` (`cvxpy.Problem.solve()`).
        """
        return self.om.prob.solve(**kwargs)

    def unpack(self, res, **kwargs):
        """
        Unpack the results from CVXPY model.
        """
        # --- solver Var results to routine algeb ---
        for _, var in self.vars.items():
            # --- copy results from routine algeb into system algeb ---
            if var.model is None:          # if no owner
                continue
            if var.src is None:            # if no source
                continue
            else:
                try:
                    idx = var.owner.get_all_idxes()
                except AttributeError:
                    idx = var.owner.idx.v

                # NOTE: only unpack the variables that are in the model or group
                try:
                    var.owner.set(src=var.src, idx=idx, attr='v', value=var.v)
                except (KeyError, TypeError):
                    logger.error(f'Failed to unpack <{var}> to <{var.owner.class_name}>.')

        # --- solver ExpressionCalc results to routine algeb ---
        for _, exprc in self.exprcs.items():
            if exprc.model is None:
                continue
            if exprc.src is None:
                continue
            else:
                try:
                    idx = exprc.owner.get_all_idxes()
                except AttributeError:
                    idx = exprc.owner.idx.v

                try:
                    exprc.owner.set(src=exprc.src, idx=idx, attr='v', value=exprc.v)
                except (KeyError, TypeError):
                    logger.error(f'Failed to unpack <{exprc}> to <{exprc.owner.class_name}>.')

        # label the most recent solved routine
        self.system.recent = self.system.routines[self.class_name]
        return True

    def _post_solve(self):
        """
        Post-solve calculations.
        """
        # NOTE: unpack Expressions if owner and arc are available
        for expr in self.exprs.values():
            if expr.owner and expr.src:
                expr.owner.set(src=expr.src, attr='v',
                               idx=expr.get_all_idxes(), value=expr.v)
        return True


class DCPF(DCPFBase):
    """
    DC power flow.
    """

    def __init__(self, system, config):
        DCPFBase.__init__(self, system, config)
        self.info = 'DC Power Flow'
        self.type = 'PF'

        self.genpv = RParam(info='gen of PV',
                            name='genpv', tex_name=r'g_{DG}',
                            model='PV', src='idx',
                            no_parse=True,)
        self.cpv = VarSelect(u=self.pg, indexer='genpv',
                             name='cpv', tex_name=r'C_{PV}',
                             info='Select PV from pg',
                             no_parse=True,)

        self.pvb = Constraint(name='pvb', info='PV generator',
                              e_str='cpv @ (pg - mul(ug, pg0))',
                              is_eq=True,)

        self.obj = Objective(name='obj',
                             info='place holder', unit='$',
                             sense='min', e_str='0',)
