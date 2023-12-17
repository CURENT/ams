"""
Real-time economic dispatch.
"""
import logging  # NOQA
from collections import OrderedDict  # NOQA
import numpy as np  # NOQA

from ams.core.param import RParam  # NOQA
from ams.core.service import ZonalSum, VarSelect, NumOp, NumOpDual  # NOQA
from ams.routines.dcopf import DCOPF  # NOQA

from ams.opt.omodel import Var, Constraint  # NOQA

logger = logging.getLogger(__name__)


class RTEDBase:
    """
    Base class for real-time economic dispatch (RTED).
    """

    def __init__(self):
        # --- region ---
        self.zg = RParam(info='Gen zone',
                         name='zg', tex_name='z_{one,g}',
                         model='StaticGen', src='zone',
                         no_parse=True)
        self.zd = RParam(info='Load zone',
                         name='zd', tex_name='z_{one,d}',
                         model='StaticLoad', src='zone',
                         no_parse=True)
        self.gs = ZonalSum(u=self.zg, zone='Region',
                           name='gs', tex_name=r'S_{g}',
                           info='Sum Gen vars vector in shape of zone')
        self.ds = ZonalSum(u=self.zd, zone='Region',
                           name='ds', tex_name=r'S_{d}',
                           info='Sum pd vector in shape of zone',
                           no_parse=True,)
        self.pdz = NumOpDual(u=self.ds, u2=self.pd,
                             fun=np.multiply,
                             rfun=np.sum, rargs=dict(axis=1),
                             expand_dims=0,
                             name='pdz', tex_name=r'p_{d,z}',
                             unit='p.u.', info='zonal total load',
                             no_parse=True,)
        # --- generator ---
        self.R10 = RParam(info='10-min ramp rate',
                          name='R10', tex_name=r'R_{10}',
                          model='StaticGen', src='R10',
                          unit='p.u./h',)


class SFRBase:
    """
    Base class for SFR used in DCED.
    """

    def __init__(self):
        #  --- SFR cost ---
        self.cru = RParam(info='RegUp reserve coefficient',
                          name='cru', tex_name=r'c_{r,u}',
                          model='SFRCost', src='cru',
                          indexer='gen', imodel='StaticGen',
                          unit=r'$/(p.u.)',)
        self.crd = RParam(info='RegDown reserve coefficient',
                          name='crd', tex_name=r'c_{r,d}',
                          model='SFRCost', src='crd',
                          indexer='gen', imodel='StaticGen',
                          unit=r'$/(p.u.)',)
        # --- reserve requirement ---
        self.du = RParam(info='RegUp reserve requirement in percentage',
                         name='du', tex_name=r'd_{u}',
                         model='SFR', src='du',
                         unit='%', no_parse=True,)
        self.dd = RParam(info='RegDown reserve requirement in percentage',
                         name='dd', tex_name=r'd_{d}',
                         model='SFR', src='dd',
                         unit='%', no_parse=True,)
        self.dud = NumOpDual(u=self.pdz, u2=self.du, fun=np.multiply,
                             rfun=np.reshape, rargs=dict(newshape=(-1,)),
                             name='dud', tex_name=r'd_{u, d}',
                             info='zonal RegUp reserve requirement',)
        self.ddd = NumOpDual(u=self.pdz, u2=self.dd, fun=np.multiply,
                             rfun=np.reshape, rargs=dict(newshape=(-1,)),
                             name='ddd', tex_name=r'd_{d, d}',
                             info='zonal RegDn reserve requirement',)
        # --- SFR ---
        self.pru = Var(info='RegUp reserve',
                       unit='p.u.', name='pru', tex_name=r'p_{r,u}',
                       model='StaticGen', nonneg=True,)
        self.prd = Var(info='RegDn reserve',
                       unit='p.u.', name='prd', tex_name=r'p_{r,d}',
                       model='StaticGen', nonneg=True,)
        # NOTE: define e_str in dispatch routine
        self.rbu = Constraint(name='rbu', type='eq',
                              info='RegUp reserve balance',)
        self.rbd = Constraint(name='rbd', type='eq',
                              info='RegDn reserve balance',)
        self.rru = Constraint(name='rru', type='uq',
                              info='RegUp reserve source',)
        self.rrd = Constraint(name='rrd', type='uq',
                              info='RegDn reserve source',)
        self.rgu = Constraint(name='rgu', type='uq',
                              info='Gen ramping up',)
        self.rgd = Constraint(name='rgd', type='uq',
                              info='Gen ramping down',)


class RTED(DCOPF, RTEDBase, SFRBase):
    """
    DC-based real-time economic dispatch (RTED).
    RTED extends DCOPF with:

    1. Param ``pg0``, which can be retrieved from dynamic simulation results.

    2. RTED has mapping dicts to interface with ANDES.

    3. RTED routine adds a function ``dc2ac`` to do the AC conversion using ACOPF

    4. Vars for zonal SFR reserve: ``pru`` and ``prd``;

    5. Param for linear cost of zonal SFR reserve ``cru`` and ``crd``;

    6. Param for SFR requirement ``du`` and ``dd``;

    7. Param for generator ramping: start point ``pg0`` and ramping limit ``R10``;

    The function ``dc2ac`` sets the ``vBus`` value from solved ACOPF.
    Without this conversion, dynamic simulation might fail due to the gap between
    DC-based dispatch results and AC-based dynamic initialization.

    Notes
    -----
    1. Formulations has been adjusted with interval ``config.t``, 5/60 [Hour] by default.

    2. The tie-line flow has not been implemented in formulations.
    """

    def __init__(self, system, config):
        DCOPF.__init__(self, system, config)
        RTEDBase.__init__(self)
        SFRBase.__init__(self)

        self.config.add(OrderedDict((('t', 5/60),
                                     )))
        self.config.add_extra("_help",
                              t="time interval in hours",
                              )

        self.map1 = OrderedDict([
            ('StaticGen', {
                'pg0': 'p',
            }),
        ])
        # NOTE: define map2
        # DC-based RTED assume bus voltage to be 1
        # here we mock the ACOPF bus voltage results to fit the interface
        self.map2 = OrderedDict([
            ('Bus', {
                'vBus': 'v0',
            }),
            ('StaticGen', {
                'pg': 'p0',
            }),
        ])
        self.info = 'Real-time economic dispatch'
        self.type = 'DCED'

        # --- Model Section ---
        # --- SFR ---
        # RegUp/Dn reserve balance
        self.rbu.e_str = 'gs @ mul(ug, pru) - dud'
        self.rbd.e_str = 'gs @ mul(ug, prd) - ddd'
        # RegUp/Dn reserve source
        self.rru.e_str = 'mul(ug, pg + pru) - mul(ug, pmax)'
        self.rrd.e_str = 'mul(ug, -pg + prd) - mul(ug, pmin)'
        # Gen ramping up/down
        self.rgu.e_str = 'mul(ug, pg-pg0-R10)'
        self.rgd.e_str='mul(ug, -pg+pg0-R10)'

        # --- objective ---
        self.obj.info = 'total generation and reserve cost'
        # NOTE: the product of dt and pg is processed using ``dot``,
        # because dt is a numnber
        cost = 'sum(mul(c2, power(pg, 2)))'
        cost += '+ sum(c1 @ (t dot pg))'
        cost += '+ ug * c0'  # constant cost
        cost += '+ sum(cru * pru + crd * prd)'  # reserve cost
        self.obj.e_str = cost

    def dc2ac(self, **kwargs):
        """
        Convert the RTED results with ACOPF.

        Overload ``dc2ac`` method.
        """
        if self.exec_time == 0 or self.exit_code != 0:
            logger.warning('RTED is not executed successfully, quit conversion.')
            return False
        # set pru and prd into pmin and pmax
        pr_idx = self.pru.get_idx()
        pmin0 = self.system.StaticGen.get(src='pmin', attr='v', idx=pr_idx)
        pmax0 = self.system.StaticGen.get(src='pmax', attr='v', idx=pr_idx)
        p00 = self.system.StaticGen.get(src='p0', attr='v', idx=pr_idx)

        # solve ACOPF
        ACOPF = self.system.ACOPF
        pmin = pmin0 + self.prd.v
        pmax = pmax0 - self.pru.v
        self.system.StaticGen.set(src='pmin', attr='v', idx=pr_idx, value=pmin)
        self.system.StaticGen.set(src='pmax', attr='v', idx=pr_idx, value=pmax)
        self.system.StaticGen.set(src='p0', attr='v', idx=pr_idx, value=self.pg.v)
        ACOPF.run()
        self.pg.v = ACOPF.pg.v

        # NOTE: mock results to fit interface with ANDES
        self.vBus = ACOPF.vBus

        # reset pmin, pmax, p0
        self.system.StaticGen.set(src='pmin', attr='v', idx=pr_idx, value=pmin0)
        self.system.StaticGen.set(src='pmax', attr='v', idx=pr_idx, value=pmax0)
        self.system.StaticGen.set(src='p0', attr='v', idx=pr_idx, value=p00)
        self.system.recent = self

        self.is_ac = True
        logger.warning('RTED is converted to AC.')
        return True

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

        Notes
        -----
        1. remove ``vBus`` if has been converted with ``dc2ac``
        """
        if self.is_ac:
            delattr(self, 'vBus')
            self.is_ac = False
        return super().run(**kwargs)


class ESD1Base:
    """
    Base class for ESD1 used in DCED.
    """

    def __init__(self):
        # --- params ---
        self.En = RParam(info='Rated energy capacity',
                         name='En', src='En',
                         tex_name='E_n', unit='MWh',
                         model='ESD1', no_parse=True,)
        self.SOCmax = RParam(info='Maximum allowed value for SOC in limiter',
                             name='SOCmax', src='SOCmax',
                             tex_name=r'SOC_{max}', unit='%',
                             model='ESD1',)
        self.SOCmin = RParam(info='Minimum required value for SOC in limiter',
                             name='SOCmin', src='SOCmin',
                             tex_name=r'SOC_{min}', unit='%',
                             model='ESD1',)
        self.SOCinit = RParam(info='Initial SOC',
                              name='SOCinit', src='SOCinit',
                              tex_name=r'SOC_{init}', unit='%',
                              model='ESD1',)
        self.EtaC = RParam(info='Efficiency during charging',
                           name='EtaC', src='EtaC',
                           tex_name=r'\eta_c', unit='%',
                           model='ESD1', no_parse=True,)
        self.EtaD = RParam(info='Efficiency during discharging',
                           name='EtaD', src='EtaD',
                           tex_name=r'\eta_d', unit='%',
                           model='ESD1', no_parse=True,)
        self.gene = RParam(info='gen of ESD1',
                           name='gene', tex_name=r'g_{E}',
                           model='ESD1', src='gen',
                           no_parse=True,)
        info = 'Ratio of ESD1.pge w.r.t to that of static generator',
        self.gammape = RParam(name='gammape', tex_name=r'\gamma_{p,e}',
                              model='ESD1', src='gammap',
                              no_parse=True, info=info)

        # --- service ---
        self.REtaD = NumOp(name='REtaD', tex_name=r'\frac{1}{\eta_d}',
                           u=self.EtaD, fun=np.reciprocal,)
        self.Mb = NumOp(info='10 times of max of pmax as big M',
                        name='Mb', tex_name=r'M_{big}',
                        u=self.pmax, fun=np.max,
                        rfun=np.dot, rargs=dict(b=10),
                        array_out=False,)

        # --- vars ---
        self.SOC = Var(info='ESD1 State of Charge', unit='%',
                       name='SOC', tex_name=r'SOC',
                       model='ESD1', pos=True,
                       v0=self.SOCinit,)
        self.SOClb = Constraint(name='SOClb', type='uq',
                                info='SOC lower bound',
                                e_str='-SOC + SOCmin',)
        self.SOCub = Constraint(name='SOCub', type='uq',
                                info='SOC upper bound',
                                e_str='SOC - SOCmax',)
        self.ce = VarSelect(u=self.pg, indexer='gene',
                            name='ce', tex_name=r'C_{E}',
                            info='Select zue from pg',
                            gamma='gammape', no_parse=True,)
        self.pce = Var(info='ESD1 charging power',
                       unit='p.u.', name='pce',
                       tex_name=r'p_{c,E}',
                       model='ESD1', nonneg=True,)
        self.pde = Var(info='ESD1 discharging power',
                       unit='p.u.', name='pde',
                       tex_name=r'p_{d,E}',
                       model='ESD1', nonneg=True,)
        self.uce = Var(info='ESD1 charging decision',
                       name='uce', tex_name=r'u_{c,E}',
                       model='ESD1', boolean=True,)
        self.ude = Var(info='ESD1 discharging decision',
                       name='ude', tex_name=r'u_{d,E}',
                       model='ESD1', boolean=True,)
        self.zce = Var(name='zce', tex_name=r'z_{c,E}',
                       model='ESD1', nonneg=True,)
        self.zce.info = 'Aux var for charging, :math:`z_{c,e}=u_{c,E}*p_{c,E}`'
        self.zde = Var(name='zde', tex_name=r'z_{d,E}',
                       model='ESD1', nonneg=True,)
        self.zde.info = 'Aux var for discharging, :math:`z_{d,e}=u_{d,E}*p_{d,E}`'

        # --- constraints ---
        self.ceb = Constraint(name='ceb', type='eq',
                              info='Charging decision bound',
                              e_str='uce + ude - 1',)
        self.cpe = Constraint(name='cpe', type='eq',
                              info='Select pce from pg',
                              e_str='ce @ pg - zce - zde',)

        self.zce1 = Constraint(name='zce1', type='uq', info='zce bound 1',
                               e_str='-zce + pce',)
        self.zce2 = Constraint(name='zce2', type='uq', info='zce bound 2',
                               e_str='zce - pce - Mb dot (1-uce)',)
        self.zce3 = Constraint(name='zce3', type='uq', info='zce bound 3',
                               e_str='zce - Mb dot uce',)

        self.zde1 = Constraint(name='zde1', type='uq', info='zde bound 1',
                               e_str='-zde + pde',)
        self.zde2 = Constraint(name='zde2', type='uq', info='zde bound 2',
                               e_str='zde - pde - Mb dot (1-ude)',)
        self.zde3 = Constraint(name='zde3', type='uq', info='zde bound 3',
                               e_str='zde - Mb dot ude',)

        SOCb = 'mul(En, (SOC - SOCinit)) - t dot mul(EtaC, zce)'
        SOCb += '+ t dot mul(REtaD, zde)'
        self.SOCb = Constraint(name='SOCb', type='eq',
                               info='ESD1 SOC balance',
                               e_str=SOCb,)


class RTEDES(RTED, ESD1Base):
    """
    RTED with energy storage :ref:`ESD1`.
    The bilinear term in the formulation is linearized with big-M method.
    """

    def __init__(self, system, config):
        RTED.__init__(self, system, config)
        ESD1Base.__init__(self)
        self.info = 'Real-time economic dispatch with energy storage'
        self.type = 'DCED'
