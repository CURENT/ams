"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

from ams.core.param import RParam
from ams.core.service import ZonalSum, VarSelect, NumOp, NumOpDual
from ams.routines.dcopf import DCOPF

from ams.opt.omodel import Var, Constraint

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
                           info='Sum Gen vars vector in shape of zone',
                           no_parse=True, sparse=True)
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

    - Mapping dicts to interface with ANDES
    - Function ``dc2ac`` to do the AC conversion
    - Vars for SFR reserve: ``pru`` and ``prd``
    - Param for linear SFR cost: ``cru`` and ``crd``
    - Param for SFR requirement: ``du`` and ``dd``
    - Param for ramping: start point ``pg0`` and ramping limit ``R10``
    - Param ``pg0``, which can be retrieved from dynamic simulation results.

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
        self.config.add_extra("_tex",
                              t='T_{cfg}',
                              )

        self.info = 'Real-time economic dispatch'
        self.type = 'DCED'

        # --- Mapping Section ---
        # --- from map ---
        self.map1.update({
            'ug': ('StaticGen', 'u'),
            'pg0': ('StaticGen', 'p'),
        })
        # --- to map ---
        self.map2.update({
            'vBus': ('Bus', 'v0'),
            'ug': ('StaticGen', 'u'),
            'pg': ('StaticGen', 'p0'),
        })

        # --- Model Section ---
        # --- SFR ---
        # RegUp/Dn reserve balance
        self.rbu.e_str = 'gs @ mul(ug, pru) - dud'
        self.rbd.e_str = 'gs @ mul(ug, prd) - ddd'
        # RegUp/Dn reserve source
        self.rru.e_str = 'mul(ug, pg + pru) - mul(ug, pmax)'
        self.rrd.e_str = 'mul(ug, -pg + prd) + mul(ug, pmin)'
        # Gen ramping up/down
        self.rgu.e_str = 'mul(ug, pg-pg0-R10)'
        self.rgd.e_str = 'mul(ug, -pg+pg0-R10)'

        # --- objective ---
        self.obj.info = 'total generation and reserve cost'
        # NOTE: the product involved t should use ``dot``
        cost = 't**2 dot sum(mul(c2, pg**2)) + sum(ug * c0)'
        _to_sum = 'c1 @ pg + cru * pru + crd * prd'
        cost += f'+ t dot sum({_to_sum})'
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
        if not ACOPF.exit_code == 0:
            logger.warning('<ACOPF> did not converge, conversion failed.')
            # NOTE: mock results to fit interface with ANDES
            self.vBus = ACOPF.vBus
            self.vBus.optz.value = np.ones(self.system.Bus.n)
            self.aBus = ACOPF.aBus
            self.aBus.optz.value = np.zeros(self.system.Bus.n)
            return False
        self.pg.v = ACOPF.pg.v

        # NOTE: mock results to fit interface with ANDES
        self.vBus = ACOPF.vBus
        self.aBus = ACOPF.aBus

        # reset pmin, pmax, p0
        self.system.StaticGen.set(src='pmin', attr='v', idx=pr_idx, value=pmin0)
        self.system.StaticGen.set(src='pmax', attr='v', idx=pr_idx, value=pmax0)
        self.system.StaticGen.set(src='p0', attr='v', idx=pr_idx, value=p00)
        self.system.recent = self

        self.is_ac = True
        logger.warning(f'<{self.class_name}> is converted to AC.')
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


class DGBase:
    """
    Base class for DG used in DCED.
    """

    def __init__(self):
        # --- params ---
        self.gendg = RParam(info='gen of DG',
                            name='gendg', tex_name=r'g_{DG}',
                            model='DG', src='gen',
                            no_parse=True,)
        info = 'Ratio of DG.pge w.r.t to that of static generator',
        self.gammapdg = RParam(name='gammapd', tex_name=r'\gamma_{p,DG}',
                               model='DG', src='gammap',
                               no_parse=True, info=info)

        # --- vars ---
        # TODO: maybe there will be constraints on pgd, maybe upper/lower bound?
        # TODO: this might requre new device like DGSlot
        self.pgdg = Var(info='DG output power',
                        unit='p.u.', name='pgdg',
                        tex_name=r'p_{g,DG}',
                        model='DG',)

        # --- constraints ---
        self.cdg = VarSelect(u=self.pg, indexer='gendg',
                             name='cd', tex_name=r'C_{DG}',
                             info='Select DG power from pg',
                             gamma='gammapdg',
                             no_parse=True, sparse=True,)
        self.cdgb = Constraint(name='cdgb', type='eq',
                               info='Select DG power from pg',
                               e_str='cdg @ pg - pgdg',)


class RTEDDG(RTED, DGBase):
    """
    RTED with distributed generator :ref:`DG`.

    Note that RTEDDG only inlcudes DG output power. If ESD1 is included,
    RTEDES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config):
        RTED.__init__(self, system, config)
        DGBase.__init__(self)
        self.info = 'Real-time economic dispatch with DG'
        self.type = 'DCED'


class ESD1Base(DGBase):
    """
    Base class for ESD1 used in DCED.
    """

    def __init__(self):
        DGBase.__init__(self)
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
        self.genesd = RParam(info='gen of ESD1',
                             name='genesd', tex_name=r'g_{ESD}',
                             model='ESD1', src='gen',
                             no_parse=True,)
        info = 'Ratio of ESD1.pge w.r.t to that of static generator'
        self.gammapesd = RParam(name='gammapesd', tex_name=r'\gamma_{p,ESD}',
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
        self.pce = Var(info='ESD1 charging power',
                       unit='p.u.', name='pce',
                       tex_name=r'p_{c,ESD}',
                       model='ESD1', nonneg=True,)
        self.pde = Var(info='ESD1 discharging power',
                       unit='p.u.', name='pde',
                       tex_name=r'p_{d,ESD}',
                       model='ESD1', nonneg=True,)
        self.uce = Var(info='ESD1 charging decision',
                       name='uce', tex_name=r'u_{c,ESD}',
                       model='ESD1', boolean=True,)
        self.ude = Var(info='ESD1 discharging decision',
                       name='ude', tex_name=r'u_{d,ESD}',
                       model='ESD1', boolean=True,)
        self.zce = Var(name='zce', tex_name=r'z_{c,ESD}',
                       model='ESD1', nonneg=True,)
        self.zce.info = 'Aux var for charging, '
        self.zce.info += ':math:`z_{c,ESD}=u_{c,ESD}*p_{c,ESD}`'
        self.zde = Var(name='zde', tex_name=r'z_{d,ESD}',
                       model='ESD1', nonneg=True,)
        self.zde.info = 'Aux var for discharging, '
        self.zde.info += ':math:`z_{d,ESD}=u_{d,ESD}*p_{d,ESD}`'

        # --- constraints ---
        self.cdb = Constraint(name='cdb', type='eq',
                              info='Charging decision bound',
                              e_str='uce + ude - 1',)
        self.ces = VarSelect(u=self.pg, indexer='genesd',
                             name='ce', tex_name=r'C_{ESD}',
                             info='Select zue from pg',
                             gamma='gammapesd', no_parse=True,)
        self.cesb = Constraint(name='cesb', type='eq',
                               info='Select ESD1 power from pg',
                               e_str='ces @ pg + zce - zde',)

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


class VISBase:
    """
    Base class for virtual inertia scheduling.
    """

    def __init__(self) -> None:
        # --- Data Section ---
        self.cm = RParam(info='Virtual inertia cost',
                         name='cm', src='cm',
                         tex_name=r'c_{m}', unit=r'$/s',
                         model='VSGCost',
                         indexer='reg', imodel='VSG')
        self.cd = RParam(info='Virtual damping cost',
                         name='cd', src='cd',
                         tex_name=r'c_{d}', unit=r'$/(p.u.)',
                         model='VSGCost',
                         indexer='reg', imodel='VSG',)
        self.zvsg = RParam(info='VSG zone',
                           name='zvsg', tex_name='z_{one,vsg}',
                           model='VSG', src='zone',
                           no_parse=True)
        self.Mmax = RParam(info='Maximum inertia emulation',
                           name='Mmax', tex_name='M_{max}',
                           model='VSG', src='Mmax',
                           unit='s',)
        self.Dmax = RParam(info='Maximum damping emulation',
                           name='Dmax', tex_name='D_{max}',
                           model='VSG', src='Dmax',
                           unit='p.u.',)
        self.dvm = RParam(info='Emulated inertia requirement',
                          name='dvm', tex_name=r'd_{v,m}',
                          unit='s',
                          model='VSGR', src='dvm',)
        self.dvd = RParam(info='Emulated damping requirement',
                          name='dvd', tex_name=r'd_{v,d}',
                          unit='p.u.',
                          model='VSGR', src='dvd',)

        # --- Model Section ---
        self.M = Var(info='Emulated startup time constant (M=2H)',
                     name='M', tex_name=r'M', unit='s',
                     model='VSG', nonneg=True,)
        self.D = Var(info='Emulated damping coefficient',
                     name='D', tex_name=r'D', unit='p.u.',
                     model='VSG', nonneg=True,)

        self.gvsg = ZonalSum(u=self.zvsg, zone='Region',
                             name='gvsg', tex_name=r'S_{g}',
                             info='Sum VSG vars vector in shape of zone',
                             no_parse=True)
        self.Mub = Constraint(name='Mub', type='uq',
                              info='M upper bound',
                              e_str='M - Mmax',)
        self.Dub = Constraint(name='Dub', type='uq',
                              info='D upper bound',
                              e_str='D - Dmax',)
        self.Mreq = Constraint(name='Mreq', type='eq',
                               info='Emulated inertia requirement',
                               e_str='-gvsg@M + dvm',)
        self.Dreq = Constraint(name='Dreq', type='eq',
                               info='Emulated damping requirement',
                               e_str='-gvsg@D + dvd',)

        # NOTE: revise the objective function to include virtual inertia cost


class RTEDVIS(RTED, VISBase):
    """
    RTED with virtual inertia scheduling.

    Reference:

    [1] B. She, F. Li, H. Cui, J. Wang, Q. Zhang and R. Bo, "Virtual
    Inertia Scheduling (VIS) for Real-time Economic Dispatch of
    IBRs-penetrated Power Systems," in IEEE Transactions on
    Sustainable Energy, doi: 10.1109/TSTE.2023.3319307.
    """

    def __init__(self, system, config):
        RTED.__init__(self, system, config)
        VISBase.__init__(self)
        self.info = 'Real-time economic dispatch with virtual inertia scheduling'
        self.type = 'DCED'

        # --- objective ---
        self.obj.info = 'total generation and reserve cost'
        vsgcost = '+ t dot sum(cm * M + cd * D)'
        self.obj.e_str += vsgcost

        self.map2.update({
            'M': ('RenGen', 'M'),
            'D': ('RenGen', 'D'),
        })
