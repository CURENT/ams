"""
Real-time economic dispatch.
"""
import logging
from collections import OrderedDict
import numpy as np

import cvxpy as cp

from ams.core.param import RParam
from ams.core.service import ZonalSum, VarSelect, NumOp, NumOpDual
from ams.routines.dcopf import DCOPF

from ams.opt import Var, Constraint

logger = logging.getLogger(__name__)


# --- RTED e_fn callables (Phase 4.4) ---


def _rted_rbu(r):
    return r.gs @ cp.multiply(r.ug, r.pru) - r.dud == 0


def _rted_rbd(r):
    return r.gs @ cp.multiply(r.ug, r.prd) - r.ddd == 0


def _rted_rru(r):
    return cp.multiply(r.ug, r.pg + r.pru) - cp.multiply(r.ug, r.pmaxe) <= 0


def _rted_rrd(r):
    return cp.multiply(r.ug, -r.pg + r.prd) + cp.multiply(r.ug, r.pmine) <= 0


def _rted_rgu(r):
    return cp.multiply(r.ug, r.pg - r.pg0 - r.R10) <= 0


def _rted_rgd(r):
    return cp.multiply(r.ug, -r.pg + r.pg0 - r.R10) <= 0


def _rted_obj(r):
    return (r.t ** 2 * cp.sum(cp.multiply(r.c2, r.pg ** 2))
            + cp.sum(cp.multiply(r.ug, r.c0))
            + r.t * cp.sum(r.c1 @ r.pg + r.cru @ r.pru + r.crd @ r.prd))


def _esd1_obj_extra(r):
    """ESD1Base extra objective term (registered via Objective.add_term)."""
    return r.t * cp.sum(-cp.multiply(r.cesdc, r.pce) + cp.multiply(r.cesdd, r.pde))


def _rtedvis_obj_extra(r):
    """RTEDVIS extra objective term."""
    return r.t * cp.sum(cp.multiply(r.cm, r.M) + cp.multiply(r.cd, r.D))


# DGBase
def _dgb_cdgb(r):
    return r.idg @ r.pg - r.pgdg == 0


# ESD1PBase
def _esd1p_cesd(r):
    return r.ies @ r.pg + r.pce - r.pde == 0


def _esd1p_SOClb(r):
    return -r.SOC + r.SOCmin <= 0


def _esd1p_SOCub(r):
    return r.SOC - r.SOCmax <= 0


def _esd1p_SOCb(r):
    return (cp.multiply(r.En, (r.SOC - r.SOCinit))
            - r.t * cp.multiply(r.EtaC, r.pce)
            + r.t * cp.multiply(r.REtaD, r.pde)) == 0


def _esd1p_SOCr(r):
    return r.SOCend - r.SOC <= 0


# RTEDESP
def _rtedesp_zce(r):
    return cp.multiply(1 - r.ucd, r.pce) <= 0


def _rtedesp_zde(r):
    return cp.multiply(1 - r.udd, r.pde) <= 0


# ESD1Base (single-period)
def _esd1b_cdb(r):
    return r.ucd + r.udd - 1 <= 0


def _esd1b_zce1(r):
    return -r.zce + r.pce <= 0


def _esd1b_zce2(r):
    return r.zce - r.pce - r.Mb * (1 - r.ucd) <= 0


def _esd1b_zce3(r):
    return r.zce - r.Mb * r.ucd <= 0


def _esd1b_zde1(r):
    return -r.zde + r.pde <= 0


def _esd1b_zde2(r):
    return r.zde - r.pde - r.Mb * (1 - r.udd) <= 0


def _esd1b_zde3(r):
    return r.zde - r.Mb * r.udd <= 0


def _esd1b_tcdr(r):
    return (r.tdc0 > 0) * (r.tdc > r.tdc0) - r.ucd <= 0


def _esd1b_tddr(r):
    return (r.tdd0 > 0) * (r.tdd > r.tdd0) - r.udd <= 0


# VISBase
def _vis_Mub(r):
    return r.M - r.Mmax <= 0


def _vis_Dub(r):
    return r.D - r.Dmax <= 0


def _vis_Mreq(r):
    return -r.gvsg @ r.M + r.dvm == 0


def _vis_Dreq(r):
    return -r.gvsg @ r.D + r.dvd == 0


class SFRBase:
    """
    Base class for SFR components.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)
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
                             rfun=np.reshape, rargs=dict(shape=(-1,)),
                             name='dud', tex_name=r'd_{u, d}',
                             info='zonal RegUp reserve requirement',)
        self.ddd = NumOpDual(u=self.pdz, u2=self.dd, fun=np.multiply,
                             rfun=np.reshape, rargs=dict(shape=(-1,)),
                             name='ddd', tex_name=r'd_{d, d}',
                             info='zonal RegDn reserve requirement',)
        # --- SFR ---
        self.pru = Var(info='RegUp reserve',
                       unit='p.u.', name='pru', tex_name=r'p_{r,u}',
                       model='StaticGen', nonneg=True,)
        self.prd = Var(info='RegDn reserve',
                       unit='p.u.', name='prd', tex_name=r'p_{r,d}',
                       model='StaticGen', nonneg=True,)
        # NOTE: define e_str in scheduling routine
        self.rbu = Constraint(name='rbu', is_eq=True,
                              info='RegUp reserve balance',)
        self.rbd = Constraint(name='rbd', is_eq=True,
                              info='RegDn reserve balance',)
        self.rru = Constraint(name='rru', is_eq=False,
                              info='RegUp reserve source',)
        self.rrd = Constraint(name='rrd', is_eq=False,
                              info='RegDn reserve source',)
        self.rgu = Constraint(name='rgu', is_eq=False,
                              info='Gen ramping up',)
        self.rgd = Constraint(name='rgd', is_eq=False,
                              info='Gen ramping down',)


class RTEDBase:
    """
    Base class for RTED components.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)
        # --- area ---
        self.zg = RParam(info='Gen area',
                         name='zg', tex_name='z_{one,g}',
                         model='StaticGen', src='area',
                         no_parse=True)
        self.zd = RParam(info='Load area',
                         name='zd', tex_name='z_{one,d}',
                         model='StaticLoad', src='area',
                         no_parse=True)
        self.gs = ZonalSum(u=self.zg, zone='Area',
                           name='gs', tex_name=r'S_{g}',
                           info='Sum Gen vars vector in shape of area',
                           no_parse=True, sparse=True)
        self.ds = ZonalSum(u=self.zd, zone='Area',
                           name='ds', tex_name=r'S_{d}',
                           info='Sum pd vector in shape of area',
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


class RTED(SFRBase, RTEDBase, DCOPF):
    """
    DC-based real-time economic dispatch (RTED).

    RTED extends DCOPF with:

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
    - Formulations have been adjusted with interval ``config.t``, 5/60 [Hour] by default.
    - The tie-line flow related constraints are omitted in this formulation.
    - Power generation is balanced for the entire system.
    - SFR is balanced for each area.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        self.config.add(OrderedDict((('t', 5/60),
                                     )))
        self.config.add_extra("_help",
                              t="time interval in hours",
                              )
        self.config.add_extra("_tex",
                              t='T_{cfg}',
                              )

        self.type = 'DCED'

        # --- Mapping Section ---
        # Add p -> pg0 in from map
        self.map1.update({
            'pg0': ('StaticGen', 'p'),
        })
        # nothing to do with to map

        # --- Model Section (Phase 4.4: e_fn form) ---
        # --- SFR ---
        self.rbu.e_fn = _rted_rbu
        self.rbd.e_fn = _rted_rbd
        self.rru.e_fn = _rted_rru
        self.rrd.e_fn = _rted_rrd
        # Gen ramping up/down
        self.rgu.e_fn = _rted_rgu
        self.rgd.e_fn = _rted_rgd

        # --- objective ---
        self.obj.info = 'total generation and reserve cost'
        self.obj.e_fn = _rted_obj

    def dc2ac(self, kloss=1.0, **kwargs):
        exec_time = self.exec_time
        if self.exec_time == 0 or self.exit_code != 0:
            logger.warning(f'{self.class_name} is not executed successfully, quit conversion.')
            return False
        # set pru and prd into pmin and pmax
        pr_idx = self.pru.get_all_idxes()
        pmin0 = self.system.StaticGen.get(src='pmin', attr='v', idx=pr_idx)
        pmax0 = self.system.StaticGen.get(src='pmax', attr='v', idx=pr_idx)
        p00 = self.system.StaticGen.get(src='p0', attr='v', idx=pr_idx)

        # --- ACOPF ---
        # scale up load
        pq_idx = self.system.StaticLoad.get_all_idxes()
        pd0 = self.system.StaticLoad.get(src='p0', attr='v', idx=pq_idx).copy()
        qd0 = self.system.StaticLoad.get(src='q0', attr='v', idx=pq_idx).copy()
        self.system.StaticLoad.set(src='p0', idx=pq_idx, attr='v', value=pd0 * kloss)
        self.system.StaticLoad.set(src='q0', idx=pq_idx, attr='v', value=qd0 * kloss)
        # preserve generator reserve
        ACOPF = self.system.ACOPF
        pmin = pmin0 + self.prd.v
        pmax = pmax0 - self.pru.v
        self.system.StaticGen.set(src='pmin', idx=pr_idx, attr='v', value=pmin)
        self.system.StaticGen.set(src='pmax', idx=pr_idx, attr='v', value=pmax)
        self.system.StaticGen.set(src='p0', idx=pr_idx, attr='v', value=self.pg.v)
        # run ACOPF
        ACOPF.run()
        # scale load back
        self.system.StaticLoad.set(src='p0', idx=pq_idx, attr='v', value=pd0)
        self.system.StaticLoad.set(src='q0', idx=pq_idx, attr='v', value=qd0)
        if not ACOPF.exit_code == 0:
            logger.warning('<ACOPF> did not converge, conversion failed.')
            self.vBus.optz.value = np.ones(self.system.Bus.n)
            self.aBus.optz.value = np.zeros(self.system.Bus.n)
            return False

        self.pg.optz.value = ACOPF.pg.v
        self.vBus.optz.value = ACOPF.vBus.v
        self.aBus.optz.value = ACOPF.aBus.v
        self.exec_time = exec_time

        # reset pmin, pmax, p0
        self.system.StaticGen.set(src='pmin', idx=pr_idx, attr='v', value=pmin0)
        self.system.StaticGen.set(src='pmax', idx=pr_idx, attr='v', value=pmax0)
        self.system.StaticGen.set(src='p0', idx=pr_idx, attr='v', value=p00)

        # --- set status ---
        self.system.recent = self
        self.converted = True
        logger.warning(f'<{self.class_name}> converted to AC.')
        return True


class DGBase:
    """
    Base class for DG components.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        # --- params ---
        self.gendg = RParam(info='gen of DG',
                            name='gendg', tex_name=r'g_{DG}',
                            model='DG', src='gen',
                            no_parse=True,)
        info = 'Ratio of DG.pge w.r.t to that of static generator'
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
        self.idg = VarSelect(u=self.pg, indexer='gendg',
                             name='idg', tex_name=r'I_{DG}',
                             info='Index DG power from pg',
                             gamma='gammapdg',
                             no_parse=True, sparse=True,)
        self.cdgb = Constraint(name='cdgb',
                               info='Select DG power from pg',
                               e_fn=_dgb_cdgb,)


class RTEDDG(DGBase, RTED):
    """
    RTED with distributed generator :ref:`DG`.

    Note that RTEDDG only includes DG output power. If ESD1 is included,
    RTEDES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)


class ESD1PBase:
    """
    Base class for ESD1 price run components.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        # --- params ---
        self.En = RParam(info='Rated energy capacity',
                         name='En', src='En',
                         tex_name='E_n', unit='MWh',
                         model='ESD1', no_parse=True,)
        self.SOCmax = RParam(info='Maximum allowed value for SOC in limiter',
                             name='SOCmax', src='SOCmax',
                             tex_name=r'SOC_{max}',
                             model='ESD1',)
        self.SOCmin = RParam(info='Minimum required value for SOC in limiter',
                             name='SOCmin', src='SOCmin',
                             tex_name=r'SOC_{min}',
                             model='ESD1',)
        self.SOCinit = RParam(info='Initial SOC',
                              name='SOCinit', src='SOCinit',
                              tex_name=r'SOC_{init}',
                              model='ESD1',)
        self.SOCend = RParam(info='Target SOC at the end of the period',
                             name='SOCend', src='SOCend',
                             tex_name=r'SOC_{end}',
                             model='ESD1',)
        self.EtaC = RParam(info='Efficiency during charging',
                           name='EtaC', src='EtaC',
                           tex_name=r'\eta_c',
                           model='ESD1', no_parse=True,)
        self.EtaD = RParam(info='Efficiency during discharging',
                           name='EtaD', src='EtaD',
                           tex_name=r'\eta_d',
                           model='ESD1', no_parse=True,)

        self.cesdc = RParam(info='Charging cost',
                            name='cesdc', src='cesdc',
                            tex_name=r'c_{c,ESD}', unit=r'$/p.u.*h',
                            model='ESD1', no_parse=True,)
        self.cesdd = RParam(info='Discharging cost',
                            name='cesdd', src='cesdd',
                            tex_name=r'c_{d,ESD}', unit=r'$/p.u.*h',
                            model='ESD1', no_parse=True,)

        # --- service ---
        self.REtaD = NumOp(name='REtaD', tex_name=r'\frac{1}{\eta_d}',
                           u=self.EtaD, fun=np.reciprocal,)
        self.Mb = NumOp(info='10 times of max of pmax as big M',
                        name='Mb', tex_name=r'M_{big}',
                        u=self.pmax, fun=np.max,
                        rfun=np.dot, rargs=dict(b=10),
                        array_out=False,)

        # --- vars ---
        self.SOC = Var(info='ESD1 State of Charge', unit='p.u. (%)',
                       name='SOC', tex_name=r'SOC',
                       model='ESD1', nonneg=True,
                       v0=self.SOCinit,)

        self.pce = Var(info='ESD1 charging power',
                       unit='p.u.', name='pce',
                       tex_name=r'p_{c,ESD}',
                       model='ESD1', nonneg=True,)
        self.pde = Var(info='ESD1 discharging power',
                       unit='p.u.', name='pde',
                       tex_name=r'p_{d,ESD}',
                       model='ESD1', nonneg=True,)

        self.genesd = RParam(info='gen of ESD',
                             name='genesd', tex_name=r'g_{ESD}',
                             model='ESD1', src='gen',
                             no_parse=True,)
        self.ies = VarSelect(u=self.pg, indexer='genesd',
                             name='ies', tex_name=r'I_{ESD}',
                             info='Index ESD from StaticGen',
                             no_parse=True)
        self.cesd = Constraint(name='cesd',
                               info='Select pce and pde from pg',
                               e_fn=_esd1p_cesd,)

        self.SOClb = Constraint(name='SOClb',
                                info='SOC lower bound', e_fn=_esd1p_SOClb,)
        self.SOCub = Constraint(name='SOCub',
                                info='SOC upper bound', e_fn=_esd1p_SOCub,)

        self.SOCb = Constraint(name='SOCb',
                               info='ESD1 SOC balance', e_fn=_esd1p_SOCb,)

        self.SOCr = Constraint(name='SOCr',
                               info='ESD1 final SOC requirement',
                               e_fn=_esd1p_SOCr,)

        self.obj.add_term(_esd1_obj_extra)

    def _data_check(self):
        """
        Special data check for ESD1 included routines.
        """
        logger.info(f"Entering supplemental data check for <{self.class_name}>")

        # --- GCost correction ---
        sys = self.system
        gcost_idx_esd1 = sys.GCost.find_idx(keys='gen', values=sys.ESD1.gen.v)
        c2 = sys.GCost.get(src='c2', attr='v', idx=gcost_idx_esd1)
        c1 = sys.GCost.get(src='c1', attr='v', idx=gcost_idx_esd1)
        if not (c2 == 0).all() or not (c1 == 0).all():
            for param in ['c2', 'c1']:
                sys.GCost.set(src=param, attr='v', value=0, idx=gcost_idx_esd1)
            logger.info('Parameters c2, c1 are set to 0 as they are associated with ESD1 for'
                        f' following GCost: {", ".join(gcost_idx_esd1)}')

        # --- ESD1 initial charging/discharging time ---
        judge = sys.ESD1.tdc0.v * sys.ESD1.tdd0.v > 0
        if any(judge):
            uid = np.where(judge)[0]
            idx = [sys.ESD1.idx.v[i] for i in uid]
            logger.error(f'tdc0 and tdd0 should not be both positive! Check ESD1: {", ".join(idx)}')
            return False
        return super()._data_check()


class RTEDESP(ESD1PBase, RTEDDG):
    """
    Price run of RTED with energy storage :ref:`ESD1`.

    This routine is not intended to work standalone. It should be used after solved
    :class:`RTEDES`.

    The binary variables ``ucd`` and ``udd`` are now parameters retrieved from
    solved :class:`RTEDES`.

    The constraints ``zce1`` - ``zce3`` and ``zde1`` - ``zde3`` are now simplified
    to ``zce`` and ``zde`` as below:

    .. math::

        (1 - u_{cd}) * p_{ce} <= 0
        (1 - u_{dd}) * p_{de} <= 0
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        self.ucd = RParam(info='Retrieved ESD1 charging decision',
                          name='ucd', src='ucd0',
                          tex_name=r'u_{c,ESD}',
                          model='ESD1', no_parse=True,
                          )
        self.udd = RParam(info='Retrieved ESD1 discharging decision',
                          name='udd', src='udd0',
                          tex_name=r'u_{d,ESD}',
                          model='ESD1', no_parse=True,)

        self.zce = Constraint(name='zce', info='zce bound', e_fn=_rtedesp_zce,)

        self.zde = Constraint(name='zde', info='zde bound', e_fn=_rtedesp_zde,)

    def _preinit(self):
        """
        Extra run at the beginning of RTEDESP.init().
        """
        if not self.system.RTEDES.converged:
            raise ValueError('<RTEDES> must be solved before <RTEDESP>!')
        self._used_rtn = self.system.RTEDES

    def init(self, **kwargs):
        self._preinit()
        esd1_idx = self.system.ESD1.idx.v
        esd1_stg = self.system.ESD1.get(src='gen', attr='v', idx=esd1_idx)

        self.system.ESD1.set(src='ucd0', attr='v', idx=esd1_idx,
                             value=self._used_rtn.get(src='ucd', attr='v', idx=esd1_idx))
        self.system.ESD1.set(src='udd0', attr='v', idx=esd1_idx,
                             value=self._used_rtn.get(src='udd', attr='v', idx=esd1_idx))

        pce = self._used_rtn.get(src='pce', attr='v', idx=esd1_idx)
        pde = self._used_rtn.get(src='pde', attr='v', idx=esd1_idx)
        self.system.StaticGen.set(src='p0', attr='v', idx=esd1_stg, value=pde - pce)
        logger.info(f'<{self.class_name}>: ESD1 associated StaticGen.p0 has been set'
                    f' using the values from {self._used_rtn.class_name}.pg.v')
        return super().init(**kwargs)


class ESD1Base(DGBase, ESD1PBase):
    """
    Base class for ESD1 components.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        # --- params ---
        self.tdc = RParam(info='Minimum charging duration',
                          name='tdc', src='tdc',
                          tex_name=r't_{dc}', unit='h',
                          model='ESD1', no_parse=True,)
        self.tdd = RParam(info='Minimum discharging duration',
                          name='tdd', src='tdd',
                          tex_name=r't_{dd}', unit='h',
                          model='ESD1', no_parse=True,)

        self.tdc0 = RParam(info='Initial charging time',
                           name='tdc0', src='tdc0',
                           tex_name=r't_{dc0}', unit='h',
                           model='ESD1', no_parse=True,
                           nonneg=True)
        self.tdd0 = RParam(info='Initial discharging time',
                           name='tdd0', src='tdd0',
                           tex_name=r't_{dd0}', unit='h',
                           model='ESD1', no_parse=True,
                           nonneg=True)

        self.ucd = Var(info='ESD1 charging decision',
                       name='ucd', tex_name=r'u_{c,ESD}',
                       model='ESD1', boolean=True,)
        self.udd = Var(info='ESD1 discharging decision',
                       name='udd', tex_name=r'u_{d,ESD}',
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
        self.cdb = Constraint(name='cdb', info='Charging decision bound',
                              e_fn=_esd1b_cdb,)

        self.zce1 = Constraint(name='zce1', info='zce bound 1', e_fn=_esd1b_zce1,)
        self.zce2 = Constraint(name='zce2', info='zce bound 2', e_fn=_esd1b_zce2,)
        self.zce3 = Constraint(name='zce3', info='zce bound 3', e_fn=_esd1b_zce3,)

        self.zde1 = Constraint(name='zde1', info='zde bound 1', e_fn=_esd1b_zde1,)
        self.zde2 = Constraint(name='zde2', info='zde bound 2', e_fn=_esd1b_zde2,)
        self.zde3 = Constraint(name='zde3', info='zde bound 3', e_fn=_esd1b_zde3,)

        # force charging flag `fcd`: (tdc0 > 0) * (tdc > tdc0)
        self.tcdr = Constraint(name='tcdr',
                               info='Minimum charging duration',
                               e_fn=_esd1b_tcdr,)

        # force discharging flag `fdd`: (tdd0 > 0) * (tdd > tdd0)
        self.tddr = Constraint(name='tddr',
                               info='Minimum discharging duration',
                               e_fn=_esd1b_tddr,)


class RTEDES(ESD1Base, RTED):
    """
    RTED with energy storage :ref:`ESD1`.
    The bilinear term in the formulation is linearized with big-M method.

    While the formulation enforces SOCend, the ESD1 owner is not required to provide
    an SOC constraint for every RTED interval. The optimization treats SOCend
    as a terminal boundary condition, allowing the dispatcher maximum flexibility to optimize
    power output within the hour, provided the target is met at the interval's conclusion.

    The minimum charging/discharging duration logic is implemented in `tcdr` and `tddr`.
    For example, the logic of `tcdr` is:
    `u_{cd} >= fcd`, where `fcd = 1` if `tdc0 > 0` and `tdc > tdc0`, else `fcd = 0`.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)


class VISBase:
    """
    Base class for virtual inertia scheduling.
    """

    def __init__(self, system, config, **kwargs) -> None:
        super().__init__(system, config, **kwargs)

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
                     model='VSG', src='M',
                     nonneg=True,)
        self.D = Var(info='Emulated damping coefficient',
                     name='D', tex_name=r'D', unit='p.u.',
                     model='VSG', src='D',
                     nonneg=True,)

        self.gvsg = ZonalSum(u=self.zvsg, zone='Area',
                             name='gvsg', tex_name=r'S_{g}',
                             info='Sum VSG vars vector in shape of area',
                             no_parse=True)
        self.Mub = Constraint(name='Mub', info='M upper bound', e_fn=_vis_Mub,)
        self.Dub = Constraint(name='Dub', info='D upper bound', e_fn=_vis_Dub,)
        self.Mreq = Constraint(name='Mreq',
                               info='Emulated inertia requirement',
                               e_fn=_vis_Mreq,)
        self.Dreq = Constraint(name='Dreq',
                               info='Emulated damping requirement',
                               e_fn=_vis_Dreq,)

        # NOTE: revise the objective function to include virtual inertia cost


class RTEDVIS(VISBase, RTED):
    """
    RTED with virtual inertia scheduling.

    This class implements real-time economic dispatch with virtual inertia scheduling.
    Please ensure that the parameters `dvm` and `dvd` are set according to the system base.

    References
    ----------
    1. B. She, F. Li, H. Cui, J. Wang, Q. Zhang and R. Bo, "Virtual Inertia Scheduling (VIS) for
       Real-Time Economic Dispatch of IBR-Penetrated Power Systems," in IEEE Transactions on
       Sustainable Energy, vol. 15, no. 2, pp. 938-951, April 2024, doi: 10.1109/TSTE.2023.3319307.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        # --- objective ---
        self.obj.info = 'total generation and reserve cost'
        self.obj.add_term(_rtedvis_obj_extra)

        self.map2.update({
            'M': ('RenGen', 'M'),
            'D': ('RenGen', 'D'),
        })
