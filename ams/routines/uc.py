"""
Unit commitment routines.
"""
import logging
from collections import OrderedDict
import numpy as np
import pandas as pd

import cvxpy as cp

from ams.core.param import RParam
from ams.core.service import (NumOp, NumOpDual, MinDur)
from ams.routines.dcopf import DCOPF
from ams.routines.rted import RTEDBase
from ams.routines.ed import SRBase, MPBase, ESD1MPBase, DGMPBase

from ams.opt import Var, Constraint

logger = logging.getLogger(__name__)


# --- UC e_fn callables (Phase 4.4) ---


def _uc_pmaxe(r):
    return (cp.multiply(cp.multiply(r.nctrl, r.pg0), r.ugd)
            + cp.multiply(cp.multiply(r.ctrl, r.pmax), r.ugd))


def _uc_pmine(r):
    return (cp.multiply(cp.multiply(r.ctrl, r.pmin), r.ugd)
            + cp.multiply(cp.multiply(r.nctrl, r.pg0), r.ugd))


def _uc_pglb(r):
    return -r.pg + r.pmine <= 0


def _uc_pgub(r):
    return r.pg - r.pmaxe <= 0


def _uc_actv(r):
    return r.ugd @ r.Mr - r.vgd[:, 1:] == 0


def _uc_actv0(r):
    return r.ugd[:, 0] - r.ug[:, 0] - r.vgd[:, 0] == 0


def _uc_actw(r):
    return -r.ugd @ r.Mr - r.wgd[:, 1:] == 0


def _uc_actw0(r):
    return -r.ugd[:, 0] + r.ug[:, 0] - r.wgd[:, 0] == 0


def _uc_prsb(r):
    return cp.multiply(r.ugd, cp.multiply(r.pmax, r.tlv)) - r.zug - r.prs == 0


def _uc_rsr(r):
    return -r.gs @ r.prs + r.dsr <= 0


def _uc_prnsb(r):
    return cp.multiply(1 - r.ugd, cp.multiply(r.pmax, r.tlv)) - r.prns == 0


def _uc_rnsr(r):
    return -r.gs @ r.prns + r.dnsr <= 0


def _uc_zuglb(r):
    return -r.zug + r.pg <= 0


def _uc_zugub(r):
    return r.zug - r.pg - r.Mzug * (1 - r.ugd) <= 0


def _uc_zugub2(r):
    return r.zug - r.Mzug * r.ugd <= 0


def _uc_don(r):
    return cp.multiply(r.Con, r.vgd) - r.ugd <= 0


def _uc_doff(r):
    return cp.multiply(r.Coff, r.wgd) - (1 - r.ugd) <= 0


def _uc_plflb(r):
    return -r.Bf @ r.aBus - r.Pfinj - cp.multiply(r.rate_a, r.tlv) <= 0


def _uc_plfub(r):
    return r.Bf @ r.aBus + r.Pfinj - cp.multiply(r.rate_a, r.tlv) <= 0


def _uc_alflb(r):
    return -r.CftT @ r.aBus - r.amax @ r.tlv <= 0


def _uc_alfub(r):
    return r.CftT @ r.aBus - r.amax @ r.tlv <= 0


def _uc_pdumax(r):
    return r.pdu - cp.multiply(r.pdsp, r.dctrl @ r.tlv) <= 0


def _uc_pb(r):
    return (r.Bbus @ r.aBus + r.Pbusinj @ r.tlv
            + r.Cl @ (r.pds - r.pdu) + r.Csh @ r.gsh @ r.tlv
            - r.Cg @ r.pg) == 0


def _uc_obj(r):
    return (r.t ** 2 * cp.sum(r.c2 @ r.pg ** 2)
            + r.t * cp.sum(r.c1 @ r.pg)
            + cp.sum(cp.multiply(r.ug, r.c0) @ r.tlv)
            + cp.sum(r.csu @ r.vgd + r.csd @ r.wgd)
            + r.t * cp.sum(r.csr @ r.prs + r.cnsr @ r.prns + r.cdp @ r.pdu))


class NSRBase:
    """
    Base class for non-spinning reserve components.
    """

    def __init__(self, system, config, **kwargs) -> None:
        super().__init__(system, config, **kwargs)

        self.cnsr = RParam(info='cost for non-spinning reserve',
                           name='cnsr', tex_name=r'c_{nsr}',
                           model='NSRCost', src='cnsr',
                           unit=r'$/(p.u.*h)',
                           indexer='gen', imodel='StaticGen',)
        self.dnsrp = RParam(info='non-spinning reserve requirement in percentage',
                            name='dnsr', tex_name=r'd_{nsr}',
                            model='NSR', src='demand',
                            unit='%',)
        self.prns = Var(info='non-spinning reserve',
                        name='prns', tex_name=r'p_{r, ns}',
                        model='StaticGen', nonneg=True,)

        self.dnsrpz = NumOpDual(u=self.pdz, u2=self.dnsrp, fun=np.multiply,
                                name='dnsrpz', tex_name=r'd_{nsr, p, z}',
                                info='zonal non-spinning reserve requirement in percentage',)
        self.dnsr = NumOpDual(u=self.dnsrpz, u2=self.sd, fun=np.multiply,
                              rfun=np.transpose,
                              name='dnsr', tex_name=r'd_{nsr}',
                              info='zonal non-spinning reserve requirement',
                              no_parse=True,)

        # NOTE: define e_str in dispatch model
        self.prnsb = Constraint(info='non-spinning reserve balance',
                                name='prnsb', is_eq=True,)
        self.rnsr = Constraint(info='non-spinning reserve requirement',
                               name='rnsr', is_eq=False,)


class UC(SRBase, NSRBase, MPBase, RTEDBase, DCOPF):
    """
    DC-based unit commitment (UC):
    The bilinear term in the formulation is linearized with big-M method.

    Non-negative var `pdu` is introduced as unserved load with its penalty `cdp`.

    Constraints include power balance, ramping, spinning reserve, non-spinning reserve,
    minimum ON/OFF duration.
    The cost inludes generation cost, startup cost, shutdown cost, spinning reserve cost,
    non-spinning reserve cost, and unserved load penalty.

    Method ``_initial_guess`` is used to make initial guess for commitment decision if all
    generators are online at initial. It is a simple heuristic method, which may not be optimal.

    Notes
    -----
    - The formulations have been adjusted with interval ``config.t``
    - The tie-line flow has not been implemented in formulations.

    References
    ----------
    1. Huang, Y., Pardalos, P. M., & Zheng, Q. P. (2017). Electrical power unit commitment: deterministic and
       two-stage stochastic programming models and algorithms. Springer.
    2. D. A. Tejada-Arango, S. Lumbreras, P. Sánchez-Martín and A. Ramos, "Which Unit-Commitment Formulation
       is Best? A Comparison Framework," in IEEE Transactions on Power Systems, vol. 35, no. 4, pp. 2926-2936,
       July 2020, doi: 10.1109/TPWRS.2019.2962024.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)

        self.config.add(OrderedDict((('t', 1),
                                     )))
        self.config.add_extra("_help",
                              t="time interval in hours",
                              )
        self.config.add_extra("_tex",
                              t='T_{cfg}',
                              )

        self.type = 'DCUC'

        # --- Data Section ---
        # update timeslot model to UCTSlot
        self.timeslot.info = 'Time slot for multi-period UC'
        self.timeslot.model = 'UCTSlot'
        # --- reserve cost ---
        self.csu = RParam(info='startup cost',
                          name='csu', tex_name=r'c_{su}',
                          model='GCost', src='csu',
                          unit='$',)
        self.csd = RParam(info='shutdown cost',
                          name='csd', tex_name=r'c_{sd}',
                          model='GCost', src='csd',
                          unit='$',)
        # --- load ---
        self.cdp = RParam(info='penalty for unserved load',
                          name='cdp', tex_name=r'c_{d,p}',
                          model='DCost', src='cdp',
                          no_parse=True,
                          unit=r'$/(p.u.*h)',)
        self.dctrl = RParam(info='load controllability',
                            name='dctrl', tex_name=r'c_{trl,d}',
                            model='StaticLoad', src='ctrl',
                            expand_dims=1,
                            no_parse=True,)
        # --- gen ---
        self.td1 = RParam(info='minimum ON duration',
                          name='td1', tex_name=r't_{d1}',
                          model='StaticGen', src='td1',
                          unit='h',)
        self.td2 = RParam(info='minimum OFF duration',
                          name='td2', tex_name=r't_{d2}',
                          model='StaticGen', src='td2',
                          unit='h',)

        self.sd.info = 'zonal load factor for UC'
        self.sd.model = 'UCTSlot'

        self.ug.expand_dims = 1
        self.amin.expand_dims = 1
        self.amax.expand_dims = 1

        # --- Model Section (Phase 4.4: e_fn form) ---
        # --- gen ---
        # NOTE: extend pg to 2D matrix, where row for gen and col for timeslot
        self.pg.horizon = self.timeslot
        self.pg.info = '2D Gen power'
        # TODO: havn't test non-controllability?
        self.ctrle.u2 = self.tlv
        self.ctrle.info = 'Reshaped controllability'
        self.nctrle.u2 = self.tlv
        self.nctrle.info = 'Reshaped non-controllability'
        self.pmaxe.e_fn = _uc_pmaxe
        self.pmaxe.horizon = self.timeslot
        self.pmine.e_fn = _uc_pmine
        self.pmine.horizon = self.timeslot
        self.pglb.e_fn = _uc_pglb
        self.pgub.e_fn = _uc_pgub

        self.ugd = Var(info='commitment decision',
                       horizon=self.timeslot,
                       name='ugd', tex_name=r'u_{g,d}',
                       model='StaticGen', src='u',
                       boolean=True,)
        self.vgd = Var(info='startup action',
                       horizon=self.timeslot,
                       name='vgd', tex_name=r'v_{g,d}',
                       model='StaticGen', src='u',
                       boolean=True,)
        self.wgd = Var(info='shutdown action',
                       horizon=self.timeslot,
                       name='wgd', tex_name=r'w_{g,d}',
                       model='StaticGen', src='u',
                       boolean=True,)
        self.zug = Var(info='Aux var, :math:`z_{ug} = u_{g,d} * p_g`',
                       horizon=self.timeslot,
                       name='zug', tex_name=r'z_{ug}',
                       model='StaticGen', pos=True,)
        # NOTE: actions have two parts: initial status and the rest
        self.actv = Constraint(name='actv', info='startup action', e_fn=_uc_actv,)
        self.actv0 = Constraint(name='actv0', info='initial startup action',
                                e_fn=_uc_actv0,)
        self.actw = Constraint(name='actw', info='shutdown action', e_fn=_uc_actw,)
        self.actw0 = Constraint(name='actw0', info='initial shutdown action',
                                e_fn=_uc_actw0,)

        self.prs.horizon = self.timeslot
        self.prs.info = '2D Spinning reserve'

        self.prns.horizon = self.timeslot
        self.prns.info = '2D Non-spinning reserve'

        # spinning reserve
        self.prsb.e_fn = _uc_prsb
        self.rsr.e_fn = _uc_rsr

        # non-spinning reserve
        self.prnsb.e_fn = _uc_prnsb
        self.rnsr.e_fn = _uc_rnsr

        # --- big M for ugd*pg ---
        self.Mzug = NumOp(info='10 times of max of pmax as big M for zug',
                          name='Mzug', tex_name=r'M_{zug}',
                          u=self.pmax, fun=np.max,
                          rfun=np.dot, rargs=dict(b=10),
                          array_out=False,)
        self.zuglb = Constraint(name='zuglb', info='zug lower bound', e_fn=_uc_zuglb,)
        self.zugub = Constraint(name='zugub', info='zug upper bound', e_fn=_uc_zugub,)
        self.zugub2 = Constraint(name='zugub2', info='zug upper bound', e_fn=_uc_zugub2,)

        # --- minimum ON/OFF duration ---
        self.Con = MinDur(u=self.pg, u2=self.td1,
                          name='Con', tex_name=r'T_{on}',
                          info='minimum ON coefficient',)
        self.don = Constraint(info='minimum online duration',
                              name='don', e_fn=_uc_don,)
        self.Coff = MinDur(u=self.pg, u2=self.td2,
                           name='Coff', tex_name=r'T_{off}',
                           info='minimum OFF coefficient',)
        self.doff = Constraint(info='minimum offline duration',
                               name='doff', e_fn=_uc_doff,)

        # --- line ---
        self.plf.horizon = self.timeslot
        self.plf.info = '2D Line flow'
        self.plflb.e_fn = _uc_plflb
        self.plfub.e_fn = _uc_plfub
        self.alflb.e_fn = _uc_alflb
        self.alfub.e_fn = _uc_alfub

        # --- unserved load ---
        self.pdu = Var(info='unserved demand',
                       name='pdu', tex_name=r'p_{d,u}',
                       model='StaticLoad', unit='p.u.',
                       horizon=self.timeslot,
                       nonneg=True,)
        self.pdsp = NumOp(u=self.pds, fun=np.clip,
                          args=dict(a_min=0, a_max=None),
                          info='positive demand',
                          name='pdsp', tex_name=r'p_{d,s}^{+}',)
        self.pdumax = Constraint(info='unserved demand upper bound',
                                 name='pdumax', e_fn=_uc_pdumax,)
        # --- power balance ---
        # NOTE: nodal balance is also contributed by unserved load
        self.pb.e_fn = _uc_pb

        # --- objective ---
        self.obj.e_fn = _uc_obj

    def _initial_guess(self):
        """
        Make initial guess for commitment decision with a priority list
        defined by the weighted sum of generation cost and generator capacity.

        If there are no offline generators, turn off the first 30% of the generators
        on the priority list as initial guess.
        """
        # check trigger condition
        ug0 = self.system.PV.get(src='u', attr='v', idx=self.system.PV.idx.v)
        if (ug0 == 0).any():
            return True
        else:
            logger.warning('All generators are online at initial, make initial guess for commitment.')

        gen = pd.DataFrame()
        gen['idx'] = self.system.PV.idx.v
        gen['pmax'] = self.system.PV.get(src='pmax', attr='v', idx=gen['idx'])
        gen['bus'] = self.system.PV.get(src='bus', attr='v', idx=gen['idx'])
        gen['area'] = self.system.PV.get(src='area', attr='v', idx=gen['idx'])
        gcost_idx = self.system.GCost.find_idx(keys='gen', values=gen['idx'])
        gen['c2'] = self.system.GCost.get(src='c2', attr='v', idx=gcost_idx)
        gen['c1'] = self.system.GCost.get(src='c1', attr='v', idx=gcost_idx)
        gen['c0'] = self.system.GCost.get(src='c0', attr='v', idx=gcost_idx)
        gen['wsum'] = 0.8*gen['pmax'] + 0.05*gen['c2'] + 0.1*gen['c1'] + 0.05*gen['c0']
        gen = gen.sort_values(by='wsum', ascending=True)

        # Turn off 30% of the generators as initial guess
        # NOTE: use np.array(..., dtype=object) instead of .values to avoid
        # pandas 2.2+ returning a StringArray (unhashable in ANDES idx2uid).
        priority = np.array(gen['idx'], dtype=object)
        g_idx = priority[0:int(0.3*len(priority))]
        ug0 = np.zeros_like(g_idx)
        # NOTE: if number of generators is too small, turn off the first one
        if len(g_idx) == 0:
            g_idx = priority[0]
            ug0 = 0
            off_gen = f'{g_idx}'
        else:
            off_gen = ', '.join(g_idx)
        self.system.StaticGen.set(src='u', idx=g_idx, attr='v', value=ug0)
        logger.warning(f"As initial commitment guess, turn off StaticGen: {off_gen}")
        return g_idx

    def init(self, **kwargs):
        self._initial_guess()
        return super().init(**kwargs)

    def dc2ac(self, kloss=1.0, **kwargs):
        """
        AC conversion ``dc2ac`` is not implemented yet for
        multi-period scheduling.
        """
        raise NotImplementedError("dc2ac is not implemented for multi-period scheduling")

    def unpack(self, res, **kwargs):
        """
        Multi-period scheduling will not unpack results from
        solver into devices.

        # TODO: unpack first period results, and allow input
        # to specify which period to unpack.
        """
        return None


class UCDG(DGMPBase, UC):
    """
    UC with distributed generation :ref:`DG`.

    Note that UCDG only includes DG output power. If ESD1 is included,
    UCES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)


class UCES(ESD1MPBase, UC):
    """
    UC with energy storage :ref:`ESD1`.
    """

    def __init__(self, system, config, **kwargs):
        super().__init__(system, config, **kwargs)
