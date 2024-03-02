"""
Unit commitment routines.
"""
import logging
from collections import OrderedDict
import numpy as np
import pandas as pd

from ams.core.param import RParam
from ams.core.service import (NumOp, NumOpDual, MinDur)
from ams.routines.dcopf import DCOPF
from ams.routines.rted import RTEDBase
from ams.routines.ed import SRBase, MPBase, ESD1MPBase, DGBase

from ams.opt.omodel import Var, Constraint

logger = logging.getLogger(__name__)


class NSRBase:
    """
    Base class for non-spinning reserve.
    """

    def __init__(self) -> None:
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
                                name='prnsb', type='eq',)
        self.rnsr = Constraint(info='non-spinning reserve requirement',
                               name='rnsr', type='uq',)


class UC(DCOPF, RTEDBase, MPBase, SRBase, NSRBase):
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
    1. Formulations has been adjusted with interval ``config.t``

    3. The tie-line flow has not been implemented in formulations.

    References
    ----------
    1. Huang, Y., Pardalos, P. M., & Zheng, Q. P. (2017). Electrical power unit commitment: deterministic and
    two-stage stochastic programming models and algorithms. Springer.

    2. D. A. Tejada-Arango, S. Lumbreras, P. Sánchez-Martín and A. Ramos, "Which Unit-Commitment Formulation
    is Best? A Comparison Framework," in IEEE Transactions on Power Systems, vol. 35, no. 4, pp. 2926-2936,
    July 2020, doi: 10.1109/TPWRS.2019.2962024.
    """

    def __init__(self, system, config):
        DCOPF.__init__(self, system, config)
        RTEDBase.__init__(self)
        MPBase.__init__(self)
        SRBase.__init__(self)
        NSRBase.__init__(self)

        self.config.add(OrderedDict((('t', 1),
                                     )))
        self.config.add_extra("_help",
                              t="time interval in hours",
                              )
        self.config.add_extra("_tex",
                              t='T_{cfg}',
                              )

        self.info = 'unit commitment'
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

        # --- Model Section ---
        # --- gen ---
        # NOTE: extend pg to 2D matrix, where row for gen and col for timeslot
        self.pg.horizon = self.timeslot
        self.pg.info = '2D Gen power'
        # TODO: havn't test non-controllability?
        self.ctrle.u2 = self.tlv
        self.ctrle.info = 'Reshaped controllability'
        self.nctrle.u2 = self.tlv
        self.nctrle.info = 'Reshaped non-controllability'
        pglb = '-pg + mul(mul(nctrl, pg0), ugd)'
        pglb += '+ mul(mul(ctrl, pmin), ugd)'
        self.pglb.e_str = pglb
        pgub = 'pg - mul(mul(nctrl, pg0), ugd)'
        pgub += '- mul(mul(ctrl, pmax), ugd)'
        self.pgub.e_str = pgub

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
        self.actv = Constraint(name='actv', type='eq',
                               info='startup action',
                               e_str='ugd @ Mr - vgd[:, 1:]',)
        self.actv0 = Constraint(name='actv0', type='eq',
                                info='initial startup action',
                                e_str='ugd[:, 0] - ug[:, 0]  - vgd[:, 0]',)
        self.actw = Constraint(name='actw', type='eq',
                               info='shutdown action',
                               e_str='-ugd @ Mr - wgd[:, 1:]',)
        self.actw0 = Constraint(name='actw0', type='eq',
                                info='initial shutdown action',
                                e_str='-ugd[:, 0] + ug[:, 0] - wgd[:, 0]',)

        self.prs.horizon = self.timeslot
        self.prs.info = '2D Spinning reserve'

        self.prns.horizon = self.timeslot
        self.prns.info = '2D Non-spinning reserve'

        # spinning reserve
        self.prsb.e_str = 'mul(ugd, mul(pmax, tlv)) - zug - prs'
        # spinning reserve requirement
        self.rsr.e_str = '-gs@prs + dsr'

        # non-spinning reserve
        self.prnsb.e_str = 'mul(1-ugd, mul(pmax, tlv)) - prns'
        # non-spinning reserve requirement
        self.rnsr.e_str = '-gs@prns + dnsr'

        # --- big M for ugd*pg ---
        self.Mzug = NumOp(info='10 times of max of pmax as big M for zug',
                          name='Mzug', tex_name=r'M_{zug}',
                          u=self.pmax, fun=np.max,
                          rfun=np.dot, rargs=dict(b=10),
                          array_out=False,)
        self.zuglb = Constraint(name='zuglb', info='zug lower bound',
                                type='uq', e_str='- zug + pg')
        self.zugub = Constraint(name='zugub', info='zug upper bound',
                                type='uq', e_str='zug - pg - Mzug dot (1 - ugd)')
        self.zugub2 = Constraint(name='zugub2', info='zug upper bound',
                                 type='uq', e_str='zug - Mzug dot ugd')

        # --- minimum ON/OFF duration ---
        self.Con = MinDur(u=self.pg, u2=self.td1,
                          name='Con', tex_name=r'T_{on}',
                          info='minimum ON coefficient',)
        self.don = Constraint(info='minimum online duration',
                              name='don', type='uq',
                              e_str='multiply(Con, vgd) - ugd')
        self.Coff = MinDur(u=self.pg, u2=self.td2,
                           name='Coff', tex_name=r'T_{off}',
                           info='minimum OFF coefficient',)
        self.doff = Constraint(info='minimum offline duration',
                               name='doff', type='uq',
                               e_str='multiply(Coff, wgd) - (1 - ugd)')

        # --- line ---
        self.plf.horizon = self.timeslot
        self.plf.info = '2D Line flow'
        self.plflb.e_str = '-Bf@aBus - Pfinj - mul(rate_a, tlv)'
        self.plfub.e_str = 'Bf@aBus + Pfinj - mul(rate_a, tlv)'
        self.alflb.e_str = '-CftT@aBus - amax@tlv'
        self.alfub.e_str = 'CftT@aBus - amax@tlv'

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
                                 name='pdumax', type='uq',
                                 e_str='pdu - mul(pdsp, dctrl@tlv)')
        # --- power balance ---
        # NOTE: nodal balance is also contributed by unserved load
        pb = 'Bbus@aBus + Pbusinj@tlv + Cl@(pds-pdu) + Csh@gsh@tlv - Cg@pg'
        self.pb.e_str = pb

        # --- objective ---
        cost = 't**2 dot sum(c2 @ zug**2 + t dot c1 @ zug)'
        cost += '+ sum(mul(ug, c0) @ tlv)'
        _to_sum = 'csu * vgd + csd * wgd + csr @ prs + cnsr @ prns + cdp @ pdu'
        cost += f' + t dot sum({_to_sum})'
        self.obj.e_str = cost

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
        gen['zone'] = self.system.PV.get(src='zone', attr='v', idx=gen['idx'])
        gcost_idx = self.system.GCost.find_idx(keys='gen', values=gen['idx'])
        gen['c2'] = self.system.GCost.get(src='c2', attr='v', idx=gcost_idx)
        gen['c1'] = self.system.GCost.get(src='c1', attr='v', idx=gcost_idx)
        gen['c0'] = self.system.GCost.get(src='c0', attr='v', idx=gcost_idx)
        gen['wsum'] = 0.8*gen['pmax'] + 0.05*gen['c2'] + 0.1*gen['c1'] + 0.05*gen['c0']
        gen = gen.sort_values(by='wsum', ascending=True)

        # Turn off 30% of the generators as initial guess
        priority = gen['idx'].values
        g_idx = priority[0:int(0.3*len(priority))]
        ug0 = np.zeros_like(g_idx)
        # NOTE: if number of generators is too small, turn off the first one
        if len(g_idx) == 0:
            g_idx = priority[0]
            ug0 = 0
        self.system.StaticGen.set(src='u', attr='v', idx=g_idx, value=ug0)
        logger.warning(f'Turn off StaticGen {g_idx} as initial commitment guess.')
        return True

    def _post_solve(self):
        """
        Overwrite ``_post_solve``.
        """
        # --- post-solving calculations ---
        # line flow: Bf@aBus + Pfinj
        mats = self.system.mats
        self.plf.optz.value = mats.Bf._v@self.aBus.v + self.Pfinj.v@self.tlv.v
        return True

    def init(self, **kwargs):
        self._initial_guess()
        return super().init(**kwargs)

    def dc2ac(self, **kwargs):
        """
        AC conversion ``dc2ac`` is not implemented yet for
        multi-period dispatch.
        """
        return NotImplementedError

    def unpack(self, **kwargs):
        """
        Multi-period dispatch will not unpack results from
        solver into devices.

        # TODO: unpack first period results, and allow input
        # to specify which period to unpack.
        """
        return None


class UCDG(UC, DGBase):
    """
    UC with distributed generation :ref:`DG`.

    Note that UCDG only inlcudes DG output power. If ESD1 is included,
    UCES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config):
        UC.__init__(self, system, config)
        DGBase.__init__(self)

        self.info = 'unit commitment with distributed generation'
        self.type = 'DCUC'

        # NOTE: extend vars to 2D
        self.pgdg.horizon = self.timeslot


class UCES(UC, ESD1MPBase):
    """
    UC with energy storage :ref:`ESD1`.
    """

    def __init__(self, system, config):
        UC.__init__(self, system, config)
        ESD1MPBase.__init__(self)

        self.info = 'unit commitment with energy storage'
        self.type = 'DCUC'

        self.SOC.horizon = self.timeslot
        self.pce.horizon = self.timeslot
        self.pde.horizon = self.timeslot
        self.uce.horizon = self.timeslot
        self.ude.horizon = self.timeslot
        self.zce.horizon = self.timeslot
        self.zde.horizon = self.timeslot
