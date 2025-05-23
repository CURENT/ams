"""
Routines using PYPOWER.
"""
import logging
from typing import Optional, Union, Type
from collections import OrderedDict

from andes.shared import deg2rad, np
from andes.utils.misc import elapsed

from ams.io.pypower import system2ppc
from ams.core.param import RParam

from ams.opt import Var, Objective, ExpressionCalc
from ams.routines.routine import RoutineBase
from ams.shared import ppoption, runpf, runopf

logger = logging.getLogger(__name__)


class DCPF1(RoutineBase):
    """
    DC Power Flow using PYPOWER.

    This routine provides a wrapper for running DC power flow analysis using the
    PYPOWER.
    It leverages PYPOWER's internal DC power flow solver and maps results back to
    the AMS system.

    Notes
    -----
    - This class does not implement the AMS-style DC power flow formulation.
    - For detailed mathematical formulations and algorithmic details, refer to the
      MATPOWER User's Manual, section on Power Flow.
    """

    def __init__(self, system, config):
        RoutineBase.__init__(self, system, config)
        self.info = 'DC Power Flow'
        self.type = 'PF'

        self.map1 = OrderedDict()   # DCPF does not receive
        self.map2.update({
            'vBus': ('Bus', 'v0'),
            'ug': ('StaticGen', 'u'),
            'pg': ('StaticGen', 'p0'),
        })

        self.config.add(OrderedDict((('verbose', 1),
                                     ('out_all', 0),
                                     ('out_sys_sum', 1),
                                     ('out_area_sum', 0),
                                     ('out_bus', 1),
                                     ('out_branch', 1),
                                     ('out_gen', 0),
                                     ('out_all_lim', -1),
                                     ('out_v_lim', 1),
                                     ('out_line_lim', 1),
                                     ('out_pg_lim', 1),
                                     ('out_qg_lim', 1),
                                     )))
        self.config.add_extra("_help",
                              verbose="0: no progress info, 1: little, 2: lots, 3: all",
                              out_all="-1: individual flags control what prints, 0: none, 1: all",
                              out_sys_sum="print system summary",
                              out_area_sum="print area summaries",
                              out_bus="print bus detail",
                              out_branch="print branch detail",
                              out_gen="print generator detail (OUT_BUS also includes gen info)",
                              out_all_lim="-1: individual flags, 0: none, 1: binding, 2: all",
                              out_v_lim="0: don't print, 1: binding constraints only, 2: all constraints",
                              out_line_lim="0: don't print, 1: binding constraints only, 2: all constraints",
                              out_pg_lim="0: don't print, 1: binding constraints only, 2: all constraints",
                              out_qg_lim="0: don't print, 1: binding constraints only, 2: all constraints",
                              )
        self.config.add_extra("_alt",
                              verbose=(0, 1, 2, 3),
                              out_all=(-1, 0, 1),
                              out_sys_sum=(0, 1),
                              out_area_sum=(0, 1),
                              out_bus=(0, 1),
                              out_branch=(0, 1),
                              out_gen=(0, 1),
                              out_all_lim=(-1, 0, 1, 2),
                              out_v_lim=(0, 1, 2),
                              out_line_lim=(0, 1, 2),
                              out_pg_lim=(0, 1, 2),
                              out_qg_lim=(0, 1, 2),
                              )
        self.config.add_extra("_tex",
                              verbose=r'v_{erbose}',
                              out_all=r'o_{ut\_all}',
                              out_sys_sum=r'o_{ut\_sys\_sum}',
                              out_area_sum=r'o_{ut\_area\_sum}',
                              out_bus=r'o_{ut\_bus}',
                              out_branch=r'o_{ut\_branch}',
                              out_gen=r'o_{ut\_gen}',
                              out_all_lim=r'o_{ut\_all\_lim}',
                              out_v_lim=r'o_{ut\_v\_lim}',
                              out_line_lim=r'o_{ut\_line\_lim}',
                              out_pg_lim=r'o_{ut\_pg\_lim}',
                              out_qg_lim=r'o_{ut\_qg\_lim}',
                              )

        self.ug = RParam(info='Gen connection status',
                         name='ug', tex_name=r'u_{g}',
                         model='StaticGen', src='u',
                         no_parse=True)

        self.c2 = RParam(info='Gen cost coefficient 2',
                         name='c2', tex_name=r'c_{2}',
                         unit=r'$/(p.u.^2)', model='GCost',
                         indexer='gen', imodel='StaticGen',
                         nonneg=True, no_parse=True)
        self.c1 = RParam(info='Gen cost coefficient 1',
                         name='c1', tex_name=r'c_{1}',
                         unit=r'$/(p.u.)', model='GCost',
                         indexer='gen', imodel='StaticGen',)
        self.c0 = RParam(info='Gen cost coefficient 0',
                         name='c0', tex_name=r'c_{0}',
                         unit=r'$', model='GCost',
                         indexer='gen', imodel='StaticGen',
                         no_parse=True)

        # --- bus ---
        self.aBus = Var(info='bus voltage angle',
                        unit='rad',
                        name='aBus', tex_name=r'a_{Bus}',
                        model='Bus', src='a',)
        self.vBus = Var(info='Bus voltage magnitude',
                        unit='p.u.',
                        name='vBus', tex_name=r'v_{Bus}',
                        src='v', model='Bus',)
        # --- gen ---
        self.pg = Var(info='Gen active power',
                      unit='p.u.',
                      name='pg', tex_name=r'p_{g}',
                      model='StaticGen', src='p',)
        self.qg = Var(info='Gen reactive power',
                      unit='p.u.',
                      name='qg', tex_name=r'q_{g}',
                      model='StaticGen', src='q',)
        # --- line flow ---
        self.plf = Var(info='Line flow',
                       unit='p.u.',
                       name='plf', tex_name=r'p_{lf}',
                       model='Line',)
        # --- objective ---
        self.obj = Objective(name='obj',
                             info='total cost, placeholder',
                             e_str='0', unit='$',
                             sense='min',)

        # --- total cost ---
        tcost = 'sum(mul(c2, pg**2))'
        tcost += '+ sum(mul(c1, pg))'
        tcost += '+ sum(mul(ug, c0))'
        self.tcost = ExpressionCalc(info='Total cost', unit='$',
                                    model=None, src=None,
                                    e_str=tcost)

    def solve(self, **kwargs):
        """
        Solve by PYPOWER.
        """
        ppc = system2ppc(self.system)
        config = {key.upper(): value for key, value in self.config._dict.items()}
        # Enforece DC power flow
        ppopt = ppoption(PF_DC=True, **config)
        res, _ = runpf(casedata=ppc, ppopt=ppopt)
        return res

    def unpack(self, res, **kwargs):
        """
        Unpack results from PYPOWER.
        """
        system = self.system
        mva = res['baseMVA']

        # --- copy results from ppc into system algeb ---
        # --- Bus ---
        system.Bus.v.v = res['bus'][:, 7]               # voltage magnitude
        system.Bus.a.v = res['bus'][:, 8] * deg2rad     # voltage angle

        # --- PV ---
        system.PV.p.v = res['gen'][system.Slack.n:, 1] / mva        # active power
        system.PV.q.v = res['gen'][system.Slack.n:, 2] / mva        # reactive power

        # --- Slack ---
        system.Slack.p.v = res['gen'][:system.Slack.n, 1] / mva     # active power
        system.Slack.q.v = res['gen'][:system.Slack.n, 2] / mva     # reactive power

        # --- Line ---
        self.plf.optz.value = res['branch'][:, 13] / mva  # line flow

        # NOTE: In PYPOWER, branch status is not optimized and this assignment
        # typically has no effect on results. However, in some extensions (e.g., gurobi-optimods),
        # branch status may be optimized. This line ensures that the system's branch status
        # is updated to reflect the results from the solver, if applicable.

        system.Line.u.v = res['branch'][:, 10]

        # --- copy results from system algeb into routine algeb ---
        for vname, var in self.vars.items():
            owner = getattr(system, var.model)  # instance of owner, Model or Group
            if var.src is None:          # skip if no source variable is specified
                continue
            elif hasattr(owner, 'group'):   # if owner is a Model instance
                grp = getattr(system, owner.group)
                idx = grp.get_all_idxes()
            elif hasattr(owner, 'get_idx'):  # if owner is a Group instance
                idx = owner.get_all_idxes()
            else:
                msg = f"Failed to find valid source variable `{owner.class_name}.{var.src}` for "
                msg += f"{self.class_name}.{vname}, skip unpacking."
                logger.warning(msg)
                continue
            try:
                logger.debug(f"Unpacking {vname} into {owner.class_name}.{var.src}.")
                var.optz.value = owner.get(src=var.src, attr='v', idx=idx)
            except AttributeError:
                logger.debug(f"Failed to unpack {vname} into {owner.class_name}.{var.src}.")
                continue
        self.system.recent = self.system.routines[self.class_name]
        return True

    def run(self, **kwargs):
        """
        Run the DC power flow using PYPOWER.

        Returns
        -------
        bool
            True if the optimization converged successfully, False otherwise.
        """
        if not self.initialized:
            self.init()
        t0, _ = elapsed()

        # --- solve optimization ---
        t0, _ = elapsed()
        res = self.solve(**kwargs)
        self.converged = res['success']
        self.exit_code = 0 if res['success'] else 1
        _, s = elapsed(t0)
        self.exec_time = float(s.split(" ")[0])
        try:
            n_iter = res['raw']['output']['iterations']
        except Exception:
            n_iter = -1
        n_iter_str = f"{n_iter} iterations " if n_iter > 1 else f"{n_iter} iteration "
        if self.exit_code == 0:
            msg = f"<{self.class_name}> converged in {s}, "
            msg += n_iter_str + "with PYPOWER."
            logger.warning(msg)
            try:
                self.unpack(res)
            except Exception as e:
                logger.error(f"Failed to unpack results from {self.class_name}.\n{e}")
                return False
            self.system.report()
            return True
        else:
            msg = f"{self.class_name} failed to converge in {s}, "
            msg += n_iter_str + "with PYPOWER."
            logger.warning(msg)
            return False

    def _get_off_constrs(self):
        pass

    def _data_check(self, info=True, **kwargs):
        pass

    def update(self, params=None, build_mats=False, **kwargs):
        pass

    def enable(self, name):
        raise NotImplementedError

    def disable(self, name):
        raise NotImplementedError

    def _post_add_check(self):
        pass

    def addRParam(self,
                  name: str,
                  tex_name: Optional[str] = None,
                  info: Optional[str] = None,
                  src: Optional[str] = None,
                  unit: Optional[str] = None,
                  model: Optional[str] = None,
                  v: Optional[np.ndarray] = None,
                  indexer: Optional[str] = None,
                  imodel: Optional[str] = None,):
        raise NotImplementedError

    def addService(self,
                   name: str,
                   value: np.ndarray,
                   tex_name: str = None,
                   unit: str = None,
                   info: str = None,
                   vtype: Type = None,):
        raise NotImplementedError

    def addConstrs(self,
                   name: str,
                   e_str: str,
                   info: Optional[str] = None,
                   is_eq: Optional[str] = False,):
        raise NotImplementedError

    def addVars(self,
                name: str,
                model: Optional[str] = None,
                shape: Optional[Union[int, tuple]] = None,
                tex_name: Optional[str] = None,
                info: Optional[str] = None,
                src: Optional[str] = None,
                unit: Optional[str] = None,
                horizon: Optional[RParam] = None,
                nonneg: Optional[bool] = False,
                nonpos: Optional[bool] = False,
                cplx: Optional[bool] = False,
                imag: Optional[bool] = False,
                symmetric: Optional[bool] = False,
                diag: Optional[bool] = False,
                psd: Optional[bool] = False,
                nsd: Optional[bool] = False,
                hermitian: Optional[bool] = False,
                boolean: Optional[bool] = False,
                integer: Optional[bool] = False,
                pos: Optional[bool] = False,
                neg: Optional[bool] = False,):
        raise NotImplementedError


class PFlow1(DCPF1):
    """
    Power Flow using PYPOWER.

    This routine provides a wrapper for running power flow analysis using the
    PYPOWER.
    It leverages PYPOWER's internal power flow solver and maps results back to the
    AMS system.

    Notes
    -----
    - This class does not implement the AMS-style power flow formulation.
    - For detailed mathematical formulations and algorithmic details, refer to the
      MATPOWER User's Manual, section on Power Flow.
    - Fast-Decoupled (XB version) and Fast-Decoupled (BX version) algorithms are
      not fully supported yet.
    """

    def __init__(self, system, config):
        DCPF1.__init__(self, system, config)
        self.info = 'Power Flow'
        self.type = 'PF'

        # PFlow does not receive nor send
        self.map1 = OrderedDict()
        self.map2 = OrderedDict()

        self.config.add(OrderedDict((('pf_alg', 1),
                                     ('pf_tol', 1e-8),
                                     ('pf_max_it', 10),
                                     ('pf_max_it_fd', 30),
                                     ('pf_max_it_gs', 1000),
                                     ('enforce_q_lims', 0),
                                     )))
        self.config.add_extra("_help",
                              pf_alg="1: Newton, 2: Fast-Decoupled XB, 3: Fast-Decoupled BX, 4: Gauss Seidel",
                              pf_tol="termination tolerance on per unit P & Q mismatch",
                              pf_max_it="maximum number of iterations for Newton's method",
                              pf_max_it_fd="maximum number of iterations for fast decoupled method",
                              pf_max_it_gs="maximum number of iterations for Gauss-Seidel method",
                              enforce_q_lims="enforce gen reactive power limits, at expense of V magnitude",
                              )
        self.config.add_extra("_alt",
                              pf_alg=(1, 2, 3, 4),
                              pf_tol=(0.0, 1e-8),
                              pf_max_it=">1",
                              pf_max_it_fd=">1",
                              pf_max_it_gs=">1",
                              enforce_q_lims=(0, 1),
                              )

    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        config = {key.upper(): value for key, value in self.config._dict.items()}
        # Enforece AC power flow
        ppopt = ppoption(PF_DC=False, **config)
        res, _ = runpf(casedata=ppc, ppopt=ppopt)
        return res

    def run(self, **kwargs):
        """
        Run the power flow using PYPOWER.

        Returns
        -------
        bool
            True if the optimization converged successfully, False otherwise.
        """
        return super().run(**kwargs)


class DCOPF1(DCPF1):
    """
    DC optimal power flow using PYPOWER.

    This routine provides a wrapper for running DC optimal power flow analysis using
    the PYPOWER.
    It leverages PYPOWER's internal DC optimal power flow solver and maps results
    back to the AMS system.

    In PYPOWER, the ``c0`` term (the constant coefficient in the generator cost
    function) is always included in the objective, regardless of the generator's
    commitment status. See `pypower/opf_costfcn.py` for implementation details.

    Notes
    -----
    - This class does not implement the AMS-style DC optimal power flow formulation.
    - For detailed mathematical formulations and algorithmic details, refer to the
      MATPOWER User's Manual, section on Optimal Power Flow.
    - Algorithms 400, 500, 600, and 700 are not fully supported yet.
    """

    def __init__(self, system, config):
        DCPF1.__init__(self, system, config)
        self.info = 'DC Optimal Power Flow'
        self.type = 'DCED'

        self.map1 = OrderedDict()   # DCOPF does not receive
        self.map2.update({
            'vBus': ('Bus', 'v0'),
            'ug': ('StaticGen', 'u'),
            'pg': ('StaticGen', 'p0'),
        })
        self.config.add(OrderedDict((('opf_alg_dc', 200),
                                     ('opf_violation', 5e-6),
                                     ('opf_flow_lim', 0),
                                     ('opf_ignore_ang_lim', 0),
                                     ('grb_method', 1),
                                     ('grb_timelimit', float('inf')),
                                     ('grb_threads', 0),
                                     ('grb_opt', 0),
                                     ('pdipm_feastol', 0),
                                     ('pdipm_gradtol', 1e-6),
                                     ('pdipm_comptol', 1e-6),
                                     ('pdipm_costtol', 1e-6),
                                     ('pdipm_max_it', 150),
                                     ('scpdipm_red_it', 20),
                                     )))
        opf_alg_dc = "0: choose default solver based on availability, 200: PIPS, 250: PIPS-sc, "
        opf_alg_dc += "400: IPOPT, 500: CPLEX, 600: MOSEK, 700: GUROBI"
        opf_flow_lim = "qty to limit for branch flow constraints: 0 - apparent power flow (limit in MVA), "
        opf_flow_lim += "1 - active power flow (limit in MW), "
        opf_flow_lim += "2 - current magnitude (limit in MVA at 1 p.u. voltage)"
        grb_method = "0 - primal simplex, 1 - dual simplex, 2 - barrier, 3 - concurrent (LP only), "
        grb_method += "4 - deterministic concurrent (LP only)"
        pdipm_feastol = "feasibility (equality) tolerance for Primal-Dual Interior Points Methods, "
        pdipm_feastol += "set to value of OPF_VIOLATION by default"
        pdipm_gradtol = "gradient tolerance for Primal-Dual Interior Points Methods"
        pdipm_comptol = "complementary condition (inequality) tolerance for Primal-Dual Interior Points Methods"
        scpdipm_red_it = "maximum reductions per iteration for Step-Control Primal-Dual Interior Points Methods"
        self.config.add_extra("_help",
                              opf_alg_dc=opf_alg_dc,
                              opf_violation="constraint violation tolerance",
                              opf_flow_lim=opf_flow_lim,
                              opf_ignore_ang_lim="ignore angle difference limits for branches even if specified",
                              grb_method=grb_method,
                              grb_timelimit="maximum time allowed for solver (TimeLimit)",
                              grb_threads="(auto) maximum number of threads to use (Threads)",
                              grb_opt="See gurobi_options() for details",
                              pdipm_feastol=pdipm_feastol,
                              pdipm_gradtol=pdipm_gradtol,
                              pdipm_comptol=pdipm_comptol,
                              pdipm_costtol="optimality tolerance for Primal-Dual Interior Points Methods",
                              pdipm_max_it="maximum iterations for Primal-Dual Interior Points Methods",
                              scpdipm_red_it=scpdipm_red_it,
                              )
        self.config.add_extra("_alt",
                              opf_alg_dc=(0, 200, 250, 400, 500, 600, 700),
                              opf_violation=">=0",
                              opf_flow_lim=(0, 1, 2),
                              opf_ignore_ang_lim=(0, 1),
                              grb_method=(0, 1, 2, 3, 4),
                              grb_timelimit=(0, float('inf')),
                              grb_threads=(0, 1),
                              grb_opt=(0, 1),
                              pdipm_feastol=">=0",
                              pdipm_gradtol=">=0",
                              pdipm_comptol=">=0",
                              pdipm_costtol=">=0",
                              pdipm_max_it=">=0",
                              scpdipm_red_it=">=0",
                              )
        self.config.add_extra("_tex",
                              opf_alg_dc=r'o_{pf\_alg\_dc}',
                              opf_violation=r'o_{pf\_violation}',
                              opf_flow_lim=r'o_{pf\_flow\_lim}',
                              opf_ignore_ang_lim=r'o_{pf\_ignore\_ang\_lim}',
                              grb_method=r'o_{grb\_method}',
                              grb_timelimit=r'o_{grb\_timelimit}',
                              grb_threads=r'o_{grb\_threads}',
                              grb_opt=r'o_{grb\_opt}',
                              pdipm_feastol=r'o_{pdipm\_feastol}',
                              pdipm_gradtol=r'o_{pdipm\_gradtol}',
                              pdipm_comptol=r'o_{pdipm\_comptol}',
                              pdipm_costtol=r'o_{pdipm\_costtol}',
                              pdipm_max_it=r'o_{pdipm\_max\_it}',
                              scpdipm_red_it=r'o_{scpdipm\_red\_it}',
                              )

        self.obj.e_str = 'sum(c2 * pg**2 + c1 * pg + c0)'

        self.pi = Var(info='Lagrange multiplier on real power mismatch',
                      name='pi', unit='$/p.u.',
                      model='Bus', src=None,)
        self.piq = Var(info='Lagrange multiplier on reactive power mismatch',
                       name='piq', unit='$/p.u.',
                       model='Bus', src=None,)

        self.mu1 = Var(info='Kuhn-Tucker multiplier on MVA limit at bus1',
                       name='mu1', unit='$/p.u.',
                       model='Line', src=None,)
        self.mu2 = Var(info='Kuhn-Tucker multiplier on MVA limit at bus2',
                       name='mu2', unit='$/p.u.',
                       model='Line', src=None,)

    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        config = {key.upper(): value for key, value in self.config._dict.items()}
        ppopt = ppoption(PF_DC=True, **config)  # Enforce DCOPF
        res = runopf(casedata=ppc, ppopt=ppopt)
        return res

    def unpack(self, res, **kwargs):
        mva = res['baseMVA']
        self.pi.optz.value = res['bus'][:, 13] / mva
        self.piq.optz.value = res['bus'][:, 14] / mva
        self.mu1.optz.value = res['branch'][:, 17] / mva
        self.mu2.optz.value = res['branch'][:, 18] / mva
        return super().unpack(res)

    def run(self, **kwargs):
        """
        Run the DCOPF routine using PYPOWER.

        Returns
        -------
        bool
            True if the optimization converged successfully, False otherwise.
        """
        return super().run(**kwargs)


class ACOPF1(DCOPF1):
    """
    AC optimal power flow using PYPOWER.

    This routine provides a wrapper for running AC optimal power flow analysis using
    the PYPOWER.
    It leverages PYPOWER's internal AC optimal power flow solver and maps results
    back to the AMS system.

    In PYPOWER, the ``c0`` term (the constant coefficient in the generator cost
    function) is always included in the objective, regardless of the generator's
    commitment status. See `pypower/opf_costfcn.py` for implementation details.

    Notes
    -----
    - This class does not implement the AMS-style AC optimal power flow formulation.
    - For detailed mathematical formulations and algorithmic details, refer to the
      MATPOWER User's Manual, section on Optimal Power Flow.
    """

    def __init__(self, system, config):
        DCOPF1.__init__(self, system, config)
        self.info = 'AC Optimal Power Flow'
        self.type = 'ACED'

        self.map1 = OrderedDict()   # ACOPF does not receive
        self.map2.update({
            'vBus': ('Bus', 'v0'),
            'ug': ('StaticGen', 'u'),
            'pg': ('StaticGen', 'p0'),
        })

        self.config.add(OrderedDict((('opf_alg', 0),
                                     )))
        self.config.add_extra("_help",
                              opf_alg="algorithm to use for OPF: 0 - default, 580 - PIPS"
                              )
        self.config.add_extra("_alt",
                              opf_alg=(0, 580),
                              )
        self.config.add_extra("_tex",
                              opf_alg=r'o_{pf\_alg}',
                              )

    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        config = {key.upper(): value for key, value in self.config._dict.items()}
        ppopt = ppoption(PF_DC=False, **config)
        res = runopf(casedata=ppc, ppopt=ppopt)
        return res

    def run(self, **kwargs):
        """
        Run the ACOPF routine using PYPOWER.

        Returns
        -------
        bool
            True if the optimization converged successfully, False otherwise.
        """
        super().run(**kwargs)
