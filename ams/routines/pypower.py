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
                             info='total cost',
                             e_str='0',
                             sense='min',)

        # --- total cost ---
        tcost = 'sum(mul(c2, pg**2))'
        tcost += '+ sum(mul(c1, pg))'
        tcost += '+ sum(mul(ug, c0))'
        self.tcost = ExpressionCalc(info='Total cost', unit='$',
                                    model=None, src=None,
                                    e_str=tcost)

    def solve(self, OUT_ALL=0, VERBOSE=1, **kwargs):
        """
        Solve by PYPOWER.
        """
        ppc = system2ppc(self.system)
        # Enforece DC power flow
        ppopt = ppoption(PF_DC=True, OUT_ALL=OUT_ALL, VERBOSE=VERBOSE, **kwargs)
        res, success = runpf(casedata=ppc, ppopt=ppopt)
        return res

    def unpack(self, res):
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

    def run(self, OUT_ALL=0, VERBOSE=1, **kwargs):
        """
        Run the DC power flow using PYPOWER.

        This method invokes `self.solve(**kwargs)`, which internally utilizes
        `pypower.ppoption` and `pypower.runpf` to solve the DC power flow problem.

        Parameters
        ----------
        OUT_ALL : int, optional
            Controls the amount of output printed (default: 0; 0: none, 1: little,
            2: lots, 3: all). This is passed to `self.solve()`.
        VERBOSE : int, optional
            Controls the verbosity of the output (default: 1; 0: none, 1: little,
            2: lots, 3: all). This is passed to `self.solve()`.

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
    """

    def __init__(self, system, config):
        DCPF1.__init__(self, system, config)
        self.info = 'Power Flow'
        self.type = 'PF'

        self.map1 = OrderedDict()   # PFlow does not receive
        self.map2 = OrderedDict()   # PFlow does not send

    def solve(self, OUT_ALL=0, VERBOSE=1, **kwargs):
        ppc = system2ppc(self.system)
        # Enforece AC power flow
        ppopt = ppoption(PF_DC=False, OUT_ALL=OUT_ALL, VERBOSE=VERBOSE, **kwargs)
        res, success = runpf(casedata=ppc, ppopt=ppopt)
        return res

    def run(self, OUT_ALL=0, VERBOSE=1, **kwargs):
        """
        Run the power flow using PYPOWER.

        This method invokes `self.solve(**kwargs)`, which internally utilizes
        `pypower.ppoption` and `pypower.runpf` to solve the power flow problem.

        Parameters
        ----------
        OUT_ALL : int, optional
            Controls the amount of output printed (default: 0; 0: none, 1: little,
            2: lots, 3: all). This is passed to `self.solve()`.
        VERBOSE : int, optional
            Controls the verbosity of the output (default: 1; 0: none, 1: little,
            2: lots, 3: all). This is passed to `self.solve()`.

        Keyword Arguments
        -------------------
        - ``PF_ALG``
            Power flow algorithm (default: 1; 1: Newton's method, 2: Fast-Decoupled (XB version),
            3: Fast-Decoupled (BX version), 4: Gauss-Seidel).
        - ``PF_TOL``
            Termination tolerance on per unit P & Q mismatch (default: 1e-8).
        - ``PF_MAX_IT``
            Maximum number of iterations for Newton's method (default: 10).
        - ``PF_MAX_IT_FD``
            Maximum number of iterations for fast decoupled method (default: 30).
        - ``PF_MAX_IT_GS``
            Maximum number of iterations for Gauss-Seidel method (default: 1000).
        - ``ENFORCE_Q_LIMS``
            Enforce generator reactive power limits at the expense of voltage magnitude
            (default: False).

        Returns
        -------
        bool
            True if the optimization converged successfully, False otherwise.
        """
        return super().run(OUT_ALL=OUT_ALL, VERBOSE=VERBOSE, **kwargs)


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

        self.obj = Objective(name='obj',
                             info='total cost, placeholder',
                             e_str='sum(c2 * pg**2 + c1 * pg + c0)',
                             sense='min',)

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

    def solve(self, OUT_ALL=0, VERBOSE=1, OPF_ALG_DC=200, **kwargs):
        ppc = system2ppc(self.system)
        ppopt = ppoption(PF_DC=True, OUT_ALL=OUT_ALL, VERBOSE=VERBOSE,
                         OPF_ALG_DC=OPF_ALG_DC, **kwargs)
        res = runopf(casedata=ppc, ppopt=ppopt)
        return res

    def unpack(self, res):
        mva = res['baseMVA']
        self.pi.optz.value = res['bus'][:, 13] / mva
        self.piq.optz.value = res['bus'][:, 14] / mva
        self.mu1.optz.value = res['branch'][:, 17] / mva
        self.mu2.optz.value = res['branch'][:, 18] / mva
        return super().unpack(res)

    def run(self, OUT_ALL=0, VERBOSE=1, OPF_ALG_DC=200, **kwargs):
        """
        Run the DCOPF routine using PYPOWER.

        This method invokes `self.solve(**kwargs)`, which internally utilizes
        `pypower.ppoption` and `pypower.runopf` to solve the DCOPF problem.

        Parameters
        ----------
        OUT_ALL : int, optional
            Controls the amount of output printed (default: 0; 0: none, 1: little,
            2: lots, 3: all). This is passed to `self.solve()`.
        VERBOSE : int, optional
            Controls the verbosity of the output (default: 1; 0: none, 1: little,
            2: lots, 3: all). This is passed to `self.solve()`.

        Keyword Arguments
        -------------------
        - ``OPF_ALG_DC``
            DC OPF algorithm (default: 200; 0: choose default solver based on availability,
            200: PIPS, 250: PIPS-sc, 400: IPOPT, 500: CPLEX, 600: MOSEK,
            700: GUROBI).

        Returns
        -------
        bool
            True if the optimization converged successfully, False otherwise.
        """
        return super().run(OUT_ALL=OUT_ALL, VERBOSE=VERBOSE, OPF_ALG_DC=OPF_ALG_DC, **kwargs)


class ACOPF1(DCOPF1):
    """
    AC optimal power flow using PYPOWER.

    This routine provides a wrapper for running AC optimal power flow analysis using
    the PYPOWER.
    It leverages PYPOWER's internal AC optimal power flow solver and maps results
    back to the AMS system.

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

    def solve(self, OUT_ALL=0, VERBOSE=1, **kwargs):
        ppc = system2ppc(self.system)
        ppopt = ppoption(OUT_ALL=OUT_ALL, VERBOSE=VERBOSE, **kwargs)
        res = runopf(casedata=ppc, ppopt=ppopt)
        return res

    def run(self, OUT_ALL=0, VERBOSE=1, **kwargs):
        """
        Run the ACOPF routine using PYPOWER.

        This method invokes `self.solve(**kwargs)`, which internally utilizes
        `pypower.ppoption` and `pypower.runopf` to solve the ACOPF problem.

        In PYPOWER, the ``c0`` term (the constant coefficient in the generator cost
        function) is always included in the objective, regardless of the generator's
        commitment status. See `pypower/opf_costfcn.py` for implementation details.

        Parameters
        ----------
        OUT_ALL : int, optional
            Controls the amount of output printed (default: 0; 0: none, 1: little,
            2: lots, 3: all). This is passed to `self.solve()`.
        VERBOSE : int, optional
            Controls the verbosity of the output (default: 1; 0: none, 1: little,
            2: lots, 3: all). This is passed to `self.solve()`.
        **kwargs : dict, optional
            Additional keyword arguments passed to `self.solve()`. These are
            forwarded to PYPOWER's `ppoption` and `runopf` functions to customize
            solver options, tolerances, output verbosity, and other solver-specific
            settings.

            For a full list of options, refer to the PYPOWER documentation:
            https://github.com/rwl/PYPOWER/blob/master/pypower/ppoption.py

        Keyword Arguments
        -------------------
        - ``OPF_VIOLATION`` : float
            Constraint violation tolerance (default: 5e-6).
        - ``OPF_FLOW_LIM`` : int
            Quantity to limit for branch flow constraints (default: 0; 0: MVA, 1: MW, 2: current).
        - ``OPF_IGNORE_ANG_LIM``: bool
            Ignore angle difference limits for branches even if specified (default: False).
        - ``VERBOSE`` : int
            Amount of progress info printed (default: 1; 0: none, 1: little, 2: lots, 3: all).
        - ``OUT_ALL`` : int
            Controls printing of results (default: -1; -1: individual flags control what prints,
            0: don't print anything (overrides individual flags),
            1: print everything (overrides individual flags)).
        - ``OUT_SYS_SUM`` : bool
            Print system summary (default: True).
        - ``OUT_AREA_SUM`` : bool
            Print area summaries (default: False).
        - ``OUT_BUS`` : bool
            Print bus detail (default: True).
        - ``OUT_BRANCH`` : bool
            Print branch detail (default: True).
        - ``OUT_GEN`` : bool
            Print generator detail (default: False; OUT_BUS also includes gen info).
        - ``OUT_ALL_LIM`` : int
            Control constraint info output (default: -1; -1: individual flags control what constraint info prints,
            0: no constraint info (overrides individual flags),
            1: binding constraint info (overrides individual flags),
            2: all constraint info (overrides individual flags)).
        - ``OUT_V_LIM`` : int
            Control output of voltage limit info (default: 1; 0: don't print,
            1: print binding constraints only, 2: print all constraints).
        - ``OUT_LINE_LIM`` : int
            Control output of line limit info (default: 1; 0: don't print,
            1: print binding constraints only, 2: print all constraints).
        - ``OUT_PG_LIM`` : int
            Control output of generator P limit info (default: 1; 0: don't print,
            1: print binding constraints only, 2: print all constraints).
        - ``OUT_QG_LIM`` : int
            Control output of generator Q limit info (default: 1; 0: don't print,
            1: print binding constraints only, 2: print all constraints).
        - ``OUT_RAW`` : bool
            Print raw data (default: False).
        - ``OPF_VIOLATION`` : float
            Constraint violation tolerance (default: 5e-6).
        - ``OPF_FLOW_LIM`` : int
            Quantity to limit for branch flow constraints (default: 0; 0: MVA, 1: MW, 2: current).
        - ``OPF_IGNORE_ANG_LIM``: bool
            Ignore angle difference limits for branches even if specified (default: False).
        - ``VERBOSE`` : int
            Amount of progress info printed (default: 1; 0: none, 1: little, 2: lots, 3: all).
        - ``OUT_ALL`` : int
            Controls printing of results (default: -1; -1: individual flags control what prints,
            0: don't print anything (overrides individual flags),
            1: print everything (overrides individual flags)).
        - ``OUT_SYS_SUM`` : bool
            Print system summary (default: True).
        - ``OUT_AREA_SUM`` : bool
            Print area summaries (default: False).
        - ``OUT_BUS`` : bool
            Print bus detail (default: True).
        - ``OUT_BRANCH`` : bool
            Print branch detail (default: True).
        - ``OUT_GEN`` : bool
            Print generator detail (default: False; OUT_BUS also includes gen info).
        - ``OUT_ALL_LIM`` : int
            Control constraint info output (default: -1;
            -1: individual flags control what constraint info prints,
             0: no constraint info (overrides individual flags),
             1: binding constraint info (overrides individual flags),
             2: all constraint info (overrides individual flags)).
        - ``OUT_V_LIM`` : int
            Control output of voltage limit info (default: 1; 0: don't print,
            1: print binding constraints only, 2: print all constraints).
        - ``OUT_LINE_LIM`` : int
            Control output of line limit info (default: 1; 0: don't print,
            1: print binding constraints only, 2: print all constraints).
        - ``OUT_PG_LIM`` : int
            Control output of generator P limit info (default: 1; 0: don't print,
            1: print binding constraints only, 2: print all constraints).
        - ``OUT_QG_LIM`` : int
            Control output of generator Q limit info (default: 1; 0: don't print,
            1: print binding constraints only, 2: print all constraints).
        - ``OUT_RAW`` : bool
            Print raw data (default: False).

        Returns
        -------
        bool
            True if the optimization converged successfully, False otherwise.
        """
        super().run(OUT_ALL=OUT_ALL, VERBOSE=VERBOSE, **kwargs)
