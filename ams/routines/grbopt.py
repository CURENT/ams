"""
Routines using gurobi-optimods.
"""
import logging

import io

import scipy.io

from andes.shared import np, pd
from andes.utils.misc import elapsed

from ams.core import Config

from ams.io.pypower import system2ppc

from ams.opt import Var, Objective

from ams.routines.pypower import DCPF1
from ams.shared import opf

logger = logging.getLogger(__name__)


class OPF(DCPF1):
    """
    Optimal Power Flow (OPF) routine using gurobi-optimods.

    This class provides an interface for performing optimal power flow analysis
    with gurobi-optimods, supporting both AC and DC OPF formulations.

    In addition to optimizing generator dispatch, this routine can also optimize
    transmission line statuses (branch switching), enabling topology optimization.
    Refer to the gurobi-optimods documentation for further details:

    https://gurobi-optimods.readthedocs.io/en/stable/mods/opf/opf.html
    """

    def __init__(self, system, config):
        DCPF1.__init__(self, system, config)
        self.info = 'Optimal Power Flow'
        self.type = 'ACED'

        # Overwrite the config to be empty, as it is not used in this routine
        self.config = Config(self.class_name)

        self.obj = Objective(name='obj',
                             info='total cost, placeholder',
                             e_str='sum(c2 * pg**2 + c1 * pg + c0)',
                             unit='$',
                             sense='min',)

        self.pi = Var(info='Lagrange multiplier on real power mismatch',
                      name='pi', unit='$/p.u.',
                      model='Bus', src=None,)

        self.uld = Var(info='Line commitment decision',
                       name='uld', tex_name=r'u_{l,d}',
                       model='Line', src='u',)

    def solve(self, **kwargs):
        ppc = system2ppc(self.system)
        mat = io.BytesIO()
        scipy.io.savemat(mat, {'mpc': ppc})
        mat.seek(0)
        res = opf.solve_opf(opf.read_case_matpower(mat), **kwargs)
        return res

    def unpack(self, res, **kwargs):
        """
        Unpack the results from the gurobi-optimods.
        """
        # NOTE: Map gurobi-optimods results to PPC-compatible format.
        # Only relevant columns are populated, as required by `DCOPF.unpack()`.
        # If future versions of gurobi-optimods provide additional outputs,
        # this mapping may need to be updated to extract and assign new fields.

        res_new = dict()
        res_new['success'] = res['success']
        res_new['et'] = res['et']
        res_new['f'] = res['f']
        res_new['baseMVA'] = res['baseMVA']

        bus = pd.DataFrame(res['bus'])
        res_new['bus'] = np.zeros((self.system.Bus.n, 17))
        res_new['bus'][:, 7] = bus['Vm'].values
        res_new['bus'][:, 8] = bus['Va'].values
        # NOTE: As of v2.3.2, gurobi-optimods does not return LMP

        gen = pd.DataFrame(res['gen'])
        res_new['gen'] = np.zeros((self.system.StaticGen.n, 14))
        res_new['gen'][:, 1] = gen['Pg'].values
        res_new['gen'][:, 2] = gen['Qg'].values

        branch = pd.DataFrame(res['branch'])
        res_new['branch'] = np.zeros((self.system.Line.n, 14))
        res_new['branch'][:, 13] = branch['Pf'].values
        # NOTE: unpack branch_switching decision
        res_new['branch'][:, 10] = branch['switching'].values
        return super().unpack(res_new)

    def run(self, **kwargs):
        """
        Run the OPF routine using gurobi-optimods.

        This method invokes `gurobi-optimods.opf.solve_opf` to solve the OPF problem.

        Parameters
        ----------
        - opftype : str
            Type of OPF to solve (default: 'AC').
        - branch_switching : bool
            Enable branch switching (default: False).
        - min_active_branches : float
            Defines the minimum number of branches that must be turned on when
            branch switching is active, i.e. the minimum number of turned on
            branches is equal to ``numbranches * min_active_branches``. Has no
            effect if ``branch_switching`` is set to False.
        - use_mip_start : bool
            Use MIP start (default: False).
        - time_limit : float
            Time limit for the solver (default: 0.0, no limit).
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
            msg += n_iter_str + "with gurobi-optimods."
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
            msg += n_iter_str + "with gurobi-optimods."
            logger.warning(msg)
            return False
