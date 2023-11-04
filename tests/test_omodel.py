import unittest
import numpy as np

import ams
import cvxpy as cp


def require_MIP_solver(f):
    """
    Decorator for skipping tests that require MIP solver.
    """
    def wrapper(*args, **kwargs):
        all_solvers = cp.installed_solvers()
        mip_solvers = ['CBC', 'COPT', 'GLPK_MI', 'CPLEX', 'GUROBI',
                       'MOSEK', 'SCIP', 'XPRESS', 'SCIPY']
        if any(s in mip_solvers for s in all_solvers):
            pass
        else:
            raise unittest.SkipTest("MIP solver is not available.")
        return f(*args, **kwargs)
    return wrapper


class TestOMdel(unittest.TestCase):
    """
    Test methods of `OModel`.
    """
    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                           default_config=True,
                           no_output=True,
                           )

    def test_var_access_brfore_solve(self):
        """
        Test `Var` access before solve.
        """
        self.ss.DCOPF.init()
        self.assertIsNone(self.ss.DCOPF.pg.v)

    def test_var_access_after_solve(self):
        """
        Test `Var` access after solve.
        """
        self.ss.DCOPF.run()
        np.testing.assert_equal(self.ss.DCOPF.pg.v,
                                self.ss.StaticGen.get(src='p', attr='v',
                                                      idx=self.ss.DCOPF.pg.get_idx()))

    def test_constr_access_brfore_solve(self):
        """
        Test `Constr` access before solve.
        """
        self.ss.DCOPF.init(force=True)
        np.testing.assert_equal(self.ss.DCOPF.lub.v, None)

    def test_constr_access_after_solve(self):
        """
        Test `Constr` access after solve.
        """
        self.ss.DCOPF.run()
        self.assertIsInstance(self.ss.DCOPF.lub.v, np.ndarray)

    # NOTE: add Var, Constr add functions
