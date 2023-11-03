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


class TestRoutineMethods(unittest.TestCase):
    """
    Test methods of Routine.
    """
    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                           default_config=True,
                           no_output=True,
                           )

    def test_routine_set(self):
        """
        Test `Routine.set()` method.
        """

        self.ss.DCOPF.set('c2', 'GCost_1', 'v', 10)
        np.testing.assert_equal(self.ss.GCost.get('c2', 'GCost_1', 'v'), 10)

    def test_routine_get(self):
        """
        Test `Routine.get()` method.
        """

        # get a rparam value
        np.testing.assert_equal(self.ss.DCOPF.get('ug', 'PV_30'), 1)

        # before solving, vars values are not available
        self.assertRaises(KeyError, self.ss.DCOPF.get, 'pg', 'PV_30')

        self.ss.DCOPF.run(solver='OSQP')
        self.assertEqual(self.ss.DCOPF.exit_code, 0, "Exit code is not 0.")
        np.testing.assert_equal(self.ss.DCOPF.get('pg', 'PV_30', 'v'),
                                self.ss.StaticGen.get('p', 'PV_30', 'v'))

    def test_RTED(self):
        """
        Test `RTED.run()`.
        """

        self.ss.RTED.run(solver='OSQP')
        self.assertEqual(self.ss.RTED.exit_code, 0, "Exit code is not 0.")

    @require_MIP_solver
    def test_RTED2(self):
        """
        Test `RTED2.run()`.
        """

        self.ss.RTED2.run()
        self.assertEqual(self.ss.RTED2.exit_code, 0, "Exit code is not 0.")

    def test_ED(self):
        """
        Test `ED.run()`.
        """

        self.ss.ED.run(solver='OSQP')
        self.assertEqual(self.ss.ED.exit_code, 0, "Exit code is not 0.")

    @require_MIP_solver
    def test_ED2(self):
        """
        Test `ED2.run()`.
        """

        self.ss.ED2.run()
        self.assertEqual(self.ss.ED2.exit_code, 0, "Exit code is not 0.")
