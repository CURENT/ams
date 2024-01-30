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
        mip_solvers = ['CPLEX', 'GUROBI', 'MOSEK']
        if any(s in mip_solvers for s in all_solvers):
            pass
        else:
            raise unittest.SkipTest("MIP solver is not available.")
        return f(*args, **kwargs)
    return wrapper


class TestDCED(unittest.TestCase):
    """
    Test DCED types routine.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                           default_config=True,
                           no_output=True,
                           )

    def test_dcopf(self):
        """
        Test DCOPF.
        """
        init = self.ss.DCOPF.init()
        self.assertTrue(init, "DCOPF initialization failed!")
        self.ss.DCOPF.run(solver='ECOS')
        np.testing.assert_equal(self.ss.DCOPF.exit_code, 0)

    def test_rted(self) -> None:
        """
        Test RTED.
        """
        init = self.ss.RTED.init()
        self.assertTrue(init, "RTED initialization failed!")
        self.ss.RTED.run(solver='ECOS')
        np.testing.assert_equal(self.ss.RTED.exit_code, 0)

    def test_ed(self) -> None:
        """
        Test ED.
        """
        init = self.ss.ED.init()
        self.assertTrue(init, "ED initialization failed!")
        self.ss.ED.run(solver='ECOS')
        np.testing.assert_equal(self.ss.ED.exit_code, 0)

    @require_MIP_solver
    def test_rtedes(self) -> None:
        """
        Test RTEDES.
        """
        init = self.ss.RTEDES.init()
        self.assertTrue(init, "RTEDES initialization failed!")
        self.ss.RTEDES.run()
        np.testing.assert_equal(self.ss.RTEDES.exit_code, 0)

    @require_MIP_solver
    def test_edes(self) -> None:
        """
        Test EDES.
        """
        init = self.ss.EDES.init()
        self.assertTrue(init, "EDES initialization failed!")
        self.ss.EDES.run()
        np.testing.assert_equal(self.ss.EDES.exit_code, 0)


class test_DCUC(unittest.TestCase):
    """
    Test DCUC types routine.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                           default_config=True,
                           no_output=True,
                           )

    @require_MIP_solver
    def test_uc(self) -> None:
        """
        Test UC.
        """
        init = self.ss.UC.init()
        self.assertTrue(init, "UC initialization failed!")
        self.ss.UC.run()
        np.testing.assert_equal(self.ss.UC.exit_code, 0)

    @require_MIP_solver
    def test_uces(self) -> None:
        """
        Test UCES.
        """
        init = self.ss.UCES.init()
        self.assertTrue(init, "UCES initialization failed!")
        self.ss.UCES.run()
        np.testing.assert_equal(self.ss.UCES.exit_code, 0)
