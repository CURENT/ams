from functools import wraps
import unittest
import numpy as np

import ams
from ams.shared import require_igraph, MIP_SOLVERS, INSTALLED_SOLVERS


def require_MIP_solver(f):
    """
    Decorator for functions that require MIP solver.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not any(s in MIP_SOLVERS for s in INSTALLED_SOLVERS):
            raise ModuleNotFoundError("No MIP solver is available.")
        return f(*args, **kwargs)

    return wrapper


class TestRoutineMethods(unittest.TestCase):
    """
    Test methods of `Routine`.
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

        self.ss.DCOPF.run(solver='OSQP')
        self.assertEqual(self.ss.DCOPF.exit_code, 0, "Exit code is not 0.")
        np.testing.assert_equal(self.ss.DCOPF.get('pg', 'PV_30', 'v'),
                                self.ss.StaticGen.get('p', 'PV_30', 'v'))


class TestRoutineSolve(unittest.TestCase):
    """
    Test solving routines.
    """
    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                           default_config=True,
                           no_output=True,
                           )

    def test_RTED(self):
        """
        Test `RTED.run()`.
        """

        self.ss.RTED.run(solver='OSQP')
        self.assertEqual(self.ss.RTED.exit_code, 0, "Exit code is not 0.")

    def test_ED(self):
        """
        Test `ED.run()`.
        """

        self.ss.ED.run(solver='OSQP')
        self.assertEqual(self.ss.ED.exit_code, 0, "Exit code is not 0.")

    @require_MIP_solver
    def test_UC(self):
        """
        Test `UC.run()`.
        """

        self.ss.UC.run()
        self.assertEqual(self.ss.UC.exit_code, 0, "Exit code is not 0.")

    @require_MIP_solver
    def test_RTED2(self):
        """
        Test `RTED2.run()`.
        """

        self.ss.RTED2.run()
        self.assertEqual(self.ss.RTED2.exit_code, 0, "Exit code is not 0.")

    @require_MIP_solver
    def test_ED2(self):
        """
        Test `ED2.run()`.
        """

        self.ss.ED2.run()
        self.assertEqual(self.ss.ED2.exit_code, 0, "Exit code is not 0.")

    @require_MIP_solver
    def test_UC2(self):
        """
        Test `UC2.run()`.
        """

        self.ss.UC2.run()
        self.assertEqual(self.ss.UC2.exit_code, 0, "Exit code is not 0.")


class TestRoutineGraph(unittest.TestCase):
    """
    Test routine graph.
    """

    @require_igraph
    def test_5bus_graph(self):
        """
        Test routine graph of PJM 5-bus system.
        """
        ss = ams.load(ams.get_case("5bus/pjm5bus_uced.xlsx"),
                      default_config=True,
                      no_output=True,
                      )
        _, g = ss.DCOPF.igraph()
        self.assertGreaterEqual(np.min(g.degree()), 1)

    @require_igraph
    def test_ieee14_graph(self):
        """
        Test routine graph of IEEE 14-bus system.
        """
        ss = ams.load(ams.get_case("ieee14/ieee14_uced.xlsx"),
                      default_config=True,
                      no_output=True,
                      )
        _, g = ss.DCOPF.igraph()
        self.assertGreaterEqual(np.min(g.degree()), 1)

    @require_igraph
    def test_ieee39_graph(self):
        """
        Test routine graph of IEEE 39-bus system.
        """
        ss = ams.load(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                      default_config=True,
                      no_output=True,
                      )
        _, g = ss.DCOPF.igraph()
        self.assertGreaterEqual(np.min(g.degree()), 1)

    @require_igraph
    def test_npcc_graph(self):
        """
        Test routine graph of NPCC 140-bus system.
        """
        ss = ams.load(ams.get_case("npcc/npcc_uced.xlsx"),
                      default_config=True,
                      no_output=True,
                      )
        _, g = ss.DCOPF.igraph()
        self.assertGreaterEqual(np.min(g.degree()), 1)

    @require_igraph
    def test_wecc_graph(self):
        """
        Test routine graph of WECC 179-bus system.
        """
        ss = ams.load(ams.get_case("wecc/wecc_uced.xlsx"),
                      default_config=True,
                      no_output=True,
                      )
        _, g = ss.DCOPF.igraph()
        self.assertGreaterEqual(np.min(g.degree()), 1)
