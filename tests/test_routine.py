import unittest
from functools import wraps
import numpy as np

import ams

try:
    from ams.shared import igraph
    getattr(igraph, '__version__')
    HAVE_IGRAPH = True
except (ImportError, AttributeError):
    HAVE_IGRAPH = False


def require_igraph(f):
    """
    Decorator for functions that require igraph.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            getattr(igraph, '__version__')
        except AttributeError:
            raise ModuleNotFoundError("igraph needs to be manually installed.")

        return f(*args, **kwds)

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

        # get an rparam value
        np.testing.assert_equal(self.ss.DCOPF.get('ug', 'PV_30'), 1)

        # get an unpacked var value
        self.ss.DCOPF.run()
        self.assertEqual(self.ss.DCOPF.exit_code, 0, "Exit code is not 0.")
        np.testing.assert_equal(self.ss.DCOPF.get('pg', 'PV_30', 'v'),
                                self.ss.StaticGen.get('p', 'PV_30', 'v'))


class TestRoutineInit(unittest.TestCase):
    """
    Test solving routines.
    """
    def setUp(self) -> None:
        self.s1 = ams.load(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                           default_config=True,
                           no_output=True,
                           )
        self.s2 = ams.load(ams.get_case("ieee123/ieee123_regcv1.xlsx"),
                           default_config=True,
                           no_output=True,
                           )

    def test_Init(self):
        """
        Test `routine.init()`.
        """
        # NOTE: for DED, using ieee123 as a test case
        for rtn in self.s1.routines.values():
            if not rtn._data_check():
                rtn = getattr(self.s2, rtn.class_name)
                if not rtn._data_check():
                    continue  # TODO: here should be a warning?
            self.assertTrue(rtn.init(force=True),
                            f"{rtn.class_name} initialization failed!")


@unittest.skipUnless(HAVE_IGRAPH, "igraph not available")
class TestRoutineGraph(unittest.TestCase):
    """
    Test routine graph.
    """

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
