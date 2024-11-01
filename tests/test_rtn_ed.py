import unittest
import numpy as np

import ams


class TestED(unittest.TestCase):
    """
    Test routine `ED`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.xlsx"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])

    def test_trip(self):
        """
        Test generator trip.
        """
        self.ss.ED.run(solver='CLARABEL')
        obj = self.ss.ED.obj.v

        # --- trip load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        obj_pqt = self.ss.ED.obj.v

        self.assertLess(obj_pqt, obj, "Load trip does not take effect!")

        # --- trip generator ---
        self.ss.StaticGen.set(src='u', attr='v', idx='PV_1', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED.converged, "ED did not converge under generator trip!")
        obj_gt = self.ss.ED.obj.v
        self.assertGreater(obj_gt, obj, "Generator trip does not take effect!")

        pg_trip = self.ss.ED.get(src='pg', attr='v', idx='PV_1')
        np.testing.assert_almost_equal(pg_trip, 0, decimal=6)

        # --- trip line ---
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.Ed.converged, "ED did not converge under line trip!")
        obj_lt = self.ss.ED.obj.v
        self.assertGreater(obj_lt, obj_gt, "Line trip does not take effect!")

        plf_trip = self.ss.ED.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(plf_trip, 0, decimal=6)
