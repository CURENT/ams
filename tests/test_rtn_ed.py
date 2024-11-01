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

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.ED.init()
        self.assertTrue(self.ss.ED.initialized, "ED initialization failed!")

    def test_trip(self):
        """
        Test generator trip.
        """
        self.ss.ED.run(solver='CLARABEL')
        obj = self.ss.ED.obj.v

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        obj_pqt = self.ss.ED.obj.v
        self.assertLess(obj_pqt, obj, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        obj_pqt2 = self.ss.ED.obj.v
        self.assertLess(obj_pqt2, obj_pqt, "Load trip does not take effect!")

        # --- trip line ---
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED.converged, "ED did not converge under line trip!")
        plf_l3 = self.ss.ED.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        # --- trip generator ---
        # a) check StaticGen.u does not take effect
        # NOTE: in ED, `EDTSlot.ug` is used instead of `StaticGen.u`
        self.ss.StaticGen.set(src='u', attr='v', idx='PV_1', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED.get(src='pg', attr='v', idx='PV_1')
        np.testing.assert_array_less(np.zeros_like(pg_pv1), pg_pv1,
                                     err_msg="Generator trip take effect, which is unexpected!")

        # b) check EDTSlot.ug takes effect
