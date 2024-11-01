import unittest

import ams


class TestDCOPF(unittest.TestCase):
    """
    Test routine `DCOPF`.
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
        # --- run DCOPF ---
        self.ss.DCOPF.run(solver='CLARABEL')
        obj = self.ss.DCOPF.obj.v

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='CLARABEL')
        obj_pqt = self.ss.DCOPF.obj.v
        self.assertLess(obj_pqt, obj, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='CLARABEL')
        obj_pqt2 = self.ss.DCOPF.obj.v
        self.assertLess(obj_pqt2, obj_pqt, "Load trip does not take effect!")

        # --- trip generator ---
        self.ss.StaticGen.set(src='u', attr='v', idx='PV_1', value=0)
        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCOPF.converged, "DCOPF did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.DCOPF.get(src='pg', attr='v', idx='PV_1'),
                               0, places=6,
                               msg="Generator trip does not take effect!")
        self.assertAlmostEqual(self.ss.DCOPF.get(src='pg', attr='v', idx='PV_1'),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        # --- trip line ---
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCOPF.converged, "DCOPF did not converge under line trip!")

        self.assertAlmostEqual(self.ss.DCOPF.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")
