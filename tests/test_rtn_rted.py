import unittest
import numpy as np

from andes.shared import pd

import ams


class TestRTED(unittest.TestCase):
    """
    Test methods of `RTED`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.xlsx"),
                           setup=True,
                           default_config=True,
                           no_output=True,
                           )
        self.ss.RTED.run(solver='CLARABEL')

    def test_dc2ac(self):
        """
        Test `RTED.init()` method.
        """
        self.ss.RTED.dc2ac()
        self.assertTrue(self.ss.RTED.converted, "AC conversion failed!")
        self.assertTrue(self.ss.RTED.exec_time > 0, "Execution time is not greater than 0.")

        stg_idx = self.ss.StaticGen.get_idx()
        pg_rted = self.ss.RTED.get(src='pg', attr='v', idx=stg_idx)
        pg_acopf = self.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx)
        np.testing.assert_almost_equal(pg_rted, pg_acopf, decimal=3)

        bus_idx = self.ss.Bus.get_idx()
        v_rted = self.ss.RTED.get(src='vBus', attr='v', idx=bus_idx)
        v_acopf = self.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(v_rted, v_acopf, decimal=3)

        a_rted = self.ss.RTED.get(src='aBus', attr='v', idx=bus_idx)
        a_acopf = self.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(a_rted, a_acopf, decimal=3)

    def test_trip(self):
        """
        Test generator trip.
        """
        self.ss.RTED.run(solver='CLARABEL')
        obj = self.ss.RTED.obj.v

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0)
        self.ss.RTED.update()

        self.ss.RTED.run(solver='CLARABEL')
        obj_pqt = self.ss.RTED.obj.v
        self.assertLess(obj_pqt, obj, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.RTED.update()

        self.ss.RTED.run(solver='CLARABEL')
        obj_pqt2 = self.ss.RTED.obj.v
        self.assertLess(obj_pqt2, obj_pqt, "Load trip does not take effect!")

        # --- trip generator ---
        self.ss.StaticGen.set(src='u', attr='v', idx='PV_1', value=0)
        self.ss.RTED.update()

        self.ss.RTED.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTED.converged, "RTED did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.RTED.get(src='pg', attr='v', idx='PV_1'),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        pg_trip = self.ss.RTED.get(src='pg', attr='v', idx='PV_1')
        np.testing.assert_almost_equal(pg_trip, 0, decimal=6)

        # --- trip line ---
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.RTED.update()

        self.ss.RTED.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTED.converged, "RTED did not converge under line trip!")
        self.assertAlmostEqual(self.ss.RTED.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")
