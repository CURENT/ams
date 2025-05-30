import unittest
import numpy as np

import ams


class TestDCOPF2(unittest.TestCase):
    """
    Test routine `DCOPF2`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        # build PTDF
        self.ss.mats.build_ptdf()

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.DCOPF2.init()
        self.assertTrue(self.ss.DCOPF2.initialized, "DCOPF2 initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.DCOPF2.update()
        self.ss.DCOPF2.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCOPF2.converged, "DCOPF2 did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.DCOPF2.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.DCOPF2.update()
        self.ss.DCOPF2.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCOPF2.converged, "DCOPF2 did not converge under line trip!")
        self.assertAlmostEqual(self.ss.DCOPF2.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run DCOPF2 ---
        self.ss.DCOPF2.run(solver='CLARABEL')
        pgs = self.ss.DCOPF2.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.DCOPF2.update()

        self.ss.DCOPF2.run(solver='CLARABEL')
        pgs_pqt = self.ss.DCOPF2.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.DCOPF2.update()

        self.ss.DCOPF2.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.DCOPF2.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    def test_dc2ac(self):
        """
        Test `DCOPF2.dc2ac()` method.
        """
        self.ss.DCOPF2.run(solver='CLARABEL')
        self.ss.DCOPF2.dc2ac()
        self.assertTrue(self.ss.DCOPF2.converted, "AC conversion failed!")
        self.assertTrue(self.ss.DCOPF2.exec_time > 0, "Execution time is not greater than 0.")

        stg_idx = self.ss.StaticGen.get_all_idxes()
        pg_dcopf = self.ss.DCOPF2.get(src='pg', attr='v', idx=stg_idx)
        pg_acopf = self.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx)
        np.testing.assert_almost_equal(pg_dcopf, pg_acopf, decimal=3)

        bus_idx = self.ss.Bus.get_all_idxes()
        v_dcopf = self.ss.DCOPF2.get(src='vBus', attr='v', idx=bus_idx)
        v_acopf = self.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(v_dcopf, v_acopf, decimal=3)

        a_dcopf = self.ss.DCOPF2.get(src='aBus', attr='v', idx=bus_idx)
        a_acopf = self.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(a_dcopf, a_acopf, decimal=3)
