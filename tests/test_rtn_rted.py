import unittest
import numpy as np

import ams
from ams.shared import skip_unittest_without_MISOCP


class TestRTED(unittest.TestCase):
    """
    Test methods of `RTED`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])

    def test_init(self):
        """
        Test `RTED.init()` method.
        """
        self.ss.RTED.init()
        self.assertTrue(self.ss.RTED.initialized, "RTED initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.RTED.update()
        self.ss.RTED.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTED.converged, "RTED did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.RTED.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.RTED.update()
        self.ss.RTED.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTED.converged, "RTED did not converge under line trip!")
        self.assertAlmostEqual(self.ss.RTED.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.RTED.run(solver='CLARABEL')
        pgs = self.ss.RTED.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.RTED.update()

        self.ss.RTED.run(solver='CLARABEL')
        pgs_pqt = self.ss.RTED.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.RTED.update()

        self.ss.RTED.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.RTED.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    def test_dc2ac(self):
        """
        Test `RTED.dc2ac()` method.
        """
        self.ss.RTED.run(solver='CLARABEL')
        self.ss.RTED.dc2ac()
        self.assertTrue(self.ss.RTED.converted, "AC conversion failed!")
        self.assertTrue(self.ss.RTED.exec_time > 0, "Execution time is not greater than 0.")

        stg_idx = self.ss.StaticGen.get_all_idxes()
        pg_rted = self.ss.RTED.get(src='pg', attr='v', idx=stg_idx)
        pg_acopf = self.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx)
        np.testing.assert_almost_equal(pg_rted, pg_acopf, decimal=3)

        bus_idx = self.ss.Bus.get_all_idxes()
        v_rted = self.ss.RTED.get(src='vBus', attr='v', idx=bus_idx)
        v_acopf = self.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(v_rted, v_acopf, decimal=3)

        a_rted = self.ss.RTED.get(src='aBus', attr='v', idx=bus_idx)
        a_acopf = self.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(a_rted, a_acopf, decimal=3)


class TestRTEDDG(unittest.TestCase):
    """
    Test routine `RTEDDG`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.RTEDDG.init()
        self.assertTrue(self.ss.RTEDDG.initialized, "RTEDDG initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.RTEDDG.update()
        self.ss.RTEDDG.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTEDDG.converged, "RTEDDG did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.RTEDDG.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.RTEDDG.update()
        self.ss.RTEDDG.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTEDDG.converged, "RTEDDG did not converge under line trip!")
        self.assertAlmostEqual(self.ss.RTEDDG.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.RTEDDG.run(solver='CLARABEL')
        pgs = self.ss.RTEDDG.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.RTEDDG.update()

        self.ss.RTEDDG.run(solver='CLARABEL')
        pgs_pqt = self.ss.RTEDDG.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.RTEDDG.update()

        self.ss.RTEDDG.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.RTEDDG.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    def test_dc2ac(self):
        """
        Test `RTEDDG.dc2ac()` method.
        """
        self.ss.RTEDDG.run(solver='CLARABEL')
        self.ss.RTEDDG.dc2ac()
        self.assertTrue(self.ss.RTEDDG.converted, "AC conversion failed!")
        self.assertTrue(self.ss.RTEDDG.exec_time > 0, "Execution time is not greater than 0.")

        stg_idx = self.ss.StaticGen.get_all_idxes()
        pg_rted = self.ss.RTEDDG.get(src='pg', attr='v', idx=stg_idx)
        pg_acopf = self.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx)
        np.testing.assert_almost_equal(pg_rted, pg_acopf, decimal=3)

        bus_idx = self.ss.Bus.get_all_idxes()
        v_rted = self.ss.RTEDDG.get(src='vBus', attr='v', idx=bus_idx)
        v_acopf = self.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(v_rted, v_acopf, decimal=3)

        a_rted = self.ss.RTEDDG.get(src='aBus', attr='v', idx=bus_idx)
        a_acopf = self.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(a_rted, a_acopf, decimal=3)


class TestRTEDES(unittest.TestCase):
    """
    Test routine `RTEDES`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.RTEDES.init()
        self.assertTrue(self.ss.RTEDES.initialized, "RTEDES initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.RTEDES.update()
        self.ss.RTEDES.run(solver='SCIP')
        self.assertTrue(self.ss.RTEDES.converged, "RTEDES did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.RTEDES.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)

    @skip_unittest_without_MISOCP
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.RTEDES.update()
        self.ss.RTEDES.run(solver='SCIP')
        self.assertTrue(self.ss.RTEDES.converged, "RTEDES did not converge under line trip!")
        self.assertAlmostEqual(self.ss.RTEDES.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.RTEDES.run(solver='SCIP')
        pgs = self.ss.RTEDES.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.05)
        self.ss.RTEDES.update()

        self.ss.RTEDES.run(solver='SCIP')
        pgs_pqt = self.ss.RTEDES.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.RTEDES.update()

        self.ss.RTEDES.run(solver='SCIP')
        pgs_pqt2 = self.ss.RTEDES.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")
