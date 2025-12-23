import unittest
import numpy as np

import ams
from ams.shared import skip_unittest_without_MISOCP, skip_unittest_without_PYPOWER


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

    def test_vBus(self):
        """
        Test vBus is not all zero.
        """
        self.ss.RTED.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTED.converged, "RTED did not converge!")
        self.assertTrue(np.any(self.ss.RTED.vBus.v), "vBus is all zero!")

    @skip_unittest_without_PYPOWER
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

    def test_vBus(self):
        """
        Test vBus is not all zero.
        """
        self.ss.RTEDDG.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTEDDG.converged, "RTEDDG did not converge!")
        self.assertTrue(np.any(self.ss.RTEDDG.vBus.v), "vBus is all zero!")

    @skip_unittest_without_PYPOWER
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

    @skip_unittest_without_MISOCP
    def test_vBus(self):
        """
        Test vBus is not all zero.
        """
        self.ss.RTEDES.run(solver='SCIP')
        self.assertTrue(self.ss.RTEDES.converged, "RTEDES did not converge!")
        self.assertTrue(np.any(self.ss.RTEDES.vBus.v), "vBus is all zero!")

    @skip_unittest_without_MISOCP
    def test_ch_decision(self):
        """
        Test charging/discharging decision for charging/discharging duration time.

        Scenarios to validate:
                    res     ucd     tdc     tdc0
        0           True    1       1.0     0.5
        1           False   0       1.0     0.5
        2           True    1       1.0     0.0
        3           True    0       1.0     0.0
        4           True    1       0.5     1.0
        5           True    0       0.5     1.0
        """
        self.ss.RTEDES.init()

        self.ss.ESD1.set(src='tdc0', attr='v', idx='ESD1_1', value=0.5)
        self.ss.ESD1.set(src='tdc', attr='v', idx='ESD1_1', value=1.0)
        # scenario 1: initially charging, charging duration time not met, decision to charge
        self.ss.RTEDES.ucd.optz.value = np.array([1])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 1!")
        # scenario 2: initially charging, with charging duration time not met, decision not to charge
        self.ss.RTEDES.ucd.optz.value = np.array([0])
        self.assertFalse(self.ss.RTEDES.tcdr.e[0] <= 0,
                         "RTEDES.tcdr should be broken in scenario 2!")

        self.ss.ESD1.set(src='tdc0', attr='v', idx='ESD1_1', value=0.0)
        self.ss.ESD1.set(src='tdc', attr='v', idx='ESD1_1', value=1.0)
        # scenario 3: initially not charging, decision to charge
        self.ss.RTEDES.ucd.optz.value = np.array([1])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 3!")
        # scenario 4: initially not charging, decision not to charge
        self.ss.RTEDES.ucd.optz.value = np.array([0])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 4!")

        self.ss.ESD1.set(src='tdc0', attr='v', idx='ESD1_1', value=1.0)
        self.ss.ESD1.set(src='tdc', attr='v', idx='ESD1_1', value=0.5)
        # scenario 5: initially charging, charging duration time met, decision to charge
        self.ss.RTEDES.ucd.optz.value = np.array([1])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 5!")
        # scenario 6: initially charging, charging duration time met, decision not to charge
        self.ss.RTEDES.ucd.optz.value = np.array([0])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 6!")

    @skip_unittest_without_PYPOWER
    @skip_unittest_without_MISOCP
    def test_dc2ac(self):
        """
        Test `RTEDES.dc2ac()` method.
        """
        self.ss.RTEDES.run(solver='SCIP')
        self.ss.RTEDES.dc2ac()
        self.assertTrue(self.ss.RTEDES.converted, "AC conversion failed!")
        self.assertTrue(self.ss.RTEDES.exec_time > 0, "Execution time is not greater than 0.")

        stg_idx = self.ss.StaticGen.get_all_idxes()
        pg_rtedes = self.ss.RTEDES.get(src='pg', attr='v', idx=stg_idx)
        pg_acopf = self.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx)
        np.testing.assert_almost_equal(pg_rtedes, pg_acopf, decimal=3)

        bus_idx = self.ss.Bus.get_all_idxes()
        v_rtedes = self.ss.RTEDES.get(src='vBus', attr='v', idx=bus_idx)
        v_acopf = self.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(v_rtedes, v_acopf, decimal=3)

        a_rtedes = self.ss.RTEDES.get(src='aBus', attr='v', idx=bus_idx)
        a_acopf = self.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(a_rtedes, a_acopf, decimal=3)
