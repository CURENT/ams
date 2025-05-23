"""
Test wrapper routines of PYPOWER.
"""
import unittest

import ams
from ams.shared import skip_unittest_without_PYPOWER


class TestDCPF1(unittest.TestCase):
    """
    Test routine `DCPF1`.
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
        self.ss.DCPF1.init()
        self.assertTrue(self.ss.DCPF1.initialized, "DCPF1 initialization failed!")

    @skip_unittest_without_PYPOWER
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.DCPF1.update()
        self.ss.DCPF1.run()
        self.assertTrue(self.ss.DCPF1.converged, "DCPF1 did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.DCPF1.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    @skip_unittest_without_PYPOWER
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.DCPF1.update()
        self.ss.DCPF1.run()
        self.assertTrue(self.ss.DCPF1.converged, "DCPF1 did not converge under line trip!")
        self.assertAlmostEqual(self.ss.DCPF1.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    @skip_unittest_without_PYPOWER
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run DCPF1 ---
        self.ss.DCPF1.run()
        pgs = self.ss.DCPF1.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.DCPF1.update()

        self.ss.DCPF1.run()
        pgs_pqt = self.ss.DCPF1.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.DCPF1.update()

        self.ss.DCPF1.run()
        pgs_pqt2 = self.ss.DCPF1.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")


class TestPFlow1(unittest.TestCase):
    """
    Test routine `PFlow1`.
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
        self.ss.PFlow1.init()
        self.assertTrue(self.ss.PFlow1.initialized, "PFlow1 initialization failed!")

    @skip_unittest_without_PYPOWER
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.PFlow1.update()
        self.ss.PFlow1.run()
        self.assertTrue(self.ss.PFlow1.converged, "PFlow1 did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.PFlow1.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    @skip_unittest_without_PYPOWER
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.PFlow1.update()
        self.ss.PFlow1.run()
        self.assertTrue(self.ss.PFlow1.converged, "PFlow1 did not converge under line trip!")
        self.assertAlmostEqual(self.ss.PFlow1.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    @skip_unittest_without_PYPOWER
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run PFlow1 ---
        self.ss.PFlow1.run()
        pgs = self.ss.PFlow1.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.PFlow1.update()

        self.ss.PFlow1.run()
        pgs_pqt = self.ss.PFlow1.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.PFlow1.update()

        self.ss.PFlow1.run()
        pgs_pqt2 = self.ss.PFlow1.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")


class TestDCOPF1(unittest.TestCase):
    """
    Test routine `DCOPF1`.
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
        self.ss.DCOPF1.init()
        self.assertTrue(self.ss.DCOPF1.initialized, "DCOPF initialization failed!")

    @skip_unittest_without_PYPOWER
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.DCOPF1.update()
        self.ss.DCOPF1.run()
        self.assertTrue(self.ss.DCOPF1.converged, "DCOPF did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.DCOPF1.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    @skip_unittest_without_PYPOWER
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.DCOPF1.update()
        self.ss.DCOPF1.run()
        self.assertTrue(self.ss.DCOPF1.converged, "DCOPF1 did not converge under line trip!")
        self.assertAlmostEqual(self.ss.DCOPF1.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    @skip_unittest_without_PYPOWER
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run DCOPF1 ---
        self.ss.DCOPF1.run()
        pgs = self.ss.DCOPF1.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.DCOPF1.update()

        self.ss.DCOPF1.run()
        pgs_pqt = self.ss.DCOPF1.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.DCOPF1.update()

        self.ss.DCOPF1.run()
        pgs_pqt2 = self.ss.DCOPF1.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")


class TestACOPF1(unittest.TestCase):
    """
    Test routine `ACOPF1`.
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
        self.ss.ACOPF1.init()
        self.assertTrue(self.ss.ACOPF1.initialized, "ACOPF1 initialization failed!")

    @skip_unittest_without_PYPOWER
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.ACOPF1.update()
        self.ss.ACOPF1.run()
        self.assertTrue(self.ss.ACOPF1.converged, "ACOPF1 did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.ACOPF1.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    @skip_unittest_without_PYPOWER
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.ACOPF1.update()
        self.ss.ACOPF1.run()
        self.assertTrue(self.ss.ACOPF1.converged, "ACOPF1 did not converge under line trip!")
        self.assertAlmostEqual(self.ss.ACOPF1.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    @skip_unittest_without_PYPOWER
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run ACOPF1 ---
        self.ss.ACOPF1.run()
        pgs = self.ss.ACOPF1.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.ACOPF1.update()

        self.ss.ACOPF1.run()
        pgs_pqt = self.ss.ACOPF1.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.ACOPF1.update()

        self.ss.ACOPF1.run()
        pgs_pqt2 = self.ss.ACOPF1.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")
