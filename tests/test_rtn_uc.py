import unittest
import numpy as np

import ams
from ams.shared import skip_unittest_without_MISOCP


class TestUC(unittest.TestCase):
    """
    Test routine `UC`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        # run UC._initial_guess()
        self.off_gen = self.ss.UC._initial_guess()

    def test_initial_guess(self):
        """
        Test initial guess.
        """
        u_off_gen = self.ss.StaticGen.get(src='u', idx=self.off_gen)
        np.testing.assert_equal(u_off_gen, np.zeros_like(u_off_gen),
                                err_msg="UC._initial_guess() failed!")

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.UC.init()
        self.assertTrue(self.ss.UC.initialized, "UC initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        self.ss.UC.run(solver='SCIP')
        self.assertTrue(self.ss.UC.converged, "UC did not converge!")
        pg_off_gen = self.ss.UC.get(src='pg', attr='v', idx=self.off_gen)
        np.testing.assert_almost_equal(np.zeros_like(pg_off_gen),
                                       pg_off_gen, decimal=6,
                                       err_msg="Off generators are not turned off!")

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.UC.run(solver='SCIP')
        pgs = self.ss.UC.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.UC.update()

        self.ss.UC.run(solver='SCIP')
        pgs_pqt = self.ss.UC.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.UC.update()

        self.ss.UC.run(solver='SCIP')
        pgs_pqt2 = self.ss.UC.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    @skip_unittest_without_MISOCP
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.UC.update()

        self.ss.UC.run(solver='SCIP')
        self.assertTrue(self.ss.UC.converged, "UC did not converge under line trip!")
        plf_l3 = self.ss.UC.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)


class TestUCDG(unittest.TestCase):
    """
    Test routine `UCDG`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        # run `_initial_guess()`
        self.off_gen = self.ss.UCDG._initial_guess()

    def test_initial_guess(self):
        """
        Test initial guess.
        """
        u_off_gen = self.ss.StaticGen.get(src='u', idx=self.off_gen)
        np.testing.assert_equal(u_off_gen, np.zeros_like(u_off_gen),
                                err_msg="UCDG._initial_guess() failed!")

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.UCDG.init()
        self.assertTrue(self.ss.UCDG.initialized, "UCDG initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        self.ss.UCDG.run(solver='SCIP')
        self.assertTrue(self.ss.UCDG.converged, "UCDG did not converge!")
        pg_off_gen = self.ss.UCDG.get(src='pg', attr='v', idx=self.off_gen)
        np.testing.assert_almost_equal(np.zeros_like(pg_off_gen),
                                       pg_off_gen, decimal=6,
                                       err_msg="Off generators are not turned off!")

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.UCDG.run(solver='SCIP')
        pgs = self.ss.UCDG.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.UCDG.update()

        self.ss.UCDG.run(solver='SCIP')
        pgs_pqt = self.ss.UCDG.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.UCDG.update()

        self.ss.UCDG.run(solver='SCIP')
        pgs_pqt2 = self.ss.UCDG.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    @skip_unittest_without_MISOCP
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.UCDG.update()

        self.ss.UCDG.run(solver='SCIP')
        self.assertTrue(self.ss.UCDG.converged, "UCDG did not converge under line trip!")
        plf_l3 = self.ss.UCDG.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)


class TestUCES(unittest.TestCase):
    """
    Test routine `UCES`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        # run `_initial_guess()`
        self.off_gen = self.ss.UCES._initial_guess()

    def test_initial_guess(self):
        """
        Test initial guess.
        """
        u_off_gen = self.ss.StaticGen.get(src='u', idx=self.off_gen)
        np.testing.assert_equal(u_off_gen, np.zeros_like(u_off_gen),
                                err_msg="UCES._initial_guess() failed!")

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.UCES.init()
        self.assertTrue(self.ss.UCES.initialized, "UCES initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        self.ss.UCES.run(solver='SCIP')
        self.assertTrue(self.ss.UCES.converged, "UCES did not converge!")
        pg_off_gen = self.ss.UCES.get(src='pg', attr='v', idx=self.off_gen)
        np.testing.assert_almost_equal(np.zeros_like(pg_off_gen),
                                       pg_off_gen, decimal=6,
                                       err_msg="Off generators are not turned off!")

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.UCES.run(solver='SCIP')
        pgs = self.ss.UCES.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.UCES.update()

        self.ss.UCES.run(solver='SCIP')
        pgs_pqt = self.ss.UCES.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.UCES.update()

        self.ss.UCES.run(solver='SCIP')
        pgs_pqt2 = self.ss.UCES.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    @skip_unittest_without_MISOCP
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.UCES.update()

        self.ss.UCES.run(solver='SCIP')
        self.assertTrue(self.ss.UCES.converged, "UCES did not converge under line trip!")
        plf_l3 = self.ss.UCES.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)
