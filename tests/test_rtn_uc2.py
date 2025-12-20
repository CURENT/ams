import unittest
import numpy as np

import ams
from ams.shared import skip_unittest_without_MISOCP


class TestUC2(unittest.TestCase):
    """
    Test routine `UC2`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        # run UC2._initial_guess()
        self.off_gen = self.ss.UC2._initial_guess()

    def test_initial_guess(self):
        """
        Test initial guess.
        """
        u_off_gen = self.ss.StaticGen.get(src='u', idx=self.off_gen)
        np.testing.assert_equal(u_off_gen, np.zeros_like(u_off_gen),
                                err_msg="UC2._initial_guess() failed!")

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.UC2.init()
        self.assertTrue(self.ss.UC2.initialized, "UC initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        self.ss.UC2.run(solver='SCIP')
        self.assertTrue(self.ss.UC2.converged, "UC did not converge!")
        pg_off_gen = self.ss.UC2.get(src='pg', attr='v', idx=self.off_gen)
        np.testing.assert_almost_equal(np.zeros_like(pg_off_gen),
                                       pg_off_gen, decimal=6,
                                       err_msg="Off generators are not turned off!")

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.UC2.run(solver='SCIP')
        pgs = self.ss.UC2.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.UC2.update()

        self.ss.UC2.run(solver='SCIP')
        pgs_pqt = self.ss.UC2.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.UC2.update()

        self.ss.UC2.run(solver='SCIP')
        pgs_pqt2 = self.ss.UC2.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    @skip_unittest_without_MISOCP
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.UC2.update()

        self.ss.UC2.run(solver='SCIP')
        self.assertTrue(self.ss.UC2.converged, "UC did not converge under line trip!")
        plf_l3 = self.ss.UC2.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    @skip_unittest_without_MISOCP
    def test_align_uc(self):
        """
        Test if results align with UC.
        """
        self.ss.UC.run(solver='SCIP')
        self.ss.UC2.run(solver='SCIP')

        pg_idx = self.ss.StaticGen.get_all_idxes()

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.UC.obj.v,
                                       self.ss.UC2.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTED2ES and RTEDESP not match!")

        ugd = self.ss.UC.get(src='ugd', attr='v', idx=pg_idx)
        ugd2 = self.ss.UC2.get(src='ugd', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(ugd, ugd2, decimal=DECIMALS,
                                       err_msg="ugd between RTED2ES and RTEDESP not match!")


class TestUC2DG(unittest.TestCase):
    """
    Test routine `UC2DG`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        # run `_initial_guess()`
        self.off_gen = self.ss.UC2DG._initial_guess()

    def test_initial_guess(self):
        """
        Test initial guess.
        """
        u_off_gen = self.ss.StaticGen.get(src='u', idx=self.off_gen)
        np.testing.assert_equal(u_off_gen, np.zeros_like(u_off_gen),
                                err_msg="UC2DG._initial_guess() failed!")

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.UC2DG.init()
        self.assertTrue(self.ss.UC2DG.initialized, "UC2DG initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        self.ss.UC2DG.run(solver='SCIP')
        self.assertTrue(self.ss.UC2DG.converged, "UC2DG did not converge!")
        pg_off_gen = self.ss.UC2DG.get(src='pg', attr='v', idx=self.off_gen)
        np.testing.assert_almost_equal(np.zeros_like(pg_off_gen),
                                       pg_off_gen, decimal=6,
                                       err_msg="Off generators are not turned off!")

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.UC2DG.run(solver='SCIP')
        pgs = self.ss.UC2DG.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.UC2DG.update()

        self.ss.UC2DG.run(solver='SCIP')
        pgs_pqt = self.ss.UC2DG.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.UC2DG.update()

        self.ss.UC2DG.run(solver='SCIP')
        pgs_pqt2 = self.ss.UC2DG.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    @skip_unittest_without_MISOCP
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.UC2DG.update()

        self.ss.UC2DG.run(solver='SCIP')
        self.assertTrue(self.ss.UC2DG.converged, "UC2DG did not converge under line trip!")
        plf_l3 = self.ss.UC2DG.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    @skip_unittest_without_MISOCP
    def test_align_ucdg(self):
        """
        Test if results align with UCDG.
        """
        self.ss.UC2DG.run(solver='SCIP')
        self.ss.UCDG.run(solver='SCIP')

        pg_idx = self.ss.StaticGen.get_all_idxes()

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.UC2DG.obj.v,
                                       self.ss.UCDG.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTED2ES and RTEDESP not match!")

        ugd = self.ss.UC2DG.get(src='ugd', attr='v', idx=pg_idx)
        ugd2 = self.ss.UCDG.get(src='ugd', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(ugd, ugd2, decimal=DECIMALS,
                                       err_msg="ugd between RTED2ES and RTEDESP not match!")


class TestUC2ES(unittest.TestCase):
    """
    Test routine `UC2ES`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        # run `_initial_guess()`
        self.off_gen = self.ss.UC2ES._initial_guess()

    def test_initial_guess(self):
        """
        Test initial guess.
        """
        u_off_gen = self.ss.StaticGen.get(src='u', idx=self.off_gen)
        np.testing.assert_equal(u_off_gen, np.zeros_like(u_off_gen),
                                err_msg="UC2ES._initial_guess() failed!")

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.UC2ES.init()
        self.assertTrue(self.ss.UC2ES.initialized, "UC2ES initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        self.ss.UC2ES.run(solver='SCIP')
        self.assertTrue(self.ss.UC2ES.converged, "UC2ES did not converge!")
        pg_off_gen = self.ss.UC2ES.get(src='pg', attr='v', idx=self.off_gen)
        np.testing.assert_almost_equal(np.zeros_like(pg_off_gen),
                                       pg_off_gen, decimal=6,
                                       err_msg="Off generators are not turned off!")

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.UC2ES.run(solver='SCIP')
        pgs = self.ss.UC2ES.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.UC2ES.update()

        self.ss.UC2ES.run(solver='SCIP')
        pgs_pqt = self.ss.UC2ES.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.UC2ES.update()

        self.ss.UC2ES.run(solver='SCIP')
        pgs_pqt2 = self.ss.UC2ES.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    @skip_unittest_without_MISOCP
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.UC2ES.update()

        self.ss.UC2ES.run(solver='SCIP')
        self.assertTrue(self.ss.UC2ES.converged, "UC2ES did not converge under line trip!")
        plf_l3 = self.ss.UC2ES.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    @skip_unittest_without_MISOCP
    def test_align_uces(self):
        """
        Test if results align with UCES.
        """
        self.ss.UC2ES.run(solver='SCIP')
        self.ss.UCES.run(solver='SCIP')

        pg_idx = self.ss.StaticGen.get_all_idxes()

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.UC2ES.obj.v,
                                       self.ss.UCES.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTED2ES and RTEDESP not match!")

        ugd = self.ss.UC2ES.get(src='ugd', attr='v', idx=pg_idx)
        ugd2 = self.ss.UCES.get(src='ugd', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(ugd, ugd2, decimal=DECIMALS,
                                       err_msg="ugd between RTED2ES and RTEDESP not match!")
