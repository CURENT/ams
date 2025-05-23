import unittest
import numpy as np

import ams
from ams.shared import skip_unittest_without_MISOCP


class TestED(unittest.TestCase):
    """
    Test routine `ED`.
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
        self.ss.ED.init()
        self.assertTrue(self.ss.ED.initialized, "ED initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        # a) ensure EDTSlot.ug takes effect
        # NOTE: manually chang ug.v for testing purpose
        stg = 'PV_1'
        stg_uid = self.ss.ED.pg.get_all_idxes().index(stg)
        loc_offtime = np.array([0, 2, 4])
        self.ss.EDTSlot.ug.v[loc_offtime, stg_uid] = 0

        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED.get(src='pg', attr='v', idx=stg)
        np.testing.assert_almost_equal(np.zeros_like(loc_offtime),
                                       pg_pv1[loc_offtime],
                                       decimal=6,
                                       err_msg="Generator trip does not take effect!")

        self.ss.EDTSlot.ug.v[...] = 1  # reset

        # b) ensure StaticGen.u does not take effect
        # NOTE: in ED, `EDTSlot.ug` is used instead of `StaticGen.u`
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED.get(src='pg', attr='v', idx=stg)
        np.testing.assert_array_less(np.zeros_like(pg_pv1), pg_pv1,
                                     err_msg="Generator trip take effect, which is unexpected!")

        self.ss.StaticGen.set(src='u', idx=stg,  attr='v', value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED.converged, "ED did not converge under line trip!")
        plf_l3 = self.ss.ED.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.ED.run(solver='CLARABEL')
        pgs = self.ss.ED.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        pgs_pqt = self.ss.ED.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.ED.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")


class TestEDDG(unittest.TestCase):
    """
    Test routine `EDDG`.
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
        self.ss.EDDG.init()
        self.assertTrue(self.ss.EDDG.initialized, "EDDG initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        # a) ensure EDTSlot.ug takes effect
        # NOTE: manually chang ug.v for testing purpose
        stg = 'PV_1'
        stg_uid = self.ss.EDDG.pg.get_all_idxes().index(stg)
        loc_offtime = np.array([0, 2, 4])
        self.ss.EDTSlot.ug.v[loc_offtime, stg_uid] = 0

        self.ss.EDDG.run(solver='CLARABEL')
        self.assertTrue(self.ss.EDDG.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.EDDG.get(src='pg', attr='v', idx=stg)
        np.testing.assert_almost_equal(np.zeros_like(loc_offtime),
                                       pg_pv1[loc_offtime],
                                       decimal=6,
                                       err_msg="Generator trip does not take effect!")

        self.ss.EDTSlot.ug.v[...] = 1

        # b) ensure StaticGen.u does not take effect
        # NOTE: in EDDG, `EDTSlot.ug` is used instead of `StaticGen.u`
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)
        self.ss.EDDG.update()

        self.ss.EDDG.run(solver='CLARABEL')
        self.assertTrue(self.ss.EDDG.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.EDDG.get(src='pg', attr='v', idx=stg)
        np.testing.assert_array_less(np.zeros_like(pg_pv1), pg_pv1,
                                     err_msg="Generator trip take effect, which is unexpected!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.EDDG.update()

        self.ss.EDDG.run(solver='CLARABEL')
        self.assertTrue(self.ss.EDDG.converged, "ED did not converge under line trip!")
        plf_l3 = self.ss.EDDG.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.EDDG.run(solver='CLARABEL')
        pgs = self.ss.EDDG.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.EDDG.update()

        self.ss.EDDG.run(solver='CLARABEL')
        pgs_pqt = self.ss.EDDG.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.EDDG.update()

        self.ss.EDDG.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.EDDG.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")


class TestEDES(unittest.TestCase):
    """
    Test routine `EDES`.
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
        self.ss.EDES.init()
        self.assertTrue(self.ss.EDES.initialized, "EDES initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        # a) ensure EDTSlot.ug takes effect
        # NOTE: manually chang ug.v for testing purpose
        stg = 'PV_1'
        stg_uid = self.ss.EDES.pg.get_all_idxes().index(stg)
        loc_offtime = np.array([0, 2])
        self.ss.EDTSlot.ug.v[loc_offtime, stg_uid] = 0

        self.ss.EDES.run(solver='SCIP')
        self.assertTrue(self.ss.EDES.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.EDES.get(src='pg', attr='v', idx=stg)
        np.testing.assert_almost_equal(np.zeros_like(loc_offtime),
                                       pg_pv1[loc_offtime],
                                       decimal=6,
                                       err_msg="Generator trip does not take effect!")

        self.ss.EDTSlot.ug.v[...] = 1  # reset

        # b) ensure StaticGen.u does not take effect
        # NOTE: in ED, `EDTSlot.ug` is used instead of `StaticGen.u`
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)
        self.ss.EDES.update()

        self.ss.EDES.run(solver='SCIP')
        self.assertTrue(self.ss.EDES.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.EDES.get(src='pg', attr='v', idx=stg)
        np.testing.assert_array_less(np.zeros_like(pg_pv1), pg_pv1,
                                     err_msg="Generator trip take effect, which is unexpected!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    @skip_unittest_without_MISOCP
    def test_line_trip(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.EDES.run(solver='SCIP')
        self.assertTrue(self.ss.EDES.converged, "EDES did not converge!")
        plf_l3 = self.ss.EDES.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.EDES.run(solver='SCIP')
        pgs = self.ss.EDES.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.EDES.update()

        self.ss.EDES.run(solver='SCIP')
        pgs_pqt = self.ss.EDES.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.EDES.update()

        self.ss.EDES.run(solver='SCIP')
        pgs_pqt2 = self.ss.EDES.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")
