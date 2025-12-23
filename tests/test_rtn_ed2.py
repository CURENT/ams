import unittest
import numpy as np

import ams
from ams.shared import skip_unittest_without_MISOCP


class TestED2(unittest.TestCase):
    """
    Test routine `ED2`.
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
        self.ss.ED2.init()
        self.assertTrue(self.ss.ED2.initialized, "ED initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        # a) ensure EDTSlot.ug takes effect
        # NOTE: manually change ug.v for testing purpose
        stg = 'PV_1'
        stg_uid = self.ss.StaticGen.get_all_idxes().index(stg)
        loc_offtime = np.array([0, 2, 4])
        self.ss.EDTSlot.ug.v[loc_offtime, stg_uid] = 0

        self.ss.ED2.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED2.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED2.get(src='pg', attr='v', idx=stg)
        np.testing.assert_almost_equal(np.zeros_like(loc_offtime),
                                       pg_pv1[loc_offtime],
                                       decimal=6,
                                       err_msg="Generator trip does not take effect!")

        self.ss.EDTSlot.ug.v[...] = 1  # reset

        # b) ensure StaticGen.u does not take effect
        # NOTE: in ED, `EDTSlot.ug` is used instead of `StaticGen.u`
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)
        self.ss.ED2.update()

        self.ss.ED2.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED2.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED2.get(src='pg', attr='v', idx=stg)
        np.testing.assert_array_less(np.zeros_like(pg_pv1), pg_pv1,
                                     err_msg="Generator trip take effect, which is unexpected!")

        self.ss.StaticGen.set(src='u', idx=stg,  attr='v', value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.ED2.update()

        self.ss.ED2.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED2.converged, "ED did not converge under line trip!")
        plf_l3 = self.ss.ED2.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.ED2.run(solver='CLARABEL')
        pgs = self.ss.ED2.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.ED2.update()

        self.ss.ED2.run(solver='CLARABEL')
        pgs_pqt = self.ss.ED2.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.ED2.update()

        self.ss.ED2.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.ED2.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    def test_vBus_aBus(self):
        """
        Test vBus and aBus are not all zero.
        """
        self.ss.ED2.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED2.converged, "ED2 did not converge!")
        self.assertTrue(np.any(self.ss.ED2.vBus.v), "vBus is all zero!")
        self.assertTrue(np.any(self.ss.ED2.aBus.v), "aBus is all zero!")

    def test_align_ed(self):
        """
        Test if results align with ED.
        """
        self.ss.ED2.run(solver='CLARABEL')
        self.ss.ED.run(solver='CLARABEL')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.ED2.obj.v,
                                       self.ss.ED.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between ED2 and ED not match!")

        pg = self.ss.ED.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.ED2.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="Generator power between ED2 and ED not match!")

        aBus = self.ss.ED.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.ED2.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="Bus angle between ED2 and ED not match!")

        plf = self.ss.ED.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.ED2.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="Line flow between ED2 and ED not match!")

    def test_pb_formula(self):
        """
        Test the pb formula is not the angle-based formulation.
        """
        self.assertFalse('aBus' in self.ss.ED2.pb.e_str, "Bus angle is used in ED2.pb!")


class TestED2DG(unittest.TestCase):
    """
    Test routine `ED2DG`.
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
        self.ss.ED2DG.init()
        self.assertTrue(self.ss.ED2DG.initialized, "ED initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        # a) ensure EDTSlot.ug takes effect
        # NOTE: manually change ug.v for testing purpose
        stg = 'PV_1'
        stg_uid = self.ss.StaticGen.get_all_idxes().index(stg)
        loc_offtime = np.array([0, 2, 4])
        self.ss.EDTSlot.ug.v[loc_offtime, stg_uid] = 0

        self.ss.ED2DG.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED2DG.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED2DG.get(src='pg', attr='v', idx=stg)
        np.testing.assert_almost_equal(np.zeros_like(loc_offtime),
                                       pg_pv1[loc_offtime],
                                       decimal=6,
                                       err_msg="Generator trip does not take effect!")

        self.ss.EDTSlot.ug.v[...] = 1  # reset

        # b) ensure StaticGen.u does not take effect
        # NOTE: in ED, `EDTSlot.ug` is used instead of `StaticGen.u`
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)
        self.ss.ED2DG.update()

        self.ss.ED2DG.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED2DG.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED2DG.get(src='pg', attr='v', idx=stg)
        np.testing.assert_array_less(np.zeros_like(pg_pv1), pg_pv1,
                                     err_msg="Generator trip take effect, which is unexpected!")

        self.ss.StaticGen.set(src='u', idx=stg,  attr='v', value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.ED2DG.update()

        self.ss.ED2DG.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED2DG.converged, "ED did not converge under line trip!")
        plf_l3 = self.ss.ED2DG.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.ED2DG.run(solver='CLARABEL')
        pgs = self.ss.ED2DG.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.ED2DG.update()

        self.ss.ED2DG.run(solver='CLARABEL')
        pgs_pqt = self.ss.ED2DG.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.ED2DG.update()

        self.ss.ED2DG.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.ED2DG.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    def test_vBus_aBus(self):
        """
        Test vBus and aBus are not all zero.
        """
        self.ss.ED2DG.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED2DG.converged, "ED2DG did not converge!")
        self.assertTrue(np.any(self.ss.ED2DG.vBus.v), "vBus is all zero!")
        self.assertTrue(np.any(self.ss.ED2DG.aBus.v), "aBus is all zero!")

    def test_align_ED2DG(self):
        """
        Test if results align with ED.
        """
        self.ss.ED2DG.run(solver='CLARABEL')
        self.ss.EDDG.run(solver='CLARABEL')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.ED2DG.obj.v,
                                       self.ss.EDDG.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between ED2DG and EDDG not match!")

        pg = self.ss.EDDG.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.ED2DG.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="Generator power between ED2DG and EDDG not match!")

        aBus = self.ss.EDDG.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.ED2DG.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="Bus angle between ED2DG and EDDG not match!")

        plf = self.ss.EDDG.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.ED2DG.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="Line flow between ED2DG and EDDG not match!")

    def test_pb_formula(self):
        """
        Test the pb formula is not the angle-based formulation.
        """
        self.assertFalse('aBus' in self.ss.ED2DG.pb.e_str, "Bus angle is used in ED2DG.pb!")


class TestED2ES(unittest.TestCase):
    """
    Test routine `ED2ES`.
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
        self.ss.ED2ES.init()
        self.assertTrue(self.ss.ED2ES.initialized, "ED initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        # a) ensure EDTSlot.ug takes effect
        # NOTE: manually change ug.v for testing purpose
        stg = 'PV_1'
        stg_uid = self.ss.StaticGen.get_all_idxes().index(stg)
        loc_offtime = np.array([0, 2, 4])
        self.ss.EDTSlot.ug.v[loc_offtime, stg_uid] = 0

        self.ss.ED2ES.run(solver='SCIP')
        self.assertTrue(self.ss.ED2ES.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED2ES.get(src='pg', attr='v', idx=stg)
        np.testing.assert_almost_equal(np.zeros_like(loc_offtime),
                                       pg_pv1[loc_offtime],
                                       decimal=6,
                                       err_msg="Generator trip does not take effect!")

        self.ss.EDTSlot.ug.v[...] = 1  # reset

        # b) ensure StaticGen.u does not take effect
        # NOTE: in ED, `EDTSlot.ug` is used instead of `StaticGen.u`
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)
        self.ss.ED2ES.update()

        self.ss.ED2ES.run(solver='SCIP')
        self.assertTrue(self.ss.ED2ES.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED2ES.get(src='pg', attr='v', idx=stg)
        np.testing.assert_array_less(np.zeros_like(pg_pv1), pg_pv1,
                                     err_msg="Generator trip take effect, which is unexpected!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    @skip_unittest_without_MISOCP
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.ED2ES.update()

        self.ss.ED2ES.run(solver='SCIP')
        self.assertTrue(self.ss.ED2ES.converged, "ED did not converge under line trip!")
        plf_l3 = self.ss.ED2ES.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.ED2ES.run(solver='SCIP')
        pgs = self.ss.ED2ES.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.ED2ES.update()

        self.ss.ED2ES.run(solver='SCIP')
        pgs_pqt = self.ss.ED2ES.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.alter(src='u', idx='PQ_2', value=0)
        self.ss.ED2ES.update()

        self.ss.ED2ES.run(solver='SCIP')
        pgs_pqt2 = self.ss.ED2ES.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    @skip_unittest_without_MISOCP
    def test_vBus_aBus(self):
        """
        Test vBus and aBus are not all zero.
        """
        self.ss.ED2ES.run(solver='SCIP')
        self.assertTrue(self.ss.ED2ES.converged, "ED2ES did not converge!")
        self.assertTrue(np.any(self.ss.ED2ES.vBus.v), "vBus is all zero!")
        self.assertTrue(np.any(self.ss.ED2ES.aBus.v), "aBus is all zero!")

    @skip_unittest_without_MISOCP
    def test_align_edes(self):
        """
        Test if results align with EDES.
        """
        self.ss.ED2ES.run(solver='SCIP')
        self.ss.EDES.run(solver='SCIP')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 3

        np.testing.assert_almost_equal(self.ss.ED2ES.obj.v,
                                       self.ss.EDES.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between ED2ES and EDES not match!")

        pg = self.ss.EDES.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.ED2ES.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="Generator power between ED2ES and EDES not match!")

        aBus = self.ss.EDES.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.ED2ES.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="Bus angle between ED2ES and EDES not match!")

        plf = self.ss.EDES.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.ED2ES.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="Line flow between ED2ES and EDES not match!")

    def test_pb_formula(self):
        """
        Test the pb formula is not the angle-based formulation.
        """
        self.assertFalse('aBus' in self.ss.ED2ES.pb.e_str, "Bus angle is used in ED2ES.pb!")
