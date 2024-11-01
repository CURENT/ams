import unittest
import numpy as np

import ams


class TestED(unittest.TestCase):
    """
    Test routine `ED`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.xlsx"),
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
        stg_uid = self.ss.ED.pg.get_idx().index(stg)
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
        self.ss.StaticGen.set(src='u', attr='v', idx=stg, value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED.converged, "ED did not converge under generator trip!")
        pg_pv1 = self.ss.ED.get(src='pg', attr='v', idx=stg)
        np.testing.assert_array_less(np.zeros_like(pg_pv1), pg_pv1,
                                     err_msg="Generator trip take effect, which is unexpected!")

    def test_trip(self):
        """
        Test generator trip.
        """
        self.ss.ED.run(solver='CLARABEL')
        obj = self.ss.ED.obj.v

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        obj_pqt = self.ss.ED.obj.v
        self.assertLess(obj_pqt, obj, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        obj_pqt2 = self.ss.ED.obj.v
        self.assertLess(obj_pqt2, obj_pqt, "Load trip does not take effect!")

        # --- trip line ---
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
        self.ss.ED.update()

        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.ED.converged, "ED did not converge under line trip!")
        plf_l3 = self.ss.ED.get(src='plf', attr='v', idx='Line_3')
        np.testing.assert_almost_equal(np.zeros_like(plf_l3),
                                       plf_l3, decimal=6)

        # # --- trip generator ---
        # # TODO: DEBUG: NOT SURE THIS TEST FAILED
        # # b) ensure EDTSlot.ug takes effect
        # # NOTE: manually chang ug.v for testing purpose
        # stg = 'PV_1'
        # stg_uid = self.ss.ED.pg.get_idx().index(stg)
        # loc_offtime = np.array([0, 2, 4])
        # self.ss.EDTSlot.ug.v[loc_offtime, stg_uid] = 0

        # self.ss.ED.run(solver='CLARABEL')
        # self.assertTrue(self.ss.ED.converged, "ED did not converge under generator trip!")
        # pg_pv1 = self.ss.ED.get(src='pg', attr='v', idx=stg)
        # print('stg_uid', stg_uid)
        # print(self.ss.EDTSlot.ug.v)
        # print(pg_pv1)
        # np.testing.assert_almost_equal(np.zeros_like(loc_offtime),
        #                                pg_pv1[loc_offtime],
        #                                decimal=6,
        #                                err_msg="Generator trip does not take effect!")

        # # a) ensure StaticGen.u does not take effect
        # # NOTE: in ED, `EDTSlot.ug` is used instead of `StaticGen.u`
        # self.ss.StaticGen.set(src='u', attr='v', idx=stg, value=0)
        # self.ss.ED.update()

        # self.ss.ED.run(solver='CLARABEL')
        # self.assertTrue(self.ss.ED.converged, "ED did not converge under generator trip!")
        # pg_pv1 = self.ss.ED.get(src='pg', attr='v', idx=stg)
        # np.testing.assert_array_less(np.zeros_like(pg_pv1), pg_pv1,
        #                              err_msg="Generator trip take effect, which is unexpected!")
