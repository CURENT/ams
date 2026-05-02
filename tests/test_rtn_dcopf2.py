import unittest

import numpy as np

import ams


class TestDCOPF2(unittest.TestCase):
    """
    Test routine `DCOPF2`.

    Common scenarios (init, trip_gen, trip_line, set_load, vBus, dc2ac)
    are exercised in `tests/routines/test_scenarios_lp_singleperiod.py`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        # build PTDF
        self.ss.mats.build_ptdf()

    def test_align_dcopf(self):
        """
        Test if results align with DCOPF.
        """
        self.ss.DCOPF.run(solver='CLARABEL')
        self.ss.mats.build_ptdf()
        self.ss.DCOPF2.run(solver='CLARABEL')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.DCOPF.obj.v,
                                       self.ss.DCOPF2.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between DCOPF2 and DCOPF not match!")

        pg = self.ss.DCOPF.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.DCOPF2.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="pg between DCOPF2 and DCOPF not match!")

        aBus = self.ss.DCOPF.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.DCOPF2.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="aBus between DCOPF2 and DCOPF not match!")

        pi = self.ss.DCOPF.get(src='pi', attr='v', idx=bus_idx)
        pi2 = self.ss.DCOPF2.get(src='pi', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(pi, pi2, decimal=DECIMALS,
                                       err_msg="pi between DCOPF2 and DCOPF not match!")

        plf = self.ss.DCOPF.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.DCOPF2.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="plf between DCOPF2 and DCOPF not match!")

    def test_pb_formula(self):
        """
        Test the pb formula is not the angle-based formulation.
        """
        self.assertFalse('aBus' in self.ss.DCOPF2.pb.e_str, "Bus angle is used in DCOPF2.pb!")
