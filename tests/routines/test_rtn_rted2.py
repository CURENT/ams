import unittest

import numpy as np

import ams
from ams.shared import skip_unittest_without_MISOCP


class TestRTED2(unittest.TestCase):
    """
    Test routine `RTED2`.

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

    def test_align_rted(self):
        """
        Test if results align with RTED.
        """
        self.ss.RTED.run(solver='CLARABEL')
        self.ss.RTED2.run(solver='CLARABEL')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.RTED.obj.v,
                                       self.ss.RTED2.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTED2 and RTED not match!")

        pg = self.ss.RTED.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.RTED2.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="pg between RTED2 and RTED not match!")

        aBus = self.ss.RTED.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.RTED2.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="aBus between RTED2 and RTED not match!")

    def test_pb_formula(self):
        """
        Test the pb formula is not the angle-based formulation.
        """
        self.assertFalse('aBus' in self.ss.RTED2.pb.e_str, "Bus angle is used in RTED2.pb!")


class TestRTED2DG(unittest.TestCase):
    """
    Test routine `RTED2DG`.

    Common scenarios (init, trip_gen, trip_line, set_load, vBus, dc2ac)
    are exercised in `tests/routines/test_scenarios_lp_singleperiod.py`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        self.ss.mats.build_ptdf()

    def test_align_RTED2DG(self):
        """
        Test if results align with RTEDDG.
        """
        self.ss.RTEDDG.run(solver='CLARABEL')
        self.ss.RTED2DG.run(solver='CLARABEL')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.RTEDDG.obj.v,
                                       self.ss.RTED2DG.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTED2DG and RTEDDG not match!")

        pg = self.ss.RTEDDG.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.RTED2DG.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="pg between RTED2DG and RTEDDG not match!")

        aBus = self.ss.RTEDDG.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.RTED2DG.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="aBus between RTED2DG and RTEDDG not match!")

        pi = self.ss.RTEDDG.get(src='pi', attr='v', idx=bus_idx)
        pi2 = self.ss.RTED2DG.get(src='pi', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(pi, pi2, decimal=DECIMALS,
                                       err_msg="pi between RTED2DG and RTEDDG not match!")

        plf = self.ss.RTEDDG.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.RTED2DG.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="plf between RTED2DG and RTEDDG not match!")

    def test_pb_formula(self):
        """
        Test the pb formula is not the angle-based formulation.
        """
        self.assertFalse('aBus' in self.ss.RTED2DG.pb.e_str, "Bus angle is used in RTED2DG.pb!")


class TestRTED2ES(unittest.TestCase):
    """
    Test routine `RTED2ES`.

    Common scenarios (init, trip_gen, trip_line, set_load, vBus, dc2ac)
    are exercised in `tests/routines/test_scenarios_lp_singleperiod.py`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        self.ss.mats.build_ptdf()

    @skip_unittest_without_MISOCP
    def test_align_rtedes(self):
        """
        Test if results align with RTEDES.
        """
        self.ss.RTEDES.run(solver='SCIP')
        self.ss.RTED2ES.run(solver='SCIP')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.RTEDES.obj.v,
                                       self.ss.RTED2ES.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTED2ES and RTEDES not match!")

        pg = self.ss.RTEDES.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.RTED2ES.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="pg between RTED2ES and RTEDES not match!")

        aBus = self.ss.RTEDES.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.RTED2ES.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="aBus between RTED2ES and RTEDES not match!")

        plf = self.ss.RTEDES.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.RTED2ES.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="plf between RTED2ES and RTEDES not match!")

    @skip_unittest_without_MISOCP
    def test_align_rtedesp(self):
        """
        Test if results align with RTEDESP.
        """
        self.ss.RTED2ES.run(solver='SCIP')
        self.ss.RTED2ESP.run(solver='CLARABEL')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.RTED2ES.obj.v,
                                       self.ss.RTED2ESP.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTED2ES and RTEDESP not match!")

        pg = self.ss.RTED2ES.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.RTED2ESP.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="pg between RTED2ES and RTEDESP not match!")

        aBus = self.ss.RTED2ES.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.RTED2ESP.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="aBus between RTED2ES and RTEDESP not match!")

        plf = self.ss.RTED2ES.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.RTED2ESP.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="plf between RTED2ES and RTEDESP not match!")

    def test_pb_formula(self):
        """
        Test the pb formula is not the angle-based formulation.
        """
        self.assertFalse('aBus' in self.ss.RTED2ES.pb.e_str, "Bus angle is used in RTED2ES.pb!")
