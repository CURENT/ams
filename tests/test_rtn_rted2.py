import unittest
import numpy as np

import ams
from ams.shared import skip_unittest_without_MISOCP


class TestRTED2(unittest.TestCase):
    """
    Test methods of `RTED2`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        # build PTDF
        self.ss.mats.build_ptdf()

    def test_init(self):
        """
        Test `RTED2.init()` method.
        """
        self.ss.RTED2.init()
        self.assertTrue(self.ss.RTED2.initialized, "RTED2 initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.RTED2.update()
        self.ss.RTED2.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTED2.converged, "RTED did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.RTED2.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.RTED2.update()
        self.ss.RTED2.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTED2.converged, "RTED did not converge under line trip!")
        self.assertAlmostEqual(self.ss.RTED2.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.RTED2.run(solver='CLARABEL')
        pgs = self.ss.RTED2.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.RTED2.update()

        self.ss.RTED2.run(solver='CLARABEL')
        pgs_pqt = self.ss.RTED2.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.RTED2.update()

        self.ss.RTED2.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.RTED2.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    def test_dc2ac(self):
        """
        Test `RTED2.dc2ac()` method.
        """
        self.ss.RTED2.run(solver='CLARABEL')
        self.ss.RTED2.dc2ac()
        self.assertTrue(self.ss.RTED2.converted, "AC conversion failed!")
        self.assertTrue(self.ss.RTED2.exec_time > 0, "Execution time is not greater than 0.")

        stg_idx = self.ss.StaticGen.get_all_idxes()
        pg_rted = self.ss.RTED2.get(src='pg', attr='v', idx=stg_idx)
        pg_acopf = self.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx)
        np.testing.assert_almost_equal(pg_rted, pg_acopf, decimal=3)

        bus_idx = self.ss.Bus.get_all_idxes()
        v_rted = self.ss.RTED2.get(src='vBus', attr='v', idx=bus_idx)
        v_acopf = self.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(v_rted, v_acopf, decimal=3)

        a_rted = self.ss.RTED2.get(src='aBus', attr='v', idx=bus_idx)
        a_acopf = self.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(a_rted, a_acopf, decimal=3)

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


class TestRTEDDG2(unittest.TestCase):
    """
    Test routine `RTEDDG2`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        self.ss.mats.build_ptdf()

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.RTEDDG2.init()
        self.assertTrue(self.ss.RTEDDG2.initialized, "RTEDDG initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.RTEDDG2.update()
        self.ss.RTEDDG2.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTEDDG2.converged, "RTEDDG did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.RTEDDG2.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.RTEDDG2.update()
        self.ss.RTEDDG2.run(solver='CLARABEL')
        self.assertTrue(self.ss.RTEDDG2.converged, "RTEDDG did not converge under line trip!")
        self.assertAlmostEqual(self.ss.RTEDDG2.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.RTEDDG2.run(solver='CLARABEL')
        pgs = self.ss.RTEDDG2.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.RTEDDG2.update()

        self.ss.RTEDDG2.run(solver='CLARABEL')
        pgs_pqt = self.ss.RTEDDG2.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.RTEDDG2.update()

        self.ss.RTEDDG2.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.RTEDDG2.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    def test_dc2ac(self):
        """
        Test `RTEDDG2.dc2ac()` method.
        """
        self.ss.RTEDDG2.run(solver='CLARABEL')
        self.ss.RTEDDG2.dc2ac()
        self.assertTrue(self.ss.RTEDDG2.converted, "AC conversion failed!")
        self.assertTrue(self.ss.RTEDDG2.exec_time > 0, "Execution time is not greater than 0.")

        stg_idx = self.ss.StaticGen.get_all_idxes()
        pg_rted = self.ss.RTEDDG2.get(src='pg', attr='v', idx=stg_idx)
        pg_acopf = self.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx)
        np.testing.assert_almost_equal(pg_rted, pg_acopf, decimal=3)

        bus_idx = self.ss.Bus.get_all_idxes()
        v_rted = self.ss.RTEDDG2.get(src='vBus', attr='v', idx=bus_idx)
        v_acopf = self.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(v_rted, v_acopf, decimal=3)

        a_rted = self.ss.RTEDDG2.get(src='aBus', attr='v', idx=bus_idx)
        a_acopf = self.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(a_rted, a_acopf, decimal=3)

    def test_align_rteddg2(self):
        """
        Test if results align with RTEDDG.
        """
        self.ss.RTEDDG.run(solver='CLARABEL')
        self.ss.RTEDDG2.run(solver='CLARABEL')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.RTEDDG.obj.v,
                                       self.ss.RTEDDG2.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTEDDG2 and RTEDDG not match!")

        pg = self.ss.RTEDDG.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.RTEDDG2.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="pg between RTEDDG2 and RTEDDG not match!")

        aBus = self.ss.RTEDDG.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.RTEDDG2.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="aBus between RTEDDG2 and RTEDDG not match!")

        pi = self.ss.RTEDDG.get(src='pi', attr='v', idx=bus_idx)
        pi2 = self.ss.RTEDDG2.get(src='pi', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(pi, pi2, decimal=DECIMALS,
                                       err_msg="pi between RTEDDG2 and RTEDDG not match!")

        plf = self.ss.RTEDDG.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.RTEDDG2.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="plf between RTEDDG2 and RTEDDG not match!")


class TestRTEDES2(unittest.TestCase):
    """
    Test routine `RTEDES2`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
        self.ss.mats.build_ptdf()

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.RTEDES2.init()
        self.assertTrue(self.ss.RTEDES2.initialized, "RTEDES initialization failed!")

    @skip_unittest_without_MISOCP
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.RTEDES2.update()
        self.ss.RTEDES2.run(solver='SCIP')
        self.assertTrue(self.ss.RTEDES2.converged, "RTEDES did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.RTEDES2.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)

    @skip_unittest_without_MISOCP
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.RTEDES2.update()
        self.ss.RTEDES2.run(solver='SCIP')
        self.assertTrue(self.ss.RTEDES2.converged, "RTEDES did not converge under line trip!")
        self.assertAlmostEqual(self.ss.RTEDES2.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    @skip_unittest_without_MISOCP
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        self.ss.RTEDES2.run(solver='SCIP')
        pgs = self.ss.RTEDES2.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.05)
        self.ss.RTEDES2.update()

        self.ss.RTEDES2.run(solver='SCIP')
        pgs_pqt = self.ss.RTEDES2.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.RTEDES2.update()

        self.ss.RTEDES2.run(solver='SCIP')
        pgs_pqt2 = self.ss.RTEDES2.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    @skip_unittest_without_MISOCP
    def test_dc2ac(self):
        """
        Test DC to AC conversion.
        """
        self.ss.RTEDES2.run(solver='SCIP')
        self.ss.RTEDES2.dc2ac()
        self.assertTrue(self.ss.RTEDES2.converted, "AC conversion failed!")
        self.assertTrue(self.ss.RTEDES2.exec_time > 0, "Execution time is not greater than 0.")

        stg_idx = self.ss.StaticGen.get_all_idxes()
        pg_rted = self.ss.RTEDES2.get(src='pg', attr='v', idx=stg_idx)
        pg_acopf = self.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx)
        np.testing.assert_almost_equal(pg_rted, pg_acopf, decimal=3)

        bus_idx = self.ss.Bus.get_all_idxes()
        v_rted = self.ss.RTEDES2.get(src='vBus', attr='v', idx=bus_idx)
        v_acopf = self.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(v_rted, v_acopf, decimal=3)

        a_rted = self.ss.RTEDES2.get(src='aBus', attr='v', idx=bus_idx)
        a_acopf = self.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(a_rted, a_acopf, decimal=3)

    @skip_unittest_without_MISOCP
    def test_align_rtedes(self):
        """
        Test if results align with RTEDES.
        """
        self.ss.RTEDES.run(solver='SCIP')
        self.ss.RTEDES2.run(solver='SCIP')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.RTEDES.obj.v,
                                       self.ss.RTEDES2.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTEDES2 and RTEDES not match!")

        pg = self.ss.RTEDES.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.RTEDES2.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="pg between RTEDES2 and RTEDES not match!")

        aBus = self.ss.RTEDES.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.RTEDES2.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="aBus between RTEDES2 and RTEDES not match!")

        plf = self.ss.RTEDES.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.RTEDES2.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="plf between RTEDES2 and RTEDES not match!")

    def test_align_rtedesp(self):
        """
        Test if results align with RTEDESP.
        """
        self.ss.RTEDES2.run(solver='SCIP')
        self.ss.RTEDESP.run(solver='CLARABEL')

        pg_idx = self.ss.StaticGen.get_all_idxes()
        bus_idx = self.ss.Bus.idx.v
        line_idx = self.ss.Line.idx.v

        DECIMALS = 4

        np.testing.assert_almost_equal(self.ss.RTEDES2.obj.v,
                                       self.ss.RTEDESP.obj.v,
                                       decimal=DECIMALS,
                                       err_msg="Objective value between RTEDES2 and RTEDESP not match!")

        pg = self.ss.RTEDES2.get(src='pg', attr='v', idx=pg_idx)
        pg2 = self.ss.RTEDESP.get(src='pg', attr='v', idx=pg_idx)
        np.testing.assert_almost_equal(pg, pg2, decimal=DECIMALS,
                                       err_msg="pg between RTEDES2 and RTEDESP not match!")

        aBus = self.ss.RTEDES2.get(src='aBus', attr='v', idx=bus_idx)
        aBus2 = self.ss.RTEDESP.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(aBus, aBus2, decimal=DECIMALS,
                                       err_msg="aBus between RTEDES2 and RTEDESP not match!")

        plf = self.ss.RTEDES2.get(src='plf', attr='v', idx=line_idx)
        plf2 = self.ss.RTEDESP.get(src='plf', attr='v', idx=line_idx)
        np.testing.assert_almost_equal(plf, plf2, decimal=DECIMALS,
                                       err_msg="plf between RTEDES2 and RTEDESP not match!")
