import unittest
import numpy as np

import ams


class TestRoutineMethods(unittest.TestCase):
    """
    Test methods of `Routine`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("ieee39/ieee39_uced.xlsx"),
                           default_config=True,
                           no_output=True,
                           )

    def test_data_check(self):
        """
        Test `Routine._data_check()` method.
        """

        self.assertTrue(self.ss.DCOPF._data_check())
        self.assertFalse(self.ss.RTEDES._data_check())

    def test_get_off_constrs(self):
        """
        Test `Routine._get_off_constrs()` method.
        """

        self.assertIsInstance(self.ss.DCOPF._get_off_constrs(), list)

    def test_routine_set(self):
        """
        Test `Routine.set()` method.
        """

        self.ss.DCOPF.set('c2', 'GCost_1', 'v', 10)
        np.testing.assert_equal(self.ss.GCost.get('c2', 'GCost_1', 'v'), 10)

    def test_routine_get(self):
        """
        Test `Routine.get()` method.
        """
        # --- single period routine ---
        # get an rparam value
        np.testing.assert_equal(self.ss.DCOPF.get('ug', 'PV_30'), 1)

        # get an unpacked var value
        self.ss.DCOPF.run(solver='ECOS')
        self.assertEqual(self.ss.DCOPF.exit_code, 0, "Exit code is not 0.")
        np.testing.assert_equal(self.ss.DCOPF.get('pg', 'PV_30', 'v'),
                                self.ss.StaticGen.get('p', 'PV_30', 'v'))

        # test return type
        self.assertIsInstance(self.ss.DCOPF.get('pg', 'PV_30', 'v'), float)
        self.assertIsInstance(self.ss.DCOPF.get('pg', ['PV_30'], 'v'), np.ndarray)

        # --- multi period routine ---
        self.ss.ED.run(solver='ECOS')
        self.assertEqual(self.ss.ED.exit_code, 0, "Exit code is not 0.")
        np.testing.assert_equal(self.ss.ED.get('pg', 'PV_30', 'v').ndim, 1)
        np.testing.assert_equal(self.ss.ED.get('pg', ['PV_30'], 'v').ndim, 2)

    def test_rouine_init(self):
        """
        Test `Routine.init()` method.
        """

        self.assertTrue(self.ss.DCOPF.init(), "DCOPF initialization failed!")

    def test_generate_symbols(self):
        """
        Test symbol generation.
        """

        self.ss.DCOPF.syms.generate_symbols()
        self.assertTrue(self.ss.DCOPF._syms, "Symbol generation failed!")

    def test_value_method(self):
        """
        Test Contraint and Objective values.
        """

        self.ss.DCOPF.run(solver='ECOS')
        self.assertTrue(self.ss.DCOPF.converged, "DCOPF did not converge!")

        # --- constraint values ---
        for constr in self.ss.DCOPF.constrs.values():
            np.testing.assert_almost_equal(constr.v, constr.v2, decimal=6)

        # --- objective value ---
        self.assertAlmostEqual(self.ss.DCOPF.obj.v, self.ss.DCOPF.obj.v2, places=6)


class TestOModel(unittest.TestCase):
    """
    Test methods of `RTED`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.xlsx"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])

    def test_trip(self):
        """
        Test generator trip.
        """
        # --- run DCOPF ---
        self.ss.DCOPF.run(solver='ECOS')
        obj = self.ss.DCOPF.obj.v

        # --- generator trip ---
        self.ss.StaticGen.set(src='u', attr='v', idx='PV_1', value=0)

        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='ECOS')
        self.assertTrue(self.ss.DCOPF.converged, "DCOPF did not converge under generator trip!")
        obj_gt = self.ss.DCOPF.obj.v
        self.assertGreater(obj_gt, obj)

        pg_trip = self.ss.DCOPF.get(src='pg', attr='v', idx='PV_1')
        np.testing.assert_almost_equal(pg_trip, 0, decimal=6)

        # --- trip line ---
        self.ss.Line.set(src='u', attr='v', idx='Line_4', value=0)

        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='ECOS')
        self.assertTrue(self.ss.DCOPF.converged, "DCOPF did not converge under line trip!")
        obj_lt = self.ss.DCOPF.obj.v
        self.assertGreater(obj_lt, obj_gt)

        plf_trip = self.ss.DCOPF.get(src='plf', attr='v', idx='Line_4')
        np.testing.assert_almost_equal(plf_trip, 0, decimal=6)


class TestRTED(unittest.TestCase):
    """
    Test methods of `RTED`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.xlsx"),
                           setup=True,
                           default_config=True,
                           no_output=True,
                           )
        self.ss.RTED.run(solver='ECOS')

    def test_dc2ac(self):
        """
        Test `RTED.init()` method.
        """
        self.ss.RTED.dc2ac()
        self.assertTrue(self.ss.RTED.converted, "AC conversion failed!")

        stg_idx = self.ss.StaticGen.get_idx()
        pg_rted = self.ss.RTED.get(src='pg', attr='v', idx=stg_idx)
        pg_acopf = self.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx)
        np.testing.assert_almost_equal(pg_rted, pg_acopf, decimal=3)

        bus_idx = self.ss.Bus.get_idx()
        v_rted = self.ss.RTED.get(src='vBus', attr='v', idx=bus_idx)
        v_acopf = self.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(v_rted, v_acopf, decimal=3)

        a_rted = self.ss.RTED.get(src='aBus', attr='v', idx=bus_idx)
        a_acopf = self.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx)
        np.testing.assert_almost_equal(a_rted, a_acopf, decimal=3)
