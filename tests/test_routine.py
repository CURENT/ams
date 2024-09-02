import unittest
import numpy as np

from andes.shared import pd

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
        self.ss.DCOPF.run(solver='CLARABEL')
        self.assertEqual(self.ss.DCOPF.exit_code, 0, "Exit code is not 0.")
        np.testing.assert_equal(self.ss.DCOPF.get('pg', 'PV_30', 'v'),
                                self.ss.StaticGen.get('p', 'PV_30', 'v'))

        # test input type
        self.assertIsInstance(self.ss.DCOPF.get('pg', pd.Series(['PV_30']), 'v'), np.ndarray)

        # test return type
        self.assertIsInstance(self.ss.DCOPF.get('pg', 'PV_30', 'v'), float)
        self.assertIsInstance(self.ss.DCOPF.get('pg', ['PV_30'], 'v'), np.ndarray)

        # --- multi period routine ---
        self.ss.ED.run(solver='CLARABEL')
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

        self.ss.DCOPF.run(solver='CLARABEL')
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
        self.ss.DCOPF.run(solver='CLARABEL')
        obj = self.ss.DCOPF.obj.v

        # --- generator trip ---
        self.ss.StaticGen.set(src='u', attr='v', idx='PV_1', value=0)

        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCOPF.converged, "DCOPF did not converge under generator trip!")
        obj_gt = self.ss.DCOPF.obj.v
        self.assertGreater(obj_gt, obj)

        pg_trip = self.ss.DCOPF.get(src='pg', attr='v', idx='PV_1')
        np.testing.assert_almost_equal(pg_trip, 0, decimal=6)

        # --- trip line ---
        self.ss.Line.set(src='u', attr='v', idx='Line_4', value=0)

        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCOPF.converged, "DCOPF did not converge under line trip!")
        obj_lt = self.ss.DCOPF.obj.v
        self.assertGreater(obj_lt, obj_gt)

        plf_trip = self.ss.DCOPF.get(src='plf', attr='v', idx='Line_4')
        np.testing.assert_almost_equal(plf_trip, 0, decimal=6)


class TestSetOptzValueACOPF(unittest.TestCase):
    """
    Test value settings of `OptzBase` series in `ACOPF`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.xlsx"),
                           setup=True,
                           default_config=True,
                           no_output=True,
                           )

    def test_vset_before_init(self):
        """
        Test value setting before routine initialization.
        """
        # set values to `Var` before initialization is not doable
        v_ext = np.ones(self.ss.ACOPF.pg.n)
        self.ss.ACOPF.pg.v = v_ext
        self.assertIsNone(self.ss.ACOPF.pg.v)
        # set values to `Constraint` is not allowed
        self.assertRaises(AttributeError, lambda: setattr(self.ss.ACOPF.plfub, 'v', 1))
        # set values to `Objective` is not allowed
        self.assertRaises(AttributeError, lambda: setattr(self.ss.ACOPF.obj, 'v', 1))

    def test_vset_after_init(self):
        """
        Test value setting after routine initialization.
        """
        self.ss.ACOPF.init()
        # set values to `Var` after initialization is allowed
        v_ext = np.ones(self.ss.ACOPF.pg.n)
        self.ss.ACOPF.pg.v = v_ext
        np.testing.assert_equal(self.ss.ACOPF.pg.v, v_ext)
        # set values to `Constraint` is not allowed
        self.assertRaises(AttributeError, lambda: setattr(self.ss.ACOPF.plfub, 'v', 1))
        # set values to `Objective` is not allowed
        self.assertRaises(AttributeError, lambda: setattr(self.ss.ACOPF.obj, 'v', 1))


class TestSetOptzValueDCOPF(unittest.TestCase):
    """
    Test value settings of `OptzBase` series in `DCOPF`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.xlsx"),
                           setup=True,
                           default_config=True,
                           no_output=True,
                           )

    def test_vset_before_init(self):
        """
        Test value setting before routine initialization.
        """
        # set values to `Var` before initialization is not doable
        v_ext = np.ones(self.ss.DCOPF.pg.n)
        self.ss.DCOPF.pg.v = v_ext
        self.assertIsNone(self.ss.DCOPF.pg.v)
        # set values to `Constraint` is not allowed
        self.assertRaises(AttributeError, lambda: setattr(self.ss.DCOPF.plfub, 'v', 1))
        # set values to `Objective` is not allowed
        self.assertRaises(AttributeError, lambda: setattr(self.ss.DCOPF.obj, 'v', 1))

    def test_vset_after_init(self):
        """
        Test value setting after routine initialization.
        """
        self.ss.DCOPF.init()
        # set values to `Var` after initialization is allowed
        v_ext = np.ones(self.ss.DCOPF.pg.n)
        self.ss.DCOPF.pg.v = v_ext
        np.testing.assert_equal(self.ss.DCOPF.pg.v, v_ext)
        # set values to `Constraint` is not allowed
        self.assertRaises(AttributeError, lambda: setattr(self.ss.DCOPF.plfub, 'v', 1))
        # set values to `Objective` is not allowed
        self.assertRaises(AttributeError, lambda: setattr(self.ss.DCOPF.obj, 'v', 1))


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
        self.ss.RTED.run(solver='CLARABEL')

    def test_dc2ac(self):
        """
        Test `RTED.init()` method.
        """
        self.ss.RTED.dc2ac()
        self.assertTrue(self.ss.RTED.converted, "AC conversion failed!")
        self.assertTrue(self.ss.RTED.exec_time > 0, "Execution time is not greater than 0.")

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
