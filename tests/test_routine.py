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

        # get an rparam value
        np.testing.assert_equal(self.ss.DCOPF.get('ug', 'PV_30'), 1)

        # get an unpacked var value
        self.ss.DCOPF.run(solver='ECOS')
        self.assertEqual(self.ss.DCOPF.exit_code, 0, "Exit code is not 0.")
        np.testing.assert_equal(self.ss.DCOPF.get('pg', 'PV_30', 'v'),
                                self.ss.StaticGen.get('p', 'PV_30', 'v'))

    def test_rouine_init(self):
        """
        Test `Routine.init()` method.
        """

        self.assertTrue(self.ss.DCOPF.init(), "DCOPF initialization failed!")


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
        self.assertTrue(self.ss.RTED.is_ac, "AC conversion failed!")

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
