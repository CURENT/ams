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
