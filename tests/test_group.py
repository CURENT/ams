"""
Tests for group functions.
"""
import unittest

import numpy as np

import ams


class TestGroup(unittest.TestCase):
    """
    Test the group class functions.
    """

    def setUp(self):
        self.ss = ams.run(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                          default_config=True,
                          no_output=True,
                          )

    def test_group_access(self):
        """
        Test methods such as `idx2model`
        """
        ss = self.ss

        # --- idx2uid ---
        self.assertIsNone(ss.DG.idx2uid(None))
        self.assertListEqual(ss.DG.idx2uid([None]), [None])

        # --- idx2model ---
        # what works
        self.assertIs(ss.DG.idx2model('ESD1_1'), ss.ESD1)
        self.assertListEqual(ss.DG.idx2model(['ESD1_1']), [ss.ESD1])

        # what does not work
        self.assertRaises(KeyError, ss.DG.idx2model, idx='1')
        self.assertRaises(KeyError, ss.DG.idx2model, idx=88)
        self.assertRaises(KeyError, ss.DG.idx2model, idx=[1, 88])

        # --- get ---
        self.assertRaises(KeyError, ss.DG.get, 'EtaC', 999)

        np.testing.assert_equal(ss.DG.get('EtaC', 'ESD1_1',), 1.0)

        np.testing.assert_equal(ss.DG.get('EtaC', ['ESD1_1'], allow_none=True,),
                                [1.0])
        np.testing.assert_equal(ss.DG.get('EtaC', ['ESD1_1', None],
                                          allow_none=True, default=0.95),
                                [1.0, 0.95])

        # --- set ---
        ss.DG.set('EtaC', 'ESD1_1', 'v', 0.95)
        np.testing.assert_equal(ss.DG.get('EtaC', 'ESD1_1',), 0.95)

        ss.DG.set('EtaC', ['ESD1_1'], 'v', 0.97)
        np.testing.assert_equal(ss.DG.get('EtaC', ['ESD1_1'],), [0.97])

        ss.DG.set('EtaC', ['ESD1_1'], 'v', [0.99])
        np.testing.assert_equal(ss.DG.get('EtaC', ['ESD1_1'],), [0.99])

        # --- find_idx ---
        self.assertListEqual(ss.DG.find_idx('name', ['ESD1_1']),
                             ss.ESD1.find_idx('name', ['ESD1_1']),
                             )

        self.assertListEqual(ss.DG.find_idx(['name', 'Sn'],
                                            [('ESD1_1',),
                                             (100.0,)]),
                             ss.ESD1.find_idx(['name', 'Sn'],
                                              [('ESD1_1',),
                                               (100.0,)]),)

        # --- get group idx ---
        self.assertListEqual(ss.DG.get_idx(), ss.ESD1.idx.v)

class TestGroupAdditional(unittest.TestCase):
    """
    Test additional group functions.
    """

    def setUp(self):
        self.ss = ams.load(
            ams.get_case('5bus/pjm5bus_demo.xlsx'),
            default_config=True,
            no_output=True,
        )

    def test_group_alter(self):
        """
        Test `Group.alter` method.
        """

        # alter `v`
        self.ss.VSG.alter(src='M', idx='VSG_2', value=1, attr='v')
        self.assertEqual(self.ss.REGCV1.M.v[1],
                         1 * self.ss.REGCV1.M.pu_coeff[1])

        # alter `vin`
        self.ss.VSG.alter(src='M', idx='VSG_2', value=1, attr='vin')
        self.assertEqual(self.ss.REGCV1.M.v[1], 1)

        # alter `vin` on instances without `vin` falls back to `v`
        self.ss.VSG.alter(src='bus', idx='VSG_2', value=1, attr='vin')
        self.assertEqual(self.ss.REGCV1.bus.v[1], 1)

    def test_as_dict(self):
        """
        Test `Group.as_dict()`.
        """
        self.assertIsInstance(self.ss.VSG.as_dict(), dict)
