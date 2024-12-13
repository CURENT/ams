import unittest
import numpy as np

import ams


class TestModelMethods(unittest.TestCase):
    """
    Test methods of Model.
    """

    def test_model_set(self):
        """
        Test `Model.set()` method.
        """

        ss = ams.run(
            ams.get_case("ieee14/ieee14.json"),
            default_config=True,
            no_output=True,
        )

        # set a single value
        ss.PQ.set("p0", "PQ_1", "v", 0.25)
        self.assertEqual(ss.PQ.p0.v[0], 0.25)

        # set a list of values
        ss.PQ.set("p0", ["PQ_1", "PQ_2"], "v", [0.26, 0.51])
        np.testing.assert_equal(ss.PQ.p0.v[[0, 1]], [0.26, 0.51])

        # set a list of values
        ss.PQ.set("p0", ["PQ_3", "PQ_5"], "v", [0.52, 0.16])
        np.testing.assert_equal(ss.PQ.p0.v[[2, 4]], [0.52, 0.16])

        # set a list of idxes with a single element to an array of values
        ss.PQ.set("p0", ["PQ_4"], "v", np.array([0.086]))
        np.testing.assert_equal(ss.PQ.p0.v[3], 0.086)

        # set an array of idxes with a single element to an array of values
        ss.PQ.set("p0", np.array(["PQ_4"]), "v", np.array([0.096]))
        np.testing.assert_equal(ss.PQ.p0.v[3], 0.096)

        # set an array of idxes with a list of single value
        ss.PQ.set("p0", np.array(["PQ_4"]), "v", 0.097)
        np.testing.assert_equal(ss.PQ.p0.v[3], 0.097)

    def test_find_idx(self):
        ss = ams.load(ams.get_case('5bus/pjm5bus_demo.xlsx'))
        mdl = ss.REGCV1

        # not allow all matches
        self.assertListEqual(mdl.find_idx(keys='gammap', values=[0.25], allow_all=False),
                             ['VSG_1'])

        # allow all matches
        self.assertListEqual(mdl.find_idx(keys='gammap', values=[0.25], allow_all=True),
                             [['VSG_1', 'VSG_2', 'VSG_3', 'VSG_4']])

        # multiple values
        self.assertListEqual(mdl.find_idx(keys='name', values=['VSG_1', 'VSG_2'],
                                          allow_none=False, default=False),
                             ['VSG_1', 'VSG_2'])
        # non-existing value
        self.assertListEqual(mdl.find_idx(keys='name', values=['VSG_999'],
                                          allow_none=True, default=False),
                             [False])

        # non-existing value is not allowed
        with self.assertRaises(IndexError):
            mdl.find_idx(keys='name', values=['VSG_999'],
                         allow_none=False, default=False)

        # multiple keys
        self.assertListEqual(mdl.find_idx(keys=['gammap', 'name'],
                                          values=[[0.25, 0.25], ['VSG_1', 'VSG_2']]),
                             ['VSG_1', 'VSG_2'])

        # multiple keys, with non-existing values
        self.assertListEqual(mdl.find_idx(keys=['gammap', 'name'],
                                          values=[[0.25, 0.25], ['VSG_1', 'VSG_999']],
                                          allow_none=True, default='CURENT'),
                             ['VSG_1', 'CURENT'])

        # multiple keys, with non-existing values not allowed
        with self.assertRaises(IndexError):
            mdl.find_idx(keys=['gammap', 'name'],
                         values=[[0.25, 0.25], ['VSG_1', 'VSG_999']],
                         allow_none=False, default=999)

        # multiple keys, values are not iterable
        with self.assertRaises(ValueError):
            mdl.find_idx(keys=['gammap', 'name'],
                         values=[0.25, 0.25])

        # multiple keys, items length are inconsistent in values
        with self.assertRaises(ValueError):
            mdl.find_idx(keys=['gammap', 'name'],
                         values=[[0.25, 0.25], ['VSG_1']])

    def test_model_alter(self):
        """
        Test `Model.alter()` method.
        """

        ss = ams.load(
            ams.get_case('5bus/pjm5bus_demo.xlsx'),
            default_config=True,
            no_output=True,
        )
        ss.RTEDVIS.run(solver='CLARABEL')

        # alter `v`
        ss.REGCV1.alter(src='M', idx='VSG_2', value=1, attr='v')
        self.assertEqual(ss.REGCV1.M.v[1], 1 * ss.REGCV1.M.pu_coeff[1])

        # alter `vin`
        ss.REGCV1.alter(src='M', idx='VSG_2', value=1, attr='vin')
        self.assertEqual(ss.REGCV1.M.v[1], 1)

        # alter `vin` on instances without `vin` falls back to `v`
        ss.REGCV1.alter(src='bus', idx='VSG_2', value=1, attr='vin')
        self.assertEqual(ss.REGCV1.bus.v[1], 1)
