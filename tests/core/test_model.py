import unittest
import numpy as np

import ams


class TestModelMethods(unittest.TestCase):
    """
    Test methods of Model.
    """

    def setUp(self):
        self.ss = ams.run(
            ams.get_case("ieee14/ieee14.json"),
            default_config=True,
            no_output=True,
        )

    def test_model_set(self):
        """
        Test `Model.set()` method.
        """

        # set a single value
        self.ss.PQ.set("p0", "PQ_1", "v", 0.25)
        self.assertEqual(self.ss.PQ.p0.v[0], 0.25)

        # set a list of values
        self.ss.PQ.set("p0", ["PQ_1", "PQ_2"], "v", [0.26, 0.51])
        np.testing.assert_equal(self.ss.PQ.p0.v[[0, 1]], [0.26, 0.51])

        # set a list of values
        self.ss.PQ.set("p0", ["PQ_3", "PQ_5"], "v", [0.52, 0.16])
        np.testing.assert_equal(self.ss.PQ.p0.v[[2, 4]], [0.52, 0.16])

        # set a list of idxes with a single element to an array of values
        self.ss.PQ.set("p0", ["PQ_4"], "v", np.array([0.086]))
        np.testing.assert_equal(self.ss.PQ.p0.v[3], 0.086)

        # set an array of idxes with a single element to an array of values
        self.ss.PQ.set("p0", np.array(["PQ_4"]), "v", np.array([0.096]))
        np.testing.assert_equal(self.ss.PQ.p0.v[3], 0.096)

        # set an array of idxes with a list of single value
        self.ss.PQ.set("p0", np.array(["PQ_4"]), "v", 0.097)
        np.testing.assert_equal(self.ss.PQ.p0.v[3], 0.097)

    def test_model_repr(self):
        """
        Test `Model.__repr__()` method.
        """

        for mdl in self.ss.models.items():
            print(mdl.__repr__())
