import unittest
import numpy as np

from scipy.sparse import csr_matrix as c_sparse
from scipy.sparse import lil_matrix as l_sparse

import ams
from ams.core.matprocessor import MatProcessor, MParam


class TestMatProcessor(unittest.TestCase):
    """
    Test functionality of MatProcessor.
    """

    def setUp(self) -> None:
        self.ss = ams.load(
            ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
            default_config=True,
            no_output=True,
        )
        self.nR = self.ss.Region.n
        self.nB = self.ss.Bus.n
        self.nl = self.ss.Line.n

        self.mats = MatProcessor(self.ss)
        self.mats.build()

    def test_MParam(self):
        """
        Test `MParam`.
        """
        one_vec = MParam(v=c_sparse(np.ones(self.ss.Bus.n)))
        # check if `_v` is `c_sparse` instance
        self.assertIsInstance(one_vec._v, c_sparse)
        # check if `v` is 1D-array
        self.assertEqual(one_vec.v.shape, (self.ss.Bus.n,))

    def test_Cft(self):
        """
        Test `Cft`.
        """
        self.assertIsInstance(self.mats.Cft._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Cft.v, np.ndarray)
        self.assertEqual(self.mats.Cft._v.max(), 1)
        np.testing.assert_equal(self.mats.Cft._v.sum(axis=0),
                                np.zeros((1, self.nl)))
