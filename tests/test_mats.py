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
        self.ss = ams.load(ams.get_case("matpower/case300.m"),
                           default_config=True, no_output=True)
        self.nR = self.ss.Region.n
        self.nB = self.ss.Bus.n
        self.nl = self.ss.Line.n
        self.nG = self.ss.StaticGen.n
        self.nsh = self.ss.Shunt.n
        self.nD = self.ss.StaticLoad.n

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

    def test_cg(self):
        """
        Test `Cg`.
        """
        self.assertIsInstance(self.mats.Cg._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Cg.v, np.ndarray)
        self.assertEqual(self.mats.Cg._v.max(), 1)
        np.testing.assert_equal(self.mats.Cg._v.sum(axis=0),
                                np.ones((1, self.nG)))

    def test_cl(self):
        """
        Test `Cl`.
        """
        self.assertIsInstance(self.mats.Cl._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Cl.v, np.ndarray)
        self.assertEqual(self.mats.Cl._v.max(), 1)
        np.testing.assert_equal(self.mats.Cl._v.sum(axis=0),
                                np.ones((1, self.nD)))

    def test_csh(self):
        """
        Test `Csh`.
        """
        self.assertIsInstance(self.mats.Csh._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Csh.v, np.ndarray)
        self.assertEqual(self.mats.Csh._v.max(), 1)
        np.testing.assert_equal(self.mats.Csh._v.sum(axis=0),
                                np.ones((1, self.nsh)))

    def test_cft(self):
        """
        Test `Cft`.
        """
        self.assertIsInstance(self.mats.Cft._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Cft.v, np.ndarray)
        self.assertEqual(self.mats.Cft._v.max(), 1)
        np.testing.assert_equal(self.mats.Cft._v.sum(axis=0),
                                np.zeros((1, self.nl)))
