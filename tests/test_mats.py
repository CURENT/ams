import unittest
import numpy as np

from scipy.sparse import csr_matrix as c_sparse
from scipy.sparse import lil_matrix as l_sparse

import ams
from ams.core.matprocessor import MatProcessor, MParam


class TestMatProcessorBasic(unittest.TestCase):
    """
    Test basic functionality of MatProcessor.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("matpower/case300.m"),
                           default_config=True, no_output=True)
        self.nR = self.ss.Region.n
        self.nb = self.ss.Bus.n
        self.nl = self.ss.Line.n
        self.ng = self.ss.StaticGen.n
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

    def test_c(self):
        """
        Test connectivity matrices.
        """
        # Test `Cg`
        self.assertIsInstance(self.mats.Cg._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Cg.v, np.ndarray)
        self.assertEqual(self.mats.Cg._v.max(), 1)
        np.testing.assert_equal(self.mats.Cg._v.sum(axis=0), np.ones((1, self.ng)))

        # Test `Cl`
        self.assertIsInstance(self.mats.Cl._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Cl.v, np.ndarray)
        self.assertEqual(self.mats.Cl._v.max(), 1)
        np.testing.assert_equal(self.mats.Cl._v.sum(axis=0), np.ones((1, self.nD)))

        # Test `Csh`
        self.assertIsInstance(self.mats.Csh._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Csh.v, np.ndarray)
        self.assertEqual(self.mats.Csh._v.max(), 1)
        np.testing.assert_equal(self.mats.Csh._v.sum(axis=0), np.ones((1, self.nsh)))

        # Test `Cft`
        self.assertIsInstance(self.mats.Cft._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Cft.v, np.ndarray)
        self.assertEqual(self.mats.Cft._v.max(), 1)
        np.testing.assert_equal(self.mats.Cft._v.sum(axis=0), np.zeros((1, self.nl)))

    def test_b_calc(self):
        """
        Test `b_calc`.
        """
        b = self.mats._calc_b()
        self.assertIsInstance(b, np.ndarray)
        self.assertEqual(b.shape, (self.nl,))

    def test_bf(self):
        """
        Test `Bf`.
        """
        self.assertIsInstance(self.mats.Bf._v, (c_sparse, l_sparse))
        np.testing.assert_equal(self.mats.Bf._v.shape, (self.nl, self.nb))

    def test_bbus(self):
        """
        Test `Bbus`.
        """
        self.assertIsInstance(self.mats.Bbus._v, (c_sparse, l_sparse))
        np.testing.assert_equal(self.mats.Bbus._v.shape, (self.nb, self.nb))

    def test_pfinj(self):
        """
        Test `Pfinj`.
        """
        self.assertIsInstance(self.mats.Pfinj._v, np.ndarray)
        np.testing.assert_equal(self.mats.Pfinj._v.shape, (self.nl,))

    def test_pbusinj(self):
        """
        Test `Pbusinj`.
        """
        self.assertIsInstance(self.mats.Pbusinj._v, np.ndarray)
        np.testing.assert_equal(self.mats.Pbusinj._v.shape, (self.nb,))


class TestMatProcessorTDFs(unittest.TestCase):
    """
    Test PTDF, LODF, OTDF.
    """

    def setUp(self) -> None:
        self.cases = ['matpower/case14.m',
                      'matpower/case39.m',
                      'matpower/case118.m']

    def test_ptdf_before_mat_init(self):
        """
        Test `PTDF` before MatProcessor initialization.
        """

        for case in self.cases:
            ss = ams.load(ams.get_case(case),
                          setup=True, default_config=True, no_output=True)

            _ = ss.mats.build_ptdf(no_store=True)
            self.assertIsNone(ss.mats.PTDF._v)

            _ = ss.mats.build_ptdf(dtype='float64', no_store=False)
            self.assertEqual(ss.mats.PTDF._v.shape, (ss.Line.n, ss.Bus.n))
            self.assertEqual(ss.mats.PTDF._v.dtype, np.float64)

            ptdf = ss.mats.build_ptdf(dtype='float32', no_store=True)
            self.assertEqual(ptdf.dtype, np.float32)

    def test_lodf_before_ptdf(self):
        """
        Test `LODF` before `PTDF`.
        """

        for case in self.cases:
            ss = ams.load(ams.get_case(case),
                          setup=True, default_config=True, no_output=True)

            _ = ss.mats.build_lodf(no_store=True)
            self.assertIsNone(ss.mats.LODF._v)

            _ = ss.mats.build_lodf(dtype='float64', no_store=False)
            self.assertEqual(ss.mats.LODF._v.dtype, np.float64)

            lodf = ss.mats.build_lodf(dtype='float32', no_store=True)
            self.assertEqual(lodf.dtype, np.float32)

    def test_otdf_before_lodf(self):
        """
        Test `OTDF`.
        """

        for case in self.cases:
            ss = ams.load(ams.get_case(case),
                          setup=True, default_config=True, no_output=True)
            # build matrices
            ss.mats.build()

            otdf64 = ss.mats.build_otdf(dtype='float64')
            self.assertEqual(otdf64.dtype, np.float64)

            otdf32 = ss.mats.build_otdf(dtype='float32')
            self.assertEqual(otdf32.dtype, np.float32)

            np.testing.assert_allclose(otdf64, otdf32, atol=1e-3)

            # input str
            otdf_l2 = ss.mats.build_otdf(line=ss.Line.idx.v[2])
            self.assertEqual(otdf_l2.shape, (ss.Line.n, ss.Bus.n))

            # input list with single element
            otdf_l2 = ss.mats.build_otdf(line=ss.Line.idx.v[2:3])
            self.assertEqual(otdf_l2.shape, (ss.Line.n, ss.Bus.n))

            # input list with multiple elements
            otdf_l23 = ss.mats.build_otdf(line=ss.Line.idx.v[2:5])
            self.assertEqual(otdf_l23.shape, (ss.Line.n, ss.Bus.n))
