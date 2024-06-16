import unittest
import numpy as np

import ams
from ams.core.matprocessor import MatProcessor, MParam
from ams.shared import sps


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
        one_vec = MParam(v=sps.csr_matrix(np.ones(self.ss.Bus.n)))
        # check if `_v` is `sps.csr_matrix` instance
        self.assertIsInstance(one_vec._v, sps.csr_matrix)
        # check if `v` is 1D-array
        self.assertEqual(one_vec.v.shape, (self.ss.Bus.n,))

    def test_c(self):
        """
        Test connectivity matrices.
        """
        # Test `Cg`
        self.assertIsInstance(self.mats.Cg._v, (sps.csr_matrix, sps.lil_matrix))
        self.assertIsInstance(self.mats.Cg.v, np.ndarray)
        self.assertEqual(self.mats.Cg._v.max(), 1)
        np.testing.assert_equal(self.mats.Cg._v.sum(axis=0), np.ones((1, self.ng)))

        # Test `Cl`
        self.assertIsInstance(self.mats.Cl._v, (sps.csr_matrix, sps.lil_matrix))
        self.assertIsInstance(self.mats.Cl.v, np.ndarray)
        self.assertEqual(self.mats.Cl._v.max(), 1)
        np.testing.assert_equal(self.mats.Cl._v.sum(axis=0), np.ones((1, self.nD)))

        # Test `Csh`
        self.assertIsInstance(self.mats.Csh._v, (sps.csr_matrix, sps.lil_matrix))
        self.assertIsInstance(self.mats.Csh.v, np.ndarray)
        self.assertEqual(self.mats.Csh._v.max(), 1)
        np.testing.assert_equal(self.mats.Csh._v.sum(axis=0), np.ones((1, self.nsh)))

        # Test `Cft`
        self.assertIsInstance(self.mats.Cft._v, (sps.csr_matrix, sps.lil_matrix))
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
        self.assertIsInstance(self.mats.Bf._v, (sps.csr_matrix, sps.lil_matrix))
        np.testing.assert_equal(self.mats.Bf._v.shape, (self.nl, self.nb))

    def test_bbus(self):
        """
        Test `Bbus`.
        """
        self.assertIsInstance(self.mats.Bbus._v, (sps.csr_matrix, sps.lil_matrix))
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


# class TestMatProcessorTDFs(unittest.TestCase):
#     """
#     Test PTDF, LODF, OTDF.
#     """

#     def setUp(self) -> None:
#         self.cases = ['matpower/case14.m',
#                       'matpower/case39.m',
#                       'matpower/case118.m']

#     def test_ptdf_before_mat_init(self):
#         """
#         Test `build_PTDF()` before MatProcessor initialization `build()`.
#         """

#         for case in self.cases:
#             ss = ams.load(ams.get_case(case),
#                           setup=True, default_config=True, no_output=True)

#             # --- test no_store ---
#             ptdf = ss.mats.build_ptdf(no_store=True)
#             self.assertIsNone(ss.mats.PTDF._v)
#             self.assertEqual(ss.mats.PTDF._v.shape, (ss.Line.n, ss.Bus.n))

#     def test_ptdf_line(self):
#         """
#         Test `build_PTDF()` with line.
#         """

#         for case in self.cases:
#             ss = ams.load(ams.get_case(case),
#                           setup=True, default_config=True, no_output=True)

#             flag_build = ss.mats.build()
#             self.assertTrue(flag_build)

#             # TODO: test different line inputs

#     def test_lodf_before_ptdf(self):
#         """
#         Test `LODF` before `PTDF`.
#         """

#         for case in self.cases:
#             ss = ams.load(ams.get_case(case),
#                           setup=True, default_config=True, no_output=True)

#             _ = ss.mats.build_lodf(no_store=True)
#             self.assertIsNone(ss.mats.LODF._v)

#             _ = ss.mats.build_lodf(no_store=False)
#             self.assertEqual(ss.mats.LODF._v.shape, (ss.Line.n, ss.Line.n))

#     def test_otdf_before_lodf(self):
#         """
#         Test `OTDF`.
#         """

#         for case in self.cases:
#             ss = ams.load(ams.get_case(case),
#                           setup=True, default_config=True, no_output=True)
#             # build matrices
#             ss.mats.build()

#             otdf64 = ss.mats.build_otdf()
#             self.assertEqual(otdf64.shape, (ss.Line.n, ss.Bus.n))

#             # input str
#             otdf_l2 = ss.mats.build_otdf(line=ss.Line.idx.v[2])
#             self.assertEqual(otdf_l2.shape, (ss.Line.n, ss.Bus.n))

#             # input list with single element
#             otdf_l2 = ss.mats.build_otdf(line=ss.Line.idx.v[2:3])
#             self.assertEqual(otdf_l2.shape, (ss.Line.n, ss.Bus.n))

#             # input list with multiple elements
#             otdf_l23 = ss.mats.build_otdf(line=ss.Line.idx.v[2:5])
#             self.assertEqual(otdf_l23.shape, (ss.Line.n, ss.Bus.n))


class TestBuildPTDF(unittest.TestCase):
    """
    Test build PTDF.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case('matpower/case14.m'),
                           setup=True, default_config=True, no_output=True)
        self.nl = self.ss.Line.n
        self.nb = self.ss.Bus.n
        self.dec = 4

        self.assertFalse(self.ss.mats.initialized)
        # NOTE: here we test `no_store=True` option
        self.ptdf_full = self.ss.mats.build_ptdf(no_store=True)
        self.assertTrue(self.ss.mats.initialized)

    def test_ptdf_before_mat_init(self):
        """
        Test PTDF before MatProcessor initialization `mats.init()`.
        """
        self.assertIsNone(self.ss.mats.PTDF._v)
        self.assertEqual(self.ptdf_full.shape, (self.nl, self.nb))

    def test_ptdf_lines(self):
        """
        Test PTDF with `line` inputs.
        """
        # NOTE: when `line` is given, `PTDF` should not be stored even with `no_store=False`

        # input str
        ptdf_l2 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2], no_store=False)
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_l2, self.ptdf_full[2, :], decimal=self.dec)

        # input list with single element
        ptdf_l2p = self.ss.mats.build_ptdf(line=[self.ss.Line.idx.v[2]], no_store=False)
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_l2p, self.ptdf_full[[2], :], decimal=self.dec)

        # input list with multiple elements
        ptdf_l23 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], no_store=False)
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_l23, self.ptdf_full[2:4, :], decimal=self.dec)

    def test_ptdf_incremental_lines(self):
        """
        Test PTDF incremental build with `line` inputs.
        """
        # input str
        ptdf_l2 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2], incremental=True, no_store=False,
                                          no_tqdm=True)
        np.testing.assert_array_almost_equal(ptdf_l2.todense(),
                                             self.ptdf_full[[2], :],
                                             decimal=self.dec)
        self.assertTrue(sps.isspmatrix_lil(ptdf_l2))
        self.assertIsNone(self.ss.mats.PTDF._v)

        # input list with single element
        ptdf_l2p = self.ss.mats.build_ptdf(line=[self.ss.Line.idx.v[2]], incremental=True, no_store=False,
                                           no_tqdm=False)
        np.testing.assert_array_almost_equal(ptdf_l2p.todense(),
                                             self.ptdf_full[[2], :],
                                             decimal=self.dec)
        self.assertTrue(sps.isspmatrix_lil(ptdf_l2p))
        self.assertIsNone(self.ss.mats.PTDF._v)

        # input list with multiple elements
        ptdf_l23 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], incremental=True, no_store=False)
        np.testing.assert_array_almost_equal(ptdf_l23.todense(),
                                             self.ptdf_full[2:4, :],
                                             decimal=self.dec)

        # test decimals
        ptdf_l23_dec2 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                                no_store=False, decimals=2)
        self.assertTrue(sps.isspmatrix_lil(ptdf_l23_dec2))
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_l23_dec2.todense(),
                                             self.ptdf_full[2:4, :],
                                             decimal=2)

    def test_ptdf_incrementa_chunk_size(self):
        """
        Test PTDF incremental build with chunk size.
        """
        # chunk_size < line length
        ptdf_c1 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                          no_store=False, chunk_size=1)
        self.assertTrue(sps.isspmatrix_lil(ptdf_c1))
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_c1.todense(),
                                             self.ptdf_full[2:4, :],
                                             decimal=self.dec)

        # chunk_size = line length
        ptdf_c2 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                          no_store=False, chunk_size=2)
        self.assertTrue(sps.isspmatrix_lil(ptdf_c2))
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_c2.todense(),
                                             self.ptdf_full[2:4, :],
                                             decimal=self.dec)

        # chunk_size > line length
        ptdf_c5 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                          no_store=False, chunk_size=5)
        self.assertTrue(sps.isspmatrix_lil(ptdf_c5))
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_c5.todense(),
                                             self.ptdf_full[2:4, :],
                                             decimal=self.dec)
