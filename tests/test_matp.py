"""
Test module MatProcessor.
"""

import unittest
import os

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
        self.nR = self.ss.Zone.n
        self.nb = self.ss.Bus.n
        self.nl = self.ss.Line.n
        self.ng = self.ss.StaticGen.n
        self.nsh = self.ss.Shunt.n
        self.nD = self.ss.StaticLoad.n

        self.mats = MatProcessor(self.ss)
        self.mats.build()

    def test_MParams_owner(self):
        """
        Tesst MParams owner before system initialization.
        """
        self.assertIs(self.mats.Cft.owner, self.mats)
        self.assertIs(self.mats.CftT.owner, self.mats)
        self.assertIs(self.mats.Cg.owner, self.mats)
        self.assertIs(self.mats.Cl.owner, self.mats)
        self.assertIs(self.mats.Csh.owner, self.mats)
        self.assertIs(self.mats.Bbus.owner, self.mats)
        self.assertIs(self.mats.Bf.owner, self.mats)
        self.assertIs(self.mats.Pbusinj.owner, self.mats)
        self.assertIs(self.mats.Pfinj.owner, self.mats)
        self.assertIs(self.mats.PTDF.owner, self.mats)
        self.assertIs(self.mats.LODF.owner, self.mats)

    def test_MParam_export_csv(self):
        """
        Test MParams export.
        """
        # --- path is not given ---
        exported_csv = self.mats.Cft.export_csv()
        self.assertTrue(os.path.exists(exported_csv))
        os.remove(exported_csv)

        # --- path is given ---
        path = 'CASE300_Cft.csv'
        exported_csv = self.mats.Cft.export_csv(path)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_MParams_export(self) -> None:
        """
        Test MParams export.
        """
        cft = self.mats.Cft.export_csv()
        self.assertTrue(os.path.exists(cft))
        os.remove(cft)

        cftt = self.mats.CftT.export_csv()
        self.assertTrue(os.path.exists(cftt))
        os.remove(cftt)

        cg = self.mats.Cg.export_csv()
        self.assertTrue(os.path.exists(cg))
        os.remove(cg)

        cl = self.mats.Cl.export_csv()
        self.assertTrue(os.path.exists(cl))
        os.remove(cl)

        csh = self.mats.Csh.export_csv()
        self.assertTrue(os.path.exists(csh))
        os.remove(csh)

        bbus = self.mats.Bbus.export_csv()
        self.assertTrue(os.path.exists(bbus))
        os.remove(bbus)

        bf = self.mats.Bf.export_csv()
        self.assertTrue(os.path.exists(bf))
        os.remove(bf)

        ptdf = self.mats.PTDF.export_csv()
        self.assertTrue(os.path.exists(ptdf))
        os.remove(ptdf)

        lodf = self.mats.LODF.export_csv()
        self.assertTrue(os.path.exists(lodf))
        os.remove(lodf)

    def test_MParam_instance(self):
        """
        Test `MParam` instantiate.
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


class TestBuildPTDF(unittest.TestCase):
    """
    Test build PTDF.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case('matpower/case14.m'),
                           setup=True, default_config=True, no_output=True)
        self.nl = self.ss.Line.n
        self.nb = self.ss.Bus.n

        self.assertFalse(self.ss.mats.initialized)
        # NOTE: here we test `no_store=True` option
        self.ptdf_full = self.ss.mats.build_ptdf(no_store=True)
        self.assertTrue(self.ss.mats.initialized)

    def test_ptdf_before_mat_init(self):
        """
        Test PTDF before MatProcessor initialization `mats.init()`.
        """
        self.assertIsNone(self.ss.mats.PTDF._v)
        _ = self.ss.mats.build_ptdf(no_store=False)
        self.assertEqual(self.ss.mats.PTDF._v.shape, (self.nl, self.nb))

    def test_ptdf_lines(self):
        """
        Test PTDF with `line` inputs.
        """
        # NOTE: when `line` is given, `PTDF` should not be stored even with `no_store=False`

        # input str
        ptdf_l2 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2], no_store=False)
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_l2, self.ptdf_full[2, :])

        # input list with single element
        ptdf_l2p = self.ss.mats.build_ptdf(line=[self.ss.Line.idx.v[2]], no_store=False)
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_l2p, self.ptdf_full[[2], :])

        # input list with multiple elements
        ptdf_l23 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], no_store=False)
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_l23, self.ptdf_full[2:4, :])

    def test_ptdf_incremental_lines(self):
        """
        Test PTDF incremental build with `line` inputs.
        """
        # input str
        ptdf_l2 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2], incremental=True, no_store=False,
                                          no_tqdm=True)
        np.testing.assert_array_almost_equal(ptdf_l2.todense(),
                                             self.ptdf_full[[2], :])
        self.assertTrue(sps.isspmatrix_lil(ptdf_l2))
        self.assertIsNone(self.ss.mats.PTDF._v)

        # input list with single element
        ptdf_l2p = self.ss.mats.build_ptdf(line=[self.ss.Line.idx.v[2]], incremental=True, no_store=False,
                                           no_tqdm=False)
        np.testing.assert_array_almost_equal(ptdf_l2p.todense(),
                                             self.ptdf_full[[2], :])
        self.assertTrue(sps.isspmatrix_lil(ptdf_l2p))
        self.assertIsNone(self.ss.mats.PTDF._v)

        # input list with multiple elements
        ptdf_l23 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], incremental=True, no_store=False)
        np.testing.assert_array_almost_equal(ptdf_l23.todense(),
                                             self.ptdf_full[2:4, :])

    def test_ptdf_incremental_step(self):
        """
        Test PTDF incremental build with step.
        """
        # step < line length
        ptdf_c1 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                          no_store=False, step=1)
        self.assertTrue(sps.isspmatrix_lil(ptdf_c1))
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_c1.todense(),
                                             self.ptdf_full[2:4, :],)

        # step = line length
        ptdf_c2 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                          no_store=False, step=2)
        self.assertTrue(sps.isspmatrix_lil(ptdf_c2))
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_c2.todense(),
                                             self.ptdf_full[2:4, :],)

        # step > line length
        ptdf_c5 = self.ss.mats.build_ptdf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                          no_store=False, step=5)
        self.assertTrue(sps.isspmatrix_lil(ptdf_c5))
        self.assertIsNone(self.ss.mats.PTDF._v)
        np.testing.assert_array_almost_equal(ptdf_c5.todense(),
                                             self.ptdf_full[2:4, :],)


class TestBuildLODF(unittest.TestCase):
    """
    Test build LODF.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case('matpower/case14.m'),
                           setup=True, default_config=True, no_output=True)
        self.nl = self.ss.Line.n
        self.nb = self.ss.Bus.n
        self.dec = 4

        self.assertFalse(self.ss.mats.initialized)
        self.lodf_full = self.ss.mats.build_lodf(no_store=True)
        self.assertTrue(self.ss.mats.initialized)

    def test_lodf_before_ptdf(self):
        """
        Test LODF before PTDF.
        """
        self.assertIsNone(self.ss.mats.PTDF._v)
        self.assertIsNone(self.ss.mats.LODF._v)
        _ = self.ss.mats.build_lodf(no_store=False)
        self.assertEqual(self.ss.mats.LODF._v.shape, (self.nl, self.nl))
        # PTDF should not be stored for requested building
        self.assertIsNone(self.ss.mats.PTDF._v)

    def test_lodf_lines(self):
        """
        Test LODF with `line` inputs.
        """
        # input str
        lodf_l2 = self.ss.mats.build_lodf(line=self.ss.Line.idx.v[2], no_store=False)
        np.testing.assert_array_almost_equal(lodf_l2, self.lodf_full[:, 2], decimal=self.dec)

        # input list with single element
        lodf_l2p = self.ss.mats.build_lodf(line=[self.ss.Line.idx.v[2]], no_store=False)
        np.testing.assert_array_almost_equal(lodf_l2p, self.lodf_full[:, [2]], decimal=self.dec)

        # input list with multiple elements
        lodf_l23 = self.ss.mats.build_lodf(line=self.ss.Line.idx.v[2:4], no_store=False)
        np.testing.assert_array_almost_equal(lodf_l23, self.lodf_full[:, 2:4], decimal=self.dec)

    def test_lodf_incremental_lines(self):
        """
        Test LODF incremental build with `line` inputs.
        """
        # input str
        lodf_l2 = self.ss.mats.build_lodf(line=self.ss.Line.idx.v[2], incremental=True, no_store=False)
        np.testing.assert_array_almost_equal(lodf_l2.todense(),
                                             self.lodf_full[:, [2]],
                                             decimal=self.dec)
        self.assertTrue(sps.isspmatrix_lil(lodf_l2))

        # input list with single element
        lodf_l2p = self.ss.mats.build_lodf(line=[self.ss.Line.idx.v[2]], incremental=True, no_store=False)
        np.testing.assert_array_almost_equal(lodf_l2p.todense(),
                                             self.lodf_full[:, [2]],
                                             decimal=self.dec)
        self.assertTrue(sps.isspmatrix_lil(lodf_l2p))

        # input list with multiple elements
        lodf_l23 = self.ss.mats.build_lodf(line=self.ss.Line.idx.v[2:4], incremental=True, no_store=False)
        np.testing.assert_array_almost_equal(lodf_l23.todense(),
                                             self.lodf_full[:, 2:4],
                                             decimal=self.dec)

    def test_lodf_incremental_step(self):
        """
        Test LODF incremental build with step.
        """
        # step < line length
        lodf_c1 = self.ss.mats.build_lodf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                          no_store=False, step=1)
        self.assertTrue(sps.isspmatrix_lil(lodf_c1))
        np.testing.assert_array_almost_equal(lodf_c1.todense(),
                                             self.lodf_full[:, 2:4],
                                             decimal=self.dec)

        # step = line length
        lodf_c2 = self.ss.mats.build_lodf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                          no_store=False, step=2)
        self.assertTrue(sps.isspmatrix_lil(lodf_c2))
        np.testing.assert_array_almost_equal(lodf_c2.todense(),
                                             self.lodf_full[:, 2:4],
                                             decimal=self.dec)

        # step > line length
        lodf_c5 = self.ss.mats.build_lodf(line=self.ss.Line.idx.v[2:4], incremental=True,
                                          no_store=False, step=5)
        self.assertTrue(sps.isspmatrix_lil(lodf_c5))
        np.testing.assert_array_almost_equal(lodf_c5.todense(),
                                             self.lodf_full[:, 2:4],
                                             decimal=self.dec)


class TestBuildOTDF(unittest.TestCase):
    """
    Test build OTDF.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case('matpower/case14.m'),
                           setup=True, default_config=True, no_output=True)
        self.nl = self.ss.Line.n
        self.nb = self.ss.Bus.n
        self.dec = 4
        _ = self.ss.mats.build()

    def test_otdf_dense_build(self):
        _ = self.ss.mats.build_ptdf(no_store=False, incremental=False)
        _ = self.ss.mats.build_lodf(no_store=False, incremental=False)
        self.otdf_full = self.ss.mats.build_otdf()
        self.assertEqual(self.otdf_full.shape, (self.nl, self.nb))

    def test_otdf_sparse_build(self):
        # --- both PTDF and LODF are dense ---
        _ = self.ss.mats.build_ptdf(no_store=False, incremental=False)
        _ = self.ss.mats.build_lodf(no_store=False, incremental=False)
        otdf_dense = self.ss.mats.build_otdf()

        _ = self.ss.mats.build_ptdf(no_store=False, incremental=True)
        _ = self.ss.mats.build_lodf(no_store=False, incremental=True)
        otdf_sparse = self.ss.mats.build_otdf()
        np.testing.assert_array_almost_equal(otdf_sparse.todense(), otdf_dense,
                                             decimal=self.dec)

        # --- PTDF is dense and LODF is sparse ---
        _ = self.ss.mats.build_ptdf(no_store=False, incremental=False)
        _ = self.ss.mats.build_lodf(no_store=False, incremental=True)
        otdf_sps1 = self.ss.mats.build_otdf()
        np.testing.assert_array_almost_equal(otdf_sps1, otdf_dense,
                                             decimal=self.dec)

        # --- PTDF is sparse and LODF is dense ---
        _ = self.ss.mats.build_ptdf(no_store=False, incremental=True)
        _ = self.ss.mats.build_lodf(no_store=False, incremental=False)
        otdf_sps2 = self.ss.mats.build_otdf()
        np.testing.assert_array_almost_equal(otdf_sps2.todense(), otdf_dense,
                                             decimal=self.dec)

    def test_otdf_lines(self):
        """
        Test OTDF with `line` inputs.
        """
        # --- both PTDF and LODF are dense ---
        _ = self.ss.mats.build_ptdf(no_store=False, incremental=False)
        _ = self.ss.mats.build_lodf(no_store=False, incremental=False)

        # input str
        otdf_l2 = self.ss.mats.build_otdf(line=self.ss.Line.idx.v[2])
        self.assertEqual(otdf_l2.shape, (self.nl, self.nb))

        # input list with single element
        otdf_l2p = self.ss.mats.build_otdf(line=[self.ss.Line.idx.v[2]])
        self.assertEqual(otdf_l2p.shape, (self.nl, self.nb))

        # input list with multiple elements
        otdf_l23 = self.ss.mats.build_otdf(line=self.ss.Line.idx.v[2:4])
        self.assertEqual(otdf_l23.shape, (self.nl, self.nb))
