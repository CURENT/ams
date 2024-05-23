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
        # cases is for testing PTDF, LODF, etc.
        self.cases = ['matpower/case14.m',
                      'matpower/case39.m',
                      'matpower/case118.m']
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

    def test_ptdf(self):
        """
        Test `PTDF`.
        """

        for case in self.cases:
            ss = ams.load(ams.get_case(case),
                          setup=True, default_config=True, no_output=True)
            ss.DCOPF.run(solver='ECOS')

            ptdf = ss.mats.build_ptdf()

            plf = ss.DCOPF.plf.v
            plfc = ptdf@(ss.mats.Cg._v@ss.DCOPF.pg.v - ss.mats.Cl._v@ss.DCOPF.pd.v)
            np.testing.assert_allclose(plf, plfc, atol=1e-2)

    def test_lodf(self):
        """
        Test `LODF`.
        """
        for case in self.cases:
            ss = ams.load(ams.get_case(case),
                          setup=True, default_config=True, no_output=True)
            # build matrices
            ss.mats.build()
            _ = ss.mats.build_ptdf()
            lodf = ss.mats.build_lodf()

            # outage line
            oline_idx = ss.Line.idx.v[1]
            oline = ss.Line.idx2uid(oline_idx)

            # pre-outage
            ss.DCPF.run()
            plf0 = ss.DCPF.plf.v.copy()

            # post-outage
            ss.Line.set(src='u', attr='v', idx=oline_idx, value=0)
            ss.DCPF.update()
            ss.DCPF.run()
            plf1 = ss.DCPF.plf.v.copy()

            dplf = plf1 - plf0
            dplfc = -lodf[:, oline] * dplf[oline]

            np.testing.assert_allclose(dplf, dplfc, atol=1e-7)

    def test_otdf(self):
        """
        Test `OTDF`.
        """

        for case in self.cases:
            ss = ams.load(ams.get_case(case),
                          setup=True, default_config=True, no_output=True)
            # build matrices
            ss.mats.build()

            otdf64 = ss.mats.build_otdf(dtype='float64')
            otdf32 = ss.mats.build_otdf(dtype='float32')

            np.testing.assert_allclose(otdf64, otdf32, atol=1e-3)

    def test_tdf_float32(self):
        """
        Test TDFs with float32 is runnable.
        """

        for case in self.cases:
            ss = ams.load(ams.get_case(case),
                          setup=True, default_config=True, no_output=True)
            # build matrices
            ss.mats.build()

            ss.mats.build_ptdf(dtype='float32')
            ss.mats.build_lodf(dtype='float32')
            ss.mats.build_otdf(dtype='float32')
