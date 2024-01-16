import unittest
import numpy as np

from scipy.sparse import csr_matrix as c_sparse
from scipy.sparse import lil_matrix as l_sparse

import ams
from ams.core.matprocessor import MatProcessor, MParam
from ams.core.service import NumOp, NumOpDual, ZonalSum


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
        self.mats.make()

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


class TestService(unittest.TestCase):
    """
    Test functionality of Services.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                           default_config=True,
                           no_output=True,)
        self.nR = self.ss.Region.n
        self.nB = self.ss.Bus.n
        self.nL = self.ss.Line.n
        self.nD = self.ss.StaticLoad.n  # number of static loads

    def test_NumOp_norfun(self):
        """
        Test `NumOp` without return function.
        """
        CftT = NumOp(u=self.ss.mats.Cft, fun=np.transpose)
        np.testing.assert_array_equal(CftT.v.transpose(), self.ss.mats.Cft.v)

    def test_NumOp_rfun(self):
        """
        Test `NumOp` with return function.
        """
        CftTT = NumOp(u=self.ss.mats.Cft, fun=np.transpose, rfun=np.transpose)
        np.testing.assert_array_equal(CftTT.v, self.ss.mats.Cft.v)

    def test_NumOp_ArrayOut(self):
        """
        Test `NumOp` non-array output.
        """
        M = NumOp(u=self.ss.PV.pmax,
                  fun=np.max,
                  rfun=np.dot, rargs=dict(b=10),
                  array_out=True,)
        M2 = NumOp(u=self.ss.PV.pmax,
                   fun=np.max,
                   rfun=np.dot, rargs=dict(b=10),
                   array_out=False,)
        self.assertIsInstance(M.v, np.ndarray)
        self.assertIsInstance(M2.v, (int, float))

    def test_NumOpDual(self):
        """
        Test `NumOpDual`.
        """
        p_vec = MParam(v=self.ss.DCOPF.pd.v)
        one_vec = MParam(v=np.ones(self.ss.StaticLoad.n))
        p_sum = NumOpDual(u=p_vec, u2=one_vec,
                          fun=np.multiply, rfun=np.sum)
        self.assertEqual(p_sum.v, self.ss.PQ.p0.v.sum())

    def test_ZonalSum(self):
        """
        Test `ZonalSum`.
        """
        ds = ZonalSum(u=self.ss.RTED.zd, zone="Region",
                      name="ds", tex_name=r"S_{d}",
                      info="Sum pl vector in shape of zone",)
        ds.rtn = self.ss.RTED
        # check if the shape is correct
        np.testing.assert_array_equal(ds.v.shape, (self.nR, self.nD))
        # check if the values are correct
        self.assertTrue(np.all(ds.v.sum(axis=1) <= np.array([self.nB, self.nB])))
