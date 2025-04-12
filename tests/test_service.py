import unittest
import numpy as np

import ams
from ams.core.matprocessor import MParam
from ams.core.service import NumOp, NumOpDual, ZonalSum


class TestService(unittest.TestCase):
    """
    Test functionality of Services.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("ieee39/ieee39_uced_esd1.xlsx"),
                           default_config=True,
                           no_output=True,)
        self.nR = self.ss.Zone.n
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
        ds = ZonalSum(u=self.ss.RTED.zd, zone="Area",
                      name="ds", tex_name=r"S_{d}",
                      info="Sum pl vector in shape of area",)
        ds.rtn = self.ss.RTED
        # check if the shape is correct
        np.testing.assert_array_equal(ds.v.shape, (self.nR, self.nD))
        # check if the values are correct
        self.assertTrue(np.all(ds.v.sum(axis=1) <= np.array([self.nB, self.nB])))
