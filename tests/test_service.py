from functools import wraps
import unittest
import numpy as np

from scipy.sparse import csr_matrix as c_sparse  # NOQA
from scipy.sparse import lil_matrix as l_sparse  # NOQA

import ams
from ams.core.matprocessor import MatProcessor
from ams.core.service import NumOp, ZonalSum


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

    def test_PTDF(self):
        """
        Test `PTDF`.
        """
        # self.assertIsInstance(self.mats.PTDF._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.PTDF.v, np.ndarray)
        self.assertEqual(self.mats.PTDF.v.shape, (self.nl, self.nB))

    def test_Cft(self):
        """
        Test `Cft`.
        """
        self.assertIsInstance(self.mats.Cft._v, (c_sparse, l_sparse))
        self.assertIsInstance(self.mats.Cft.v, np.ndarray)
        self.assertEqual(self.mats.Cft.v.max(), 1)

    def test_pl(self):
        """
        Test `pl`.
        """
        self.assertEqual(self.mats.pl.v.ndim, 1)


class TestService(unittest.TestCase):
    """
    Test functionality of Services.
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
        M = NumOp(
            u=self.ss.PV.pmax,
            fun=np.max,
            rfun=np.dot,
            rargs=dict(b=10),
            array_out=True,
        )
        self.assertIsInstance(M.v, np.ndarray)
        M2 = NumOp(
            u=self.ss.PV.pmax,
            fun=np.max,
            rfun=np.dot,
            rargs=dict(b=10),
            array_out=False,
        )
        self.assertIsInstance(M2.v, (int, float))

    def test_NumOpDual(self):
        """
        Test `NumOpDual`.
        """
        pass

    def test_ZonalSum(self):
        """
        Test `ZonalSum`.
        """
        ds = ZonalSum(
            u=self.ss.RTED.zb,
            zone="Region",
            name="ds",
            tex_name=r"S_{d}",
            info="Sum pl vector in shape of zone",
        )
        ds.rtn = self.ss.RTED
        # check if the shape is correct
        np.testing.assert_array_equal(ds.v.shape, (self.nR, self.nB))
        # check if the values are correct
        self.assertTrue(np.all(ds.v.sum(axis=1) <= np.array([self.nB, self.nB])))
