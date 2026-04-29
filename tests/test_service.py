import unittest
import numpy as np
import scipy.sparse as sps

import ams
from ams.core.matprocessor import MParam
from ams.core.param import RParam
from ams.core.service import NumOp, NumOpDual, ZonalSum


def _as_dense(x):
    """Return a dense ndarray from either a sparse or dense input."""
    return x.toarray() if sps.issparse(x) else np.asarray(x)


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
        np.testing.assert_array_equal(_as_dense(CftT.v).transpose(),
                                      self.ss.mats.Cft.dense())

    def test_NumOp_rfun(self):
        """
        Test `NumOp` with return function.
        """
        CftTT = NumOp(u=self.ss.mats.Cft, fun=np.transpose, rfun=np.transpose)
        np.testing.assert_array_equal(_as_dense(CftTT.v),
                                      self.ss.mats.Cft.dense())

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

    def test_NumOpDual_sparse_passthrough(self):
        """
        ``NumOpDual.v0`` should pass scipy.sparse outputs through
        untouched when ``array_out=True``, mirroring the ``NumOp.v0``
        contract from PR #233.
        """
        sm = sps.csr_matrix(np.eye(3))
        u = MParam(v=sm)
        u2 = MParam(v=sm)
        op = NumOpDual(u=u, u2=u2,
                       fun=lambda a, b: a + b,
                       array_out=True)
        # Sparse-in, sparse-out: must not be wrapped in an
        # object-dtype ndarray as the pre-#233 NumOp bug used to do.
        self.assertTrue(sps.issparse(op.v0))


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


class TestRParam(unittest.TestCase):
    """
    Test functionality of ``RParam``.
    """

    def test_RParam_sparse_passthrough(self):
        """
        Externally-supplied sparse values should flow through
        ``RParam.v`` without being densified, completing the
        no-auto-densify contract started by PR #233.
        """
        sm = sps.csr_matrix([[1, 0], [0, 2]])
        rp = RParam(v=sm)
        self.assertTrue(sps.issparse(rp.v))
        np.testing.assert_array_equal(rp.v.toarray(), sm.toarray())
