import logging
import unittest

import json

import numpy as np

from andes.shared import rad2deg
import ams
from ams.shared import nan

logger = logging.getLogger(__name__)


class TestKnownResults(unittest.TestCase):

    def setUp(self) -> None:
        with open(ams.get_case('matpower/benchmark.json'), 'r') as file:
            self.mpres = json.load(file)

        self.sp = ams.load(ams.get_case('matpower/case14.m'),
                           setup=True, no_output=True, default_config=True)

    def test_DCPF_case14(self):
        """
        Test DC power flow for case14.
        """
        self.sp.DCPF.run()
        np.testing.assert_allclose(self.sp.DCPF.aBus.v * rad2deg,
                                   np.array(self.mpres['case14']['DCPF']['aBus']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)

        np.testing.assert_allclose(self.sp.DCPF.pg.v * self.sp.config.mva,
                                   np.array(self.mpres['case14']['DCPF']['pg']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)

    def test_PFlow_case14(self):
        """
        Test power flow for case14.
        """
        self.sp.PFlow.run()
        np.testing.assert_allclose(self.sp.PFlow.aBus.v * rad2deg,
                                   np.array(self.mpres['case14']['PFlow']['aBus']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(self.sp.PFlow.vBus.v,
                                   np.array(self.mpres['case14']['PFlow']['vBus']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(self.sp.PFlow.pg.v.sum() * self.sp.config.mva,
                                   np.array(self.mpres['case14']['PFlow']['pg']).sum(),
                                   rtol=1e-2, atol=1e-2)

    def test_DCOPF_case14(self):
        """
        Test DCOPF for case14.
        """
        self.sp.DCOPF.run(solver='CLARABEL')
        self.assertAlmostEqual(self.sp.DCOPF.obj.v,
                               self.mpres['case14']['DCOPF']['obj'],
                               places=4)
        np.testing.assert_allclose(self.sp.DCOPF.pi.v / self.sp.config.mva,
                                   np.array(self.mpres['case14']['DCOPF']['pi']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)

    def test_Matrices_case14(self):
        """
        Test matrices for case14.
        """
        ptdf = self.sp.mats.build_ptdf()
        lodf = self.sp.mats.build_lodf()

        ptdf_mp = load_ptdf(self.mpres, 'case14')
        lodf_mp = load_lodf(self.mpres, 'case14')

        ptdf[np.isnan(ptdf_mp)] = nan
        lodf[np.isnan(lodf_mp)] = nan

        np.testing.assert_allclose(ptdf, ptdf_mp,
                                   equal_nan=True, rtol=1e-2, atol=1e-2)

        np.testing.assert_allclose(lodf, lodf_mp,
                                   equal_nan=True, rtol=1e-2, atol=1e-2)


class TestKnownResultsIEEE39(unittest.TestCase):

    def setUp(self) -> None:
        with open(ams.get_case('matpower/benchmark.json'), 'r') as file:
            self.mpres = json.load(file)

        self.sp = ams.load(ams.get_case('matpower/case39.m'),
                           setup=True, no_output=True, default_config=True)

    def test_DCPF_case39(self):
        """
        Test DC power flow for case39.
        """
        self.sp.DCPF.run()
        np.testing.assert_allclose(self.sp.DCPF.aBus.v * rad2deg,
                                   np.array(self.mpres['case39']['DCPF']['aBus']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)

        np.testing.assert_allclose(self.sp.DCPF.pg.v.sum() * self.sp.config.mva,
                                   np.array(self.mpres['case39']['DCPF']['pg']).sum(),
                                   rtol=1e-2, atol=1e-2)

    def test_PFlow_case39(self):
        """
        Test power flow for case39.
        """
        self.sp.PFlow.run()
        np.testing.assert_allclose(self.sp.PFlow.aBus.v * rad2deg,
                                   np.array(self.mpres['case39']['PFlow']['aBus']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(self.sp.PFlow.vBus.v,
                                   np.array(self.mpres['case39']['PFlow']['vBus']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(self.sp.PFlow.pg.v.sum() * self.sp.config.mva,
                                   np.array(self.mpres['case39']['PFlow']['pg']).sum(),
                                   rtol=1e-2, atol=1e-2)

    def test_DCOPF_case39(self):
        """
        Test DCOPF for case39.
        """
        self.sp.DCOPF.run(solver='CLARABEL')
        self.assertAlmostEqual(self.sp.DCOPF.obj.v,
                               self.mpres['case39']['DCOPF']['obj'],
                               places=2)
        np.testing.assert_allclose(self.sp.DCOPF.pi.v / self.sp.config.mva,
                                   np.array(self.mpres['case39']['DCOPF']['pi']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)

    def test_Matrices_case39(self):
        """
        Test matrices for case39.
        """
        ptdf = self.sp.mats.build_ptdf()
        lodf = self.sp.mats.build_lodf()

        ptdf_mp = load_ptdf(self.mpres, 'case39')
        lodf_mp = load_lodf(self.mpres, 'case39')

        ptdf[np.isnan(ptdf_mp)] = nan
        lodf[np.isnan(lodf_mp)] = nan

        np.testing.assert_allclose(ptdf, ptdf_mp,
                                   equal_nan=True, rtol=1e-2, atol=1e-2)

        np.testing.assert_allclose(lodf, lodf_mp,
                                   equal_nan=True, rtol=1e-2, atol=10)


class TestKnownResultsIEEE118(unittest.TestCase):

    def setUp(self) -> None:
        with open(ams.get_case('matpower/benchmark.json'), 'r') as file:
            self.mpres = json.load(file)

        self.sp = ams.load(ams.get_case('matpower/case118.m'),
                           setup=True, no_output=True, default_config=True)

    def test_DCPF_case118(self):
        """
        Test DC power flow for case118.
        """
        self.sp.DCPF.run()
        aBus_mp = np.array(self.mpres['case118']['DCPF']['aBus']).reshape(-1)
        aBus_mp -= aBus_mp[0]
        np.testing.assert_allclose((self.sp.DCPF.aBus.v - self.sp.DCPF.aBus.v[0]) * rad2deg,
                                   aBus_mp,
                                   rtol=1e-2, atol=1e-2)

        np.testing.assert_allclose(self.sp.DCPF.pg.v.sum() * self.sp.config.mva,
                                   np.array(self.mpres['case118']['DCPF']['pg']).sum(),
                                   rtol=1e-2, atol=1e-2)

    def test_PFlow_case118(self):
        """
        Test power flow for case118.
        """
        self.sp.PFlow.run()
        np.testing.assert_allclose(self.sp.PFlow.aBus.v * rad2deg,
                                   np.array(self.mpres['case118']['PFlow']['aBus']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(self.sp.PFlow.vBus.v,
                                   np.array(self.mpres['case118']['PFlow']['vBus']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(self.sp.PFlow.pg.v.sum() * self.sp.config.mva,
                                   np.array(self.mpres['case118']['PFlow']['pg']).sum(),
                                   rtol=1e-2, atol=1e-2)

    def test_DCOPF_case118(self):
        """
        Test DCOPF for case118.
        """
        self.sp.DCOPF.run(solver='CLARABEL')
        self.assertAlmostEqual(self.sp.DCOPF.obj.v,
                               self.mpres['case118']['DCOPF']['obj'],
                               places=2)
        np.testing.assert_allclose(self.sp.DCOPF.pi.v / self.sp.config.mva,
                                   np.array(self.mpres['case118']['DCOPF']['pi']).reshape(-1),
                                   rtol=1e-2, atol=1e-2)

    def test_Matrices_case118(self):
        """
        Test matrices for case118.
        """
        ptdf = self.sp.mats.build_ptdf()
        lodf = self.sp.mats.build_lodf()

        ptdf_mp = load_ptdf(self.mpres, 'case118')
        lodf_mp = load_lodf(self.mpres, 'case118')

        ptdf[np.isnan(ptdf_mp)] = nan
        lodf[np.isnan(lodf_mp)] = nan

        np.testing.assert_allclose(ptdf, ptdf_mp,
                                   equal_nan=True, rtol=1e-2, atol=1e-2)

        np.testing.assert_allclose(lodf, lodf_mp,
                                   equal_nan=True, rtol=1e-2, atol=10)


def load_ptdf(mpres, case):
    """
    Load PTDF from mpres.

    Parameters
    ----------
    mpres : dict
        The result dictionary.
    case : str
        The case name.

    Returns
    -------
    ptdf : np.ndarray
        The PTDF matrix.
    """
    ptdf_data = np.array(mpres[case]['PTDF'])
    ptdf = np.array([[0 if val == "_NaN_" else val for val in row] for row in ptdf_data],
                    dtype=float)
    return ptdf


def load_lodf(mpres, case):
    """
    Load LODF from mpres.

    Parameters
    ----------
    mpres : dict
        The result dictionary.
    case : str
        The case name.

    Returns
    -------
    lodf : np.ndarray
        The LODF matrix.
    """
    lodf_data = np.array(mpres[case]['LODF'])
    lodf = np.array([[nan if val in ["_NaN_", "-_Inf_", "_Inf_"] else val for val in row] for row in lodf_data],
                    dtype=float)
    # NOTE: force the diagonal to be -1
    np.fill_diagonal(lodf, -1)
    return lodf
