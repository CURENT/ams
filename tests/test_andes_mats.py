"""
Test ANDES matrices.
"""

import unittest
import numpy as np
import importlib.metadata
from packaging.version import parse as parse_version

import ams


class TestMatrices(unittest.TestCase):
    """
    Tests for system matrices consistency.
    """

    andes_version = importlib.metadata.version("andes")
    if parse_version(andes_version) < parse_version('1.9.2'):
        raise unittest.SkipTest("Requires ANDES version >= 1.9.2")

    sp = ams.load(ams.get_case('5bus/pjm5bus_demo.xlsx'),
                  setup=True, no_output=True, default_config=True,)
    sa = sp.to_andes(setup=True, no_output=True, default_config=True,)

    def setUp(self) -> None:
        """
        Test setup.
        """

    def test_build_y(self):
        """
        Test build_y consistency.
        """
        ysp = self.sp.Line.build_y()
        ysa = self.sa.Line.build_y()
        np.testing.assert_equal(np.array(ysp.V), np.array(ysa.V))

    def test_build_Bp(self):
        """
        Test build_Bp consistency.
        """
        Bp_sp = self.sp.Line.build_Bp()
        Bp_sa = self.sa.Line.build_Bp()
        np.testing.assert_equal(np.array(Bp_sp.V), np.array(Bp_sa.V))

    def test_build_Bpp(self):
        """
        Test build_Bpp consistency.
        """
        Bpp_sp = self.sp.Line.build_Bpp()
        Bpp_sa = self.sa.Line.build_Bpp()
        np.testing.assert_equal(np.array(Bpp_sp.V), np.array(Bpp_sa.V))

    def test_build_Bdc(self):
        """
        Test build_Bdc consistency.
        """
        Bdc_sp = self.sp.Line.build_Bdc()
        Bdc_sa = self.sa.Line.build_Bdc()
        np.testing.assert_equal(np.array(Bdc_sp.V), np.array(Bdc_sa.V))
