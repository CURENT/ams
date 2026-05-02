"""
Test that ``System.setup()`` corrects model parameter values that
are zero/invalid before setup — covers the line rate_a/b/c and
amax/amin auto-correction paths.

Originally lived in ``tests/test_1st_system.py::TestParamCorrection``.
"""

import unittest

import ams


class TestParamCorrection(unittest.TestCase):
    """
    Test parameter correction.
    """

    def setUp(self) -> None:
        """
        Test setup.
        """
        self.ss = ams.load(ams.get_case('matpower/case14.m'),
                           setup=False, no_output=True, default_config=True,)

    def test_line_correction(self):
        """
        Test line correction.
        """
        self.ss.Line.rate_a.v[5] = 0.0
        self.ss.Line.rate_b.v[6] = 0.0
        self.ss.Line.rate_c.v[7] = 0.0
        self.ss.Line.amax.v[8] = 0.0
        self.ss.Line.amin.v[9] = 0.0

        self.ss.setup()

        self.assertIsNot(self.ss.Line.rate_a.v[5], 0.0)
        self.assertIsNot(self.ss.Line.rate_b.v[6], 0.0)
        self.assertIsNot(self.ss.Line.rate_c.v[7], 0.0)
        self.assertIsNot(self.ss.Line.amax.v[8], 0.0)
        self.assertIsNot(self.ss.Line.amin.v[9], 0.0)
