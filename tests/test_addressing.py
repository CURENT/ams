import unittest

import ams
import numpy as np

ams.config_logger(stream_level=40)


class TestAddressing(unittest.TestCase):
    """
    Tests for addressing.
    """

    def test_ieee14_address(self):
        """
        Test IEEE14 address.
        """

        # FIXME: why there will be case parsing information using ams.system.example()?
        ss = ams.system.example()

        # bus variable indices (internal)
        np.testing.assert_array_equal(ss.Bus.a.a,
                                      np.arange(0, ss.Bus.n, 1))
        np.testing.assert_array_equal(ss.Bus.v.a,
                                      np.arange(ss.Bus.n, 2*ss.Bus.n, 1))

        # external variable indices
        np.testing.assert_array_equal(ss.PV.ud.a,
                                      np.array([28, 29, 30, 31]))
        np.testing.assert_array_equal(ss.PV.p.a,
                                      np.array([32, 33, 34, 35]))
        np.testing.assert_array_equal(ss.PV.q.a,
                                      np.array([36, 37, 38, 39]))
        np.testing.assert_array_equal(ss.Slack.ud.a,
                                      np.array([40]))
        np.testing.assert_array_equal(ss.Slack.p.a,
                                      np.array([41]))
        np.testing.assert_array_equal(ss.Slack.q.a,
                                      np.array([42]))
