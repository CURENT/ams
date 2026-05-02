import unittest

import numpy as np

import ams
from ams.shared import skip_unittest_without_MISOCP


class TestRTEDES(unittest.TestCase):
    """
    Test routine `RTEDES`.

    Common scenarios (init, trip_gen, trip_line, set_load, vBus, dc2ac)
    are exercised in `tests/routines/test_scenarios_lp_singleperiod.py`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])

    @skip_unittest_without_MISOCP
    def test_ch_decision(self):
        """
        Test charging/discharging decision for charging/discharging duration time.

        Scenarios to validate:
                    res     ucd     tdc     tdc0
        0           True    1       1.0     0.5
        1           False   0       1.0     0.5
        2           True    1       1.0     0.0
        3           True    0       1.0     0.0
        4           True    1       0.5     1.0
        5           True    0       0.5     1.0
        """
        self.ss.RTEDES.init()

        self.ss.ESD1.set(src='tdc0', attr='v', idx='ESD1_1', value=0.5)
        self.ss.ESD1.set(src='tdc', attr='v', idx='ESD1_1', value=1.0)
        # scenario 1: initially charging, charging duration time not met, decision to charge
        self.ss.RTEDES.ucd.optz.value = np.array([1])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 1!")
        # scenario 2: initially charging, with charging duration time not met, decision not to charge
        self.ss.RTEDES.ucd.optz.value = np.array([0])
        self.assertFalse(self.ss.RTEDES.tcdr.e[0] <= 0,
                         "RTEDES.tcdr should be broken in scenario 2!")

        self.ss.ESD1.set(src='tdc0', attr='v', idx='ESD1_1', value=0.0)
        self.ss.ESD1.set(src='tdc', attr='v', idx='ESD1_1', value=1.0)
        # scenario 3: initially not charging, decision to charge
        self.ss.RTEDES.ucd.optz.value = np.array([1])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 3!")
        # scenario 4: initially not charging, decision not to charge
        self.ss.RTEDES.ucd.optz.value = np.array([0])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 4!")

        self.ss.ESD1.set(src='tdc0', attr='v', idx='ESD1_1', value=1.0)
        self.ss.ESD1.set(src='tdc', attr='v', idx='ESD1_1', value=0.5)
        # scenario 5: initially charging, charging duration time met, decision to charge
        self.ss.RTEDES.ucd.optz.value = np.array([1])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 5!")
        # scenario 6: initially charging, charging duration time met, decision not to charge
        self.ss.RTEDES.ucd.optz.value = np.array([0])
        self.assertTrue(self.ss.RTEDES.tcdr.e[0] <= 0,
                        "RTEDES.tcdr should be respected in scenario 6!")
