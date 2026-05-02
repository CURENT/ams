import os

import unittest

import ams


class TestDCOPF(unittest.TestCase):
    """
    Test routine `DCOPF`.

    Common scenarios (init, trip_gen, trip_line, set_load, vBus, dc2ac)
    are exercised in `tests/routines/test_scenarios_lp_singleperiod.py`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])

    def test_export_csv(self):
        """
        Test `Routine.export_csv()` method.
        """
        self.ss.DCOPF.run(solver='CLARABEL')

        # test when path is none
        out = self.ss.DCOPF.export_csv()
        self.assertTrue(os.path.exists(out), "CSV export failed!")

        # test when path is not none
        out2 = self.ss.DCOPF.export_csv(path='test_dcopf.csv')
        self.assertTrue(os.path.exists(out2), "CSV export with path failed!")

        # Clean up
        os.remove(out)
        os.remove(out2)
