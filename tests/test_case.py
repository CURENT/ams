import os
import unittest

import numpy as np

import ams
from ams.utils.paths import get_case


class Test5Bus(unittest.TestCase):
    """
    Tests for the 5-bus system.
    """

    def setUp(self) -> None:
        self.ss = ams.main.load(
            get_case("5bus/pjm5bus_demo.json"),
            default_config=True,
            no_output=True,
        )

    def test_essential(self):
        """
        Test essential functionalities of Model and System.
        """

        # --- test model names
        self.assertTrue("Bus" in self.ss.models)
        self.assertTrue("PQ" in self.ss.models)

        nBus = 5
        nGen = 5
        nPQ = 3
        nArea = 3
        nZone = 5
        nDT = 24
        # --- test device counts
        self.assertEqual(self.ss.Bus.n, nBus)
        self.assertEqual(self.ss.PQ.n, nPQ)
        self.assertEqual(self.ss.PV.n, 4)
        self.assertEqual(self.ss.Slack.n, 1)
        self.assertEqual(self.ss.Line.n, 7)
        self.assertEqual(self.ss.Zone.n, nZone)
        self.assertEqual(self.ss.Area.n, nArea)
        self.assertEqual(self.ss.SFR.n, nArea)
        self.assertEqual(self.ss.SR.n, nArea)
        self.assertEqual(self.ss.NSR.n, nArea)
        self.assertEqual(self.ss.GCost.n, nGen)
        self.assertEqual(self.ss.DCost.n, nPQ)
        self.assertEqual(self.ss.SFRCost.n, nGen)
        self.assertEqual(self.ss.SRCost.n, nGen)
        self.assertEqual(self.ss.NSRCost.n, nGen)
        self.assertEqual(self.ss.EDTSlot.n, nDT)
        self.assertEqual(self.ss.UCTSlot.n, nDT)

        # test idx values
        self.assertSequenceEqual(self.ss.Bus.idx.v, [0, 1, 2, 3, 4])
        self.assertSequenceEqual(self.ss.Area.idx.v, [1, 2, 3])

        # test cache refreshing
        self.ss.Bus.cache.refresh()  # used in ANDES but not in AMS

        # test conversion to dataframe
        self.ss.Bus.as_df()
        self.ss.Bus.as_df(vin=True)

        # test conversion to dataframe of ``Horizon`` model
        self.ss.EDTSlot.as_df()
        self.ss.EDTSlot.as_df(vin=True)

        self.ss.UCTSlot.as_df()
        self.ss.UCTSlot.as_df(vin=True)

    def test_pflow_reset(self):
        """
        Test resetting power flow.
        """

        self.ss.PFlow.run()
        self.ss.reset()
        self.ss.PFlow.run()

    def test_alter_param(self):
        """
        Test altering parameter for power flow.
        """

        self.ss.PV.alter("v0", "PV_3", 0.98)
        self.assertEqual(self.ss.PV.v0.v[1], 0.98)
        self.ss.PFlow.run()

    def test_alter_param_before_routine(self):
        """
        Test altering parameter before running routine.
        """

        self.ss.GCost.alter("c1", ['GCost_1', 'GCost_2'], [1500., 3100.])
        np.testing.assert_array_equal(self.ss.GCost.c1.v, [1500., 3100., 0.4, 0.1, 0.01])
        self.ss.ACOPF.run()
        np.testing.assert_array_equal(self.ss.GCost.c1.v, [1500., 3100., 0.4, 0.1, 0.01])

    def test_alter_param_after_routine(self):
        """
        Test altering parameter after running routine.
        """

        self.ss.ACOPF.run()
        self.ss.GCost.alter("c1", ['GCost_1', 'GCost_2'], [1500., 3100.])
        np.testing.assert_array_equal(self.ss.GCost.c1.v, [1500., 3100., 0.4, 0.1, 0.01])
        self.ss.ACOPF.run()
        np.testing.assert_array_equal(self.ss.GCost.c1.v, [1500., 3100., 0.4, 0.1, 0.01])

    # def test_multiple_disconnected_line(self):
    #     """
    #     Test connectivity check for systems with disconnected lines.

    #     These disconnected lines (zeros) was not excluded when counting
    #     connected buses, causing an out-of-bound error.
    #     """
    #     # TODO: need to add `connectivity` in `system`
    #     pass
    #     # self.ss.Line.u.v[[0, 6]] = 0
    #     # self.ss.PFlow.run()
    #     # self.assertEqual(len(self.ss.Bus.islands), 1)
    #     # self.assertEqual(self.ss.Bus.n_islanded_buses, 0)


class TestIEEE14RAW(unittest.TestCase):
    """
    Test IEEE14 system in the RAW format.
    """

    # TODO: after add `run` in `system`, improve this part
    def test_ieee14_raw(self):
        ss = ams.load(
            get_case("ieee14/ieee14.raw"),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        self.assertEqual(ss.PFlow.exit_code, 0, "Exit code is not 0.")

    def test_ieee14_raw_convert(self):
        ss = ams.run(
            get_case("ieee14/ieee14.raw"),
            convert=True,
            default_config=True,
        )
        os.remove(ss.files.dump)
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_ieee14_raw2xlsx(self):
        ss = ams.load(
            get_case("ieee14/ieee14.raw"),
            setup=True,
            no_output=True,
            default_config=True,
        )
        ams.io.xlsx.write(ss, "ieee14.xlsx", overwrite=True)
        self.assertTrue(os.path.exists("ieee14.xlsx"))
        os.remove("ieee14.xlsx")

    def test_ieee14_raw2json(self):
        ss = ams.load(
            get_case("ieee14/ieee14.raw"),
            setup=True,
            no_output=True,
            default_config=True,
        )
        ams.io.json.write(ss, "ieee14.json", overwrite=True)
        self.assertTrue(os.path.exists("ieee14.json"))
        os.remove("ieee14.json")

    def test_ieee14_raw2json_convert(self):
        ss = ams.run(
            get_case("ieee14/ieee14.raw"),
            convert="json",
            default_config=True,
        )

        ss2 = ams.run(
            "ieee14.json",
            default_config=True,
            no_output=True,
        )

        os.remove(ss.files.dump)
        self.assertEqual(ss2.exit_code, 0, "Exit code is not 0.")

    def test_read_json_from_memory(self):
        fd = open(get_case("ieee14/ieee14.json"), "r")

        ss = ams.main.System(
            default_config=True,
            no_output=True,
        )
        ams.io.json.read(ss, fd)
        ss.setup()
        ss.PFlow.run()

        fd.close()
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_read_mpc_from_memory(self):
        fd = open(get_case("matpower/case14.m"), "r")

        ss = ams.main.System(
            default_config=True,
            no_output=True,
        )
        ams.io.matpower.read(ss, fd)
        ss.setup()
        ss.PFlow.run()

        fd.close()
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_read_psse_from_memory(self):
        fd_raw = open(get_case("ieee14/ieee14.raw"), "r")

        ss = ams.main.System(
            default_config=True,
            no_output=True,
        )
        # suppress out-of-normal info
        ss.config.warn_limits = 0
        ss.config.warn_abnormal = 0

        ams.io.psse.read(ss, fd_raw)
        ss.setup()
        ss.PFlow.run()

        fd_raw.close()
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")


class TestCaseInit(unittest.TestCase):
    """
    Test if initializations pass.
    """

    def test_ieee39_init(self):
        """
        Test if ieee39 initialization works.
        """
        ss = ams.load(
            get_case("ieee39/ieee39_uced.xlsx"),
            default_config=True,
            no_output=True,
        )
        ss.DCOPF.init()
        ss.RTED.init()
        ss.ED.init()
        ss.UC.init()

        self.assertEqual(ss.DCOPF.exit_code, 0, "Exit code is not 0.")
        self.assertEqual(ss.RTED.exit_code, 0, "Exit code is not 0.")
        self.assertEqual(ss.ED.exit_code, 0, "Exit code is not 0.")
        self.assertEqual(ss.UC.exit_code, 0, "Exit code is not 0.")

    def test_ieee39_esd1_init(self):
        """
        Test if ieee39 with ESD1 initialization works.
        """
        ss = ams.load(
            get_case("ieee39/ieee39_uced_esd1.xlsx"),
            default_config=True,
            no_output=True,
        )
        ss.EDES.init()
        ss.UCES.init()

        self.assertEqual(ss.EDES.exit_code, 0, "Exit code is not 0.")
        self.assertEqual(ss.UCES.exit_code, 0, "Exit code is not 0.")


class TestCase14(unittest.TestCase):
    """
    Test parameter correction using case14.m
    """

    def test_parameter_correction(self):
        """
        Test if the parameter correction works.
        """
        mpc = ams.io.matpower.m2mpc(get_case("matpower/case14.m"))
        mpc['branch'][:, 11] = 0.0
        mpc['branch'][:, 12] = 0.0

        ss = ams.system.System()
        ams.io.matpower.mpc2system(mpc, ss)
        ss.setup()

        # line rate
        np.testing.assert_array_less(0.0, ss.Line.rate_a.v)
        np.testing.assert_array_less(0.0, ss.Line.rate_b.v)
        np.testing.assert_array_less(0.0, ss.Line.rate_c.v)

        # line angle difference
        np.testing.assert_array_less(0.0, ss.Line.amax.v)
        np.testing.assert_array_less(ss.Line.amin.v, 0.0)
