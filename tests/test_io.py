"""
Test file input/output.
"""
import os
import unittest
import numpy as np

import ams


class TestMATPOWER(unittest.TestCase):
    """
    Test IO functions for MATPOWER and PYPOWER.
    """

    def setUp(self):
        self.mpc5 = ams.io.matpower.m2mpc(ams.get_case('matpower/case5.m'))
        self.mpc14 = ams.io.matpower.m2mpc(ams.get_case('matpower/case14.m'))

    def test_m2mpc(self):
        """Test conversion from M file to mpc dict."""
        # NOTE: when the keys are there, read them
        self.assertTupleEqual(self.mpc5['gencost'].shape, (5, 6))
        self.assertTupleEqual(self.mpc5['gentype'].shape, (5,))
        self.assertTupleEqual(self.mpc5['genfuel'].shape, (5,))

        # NOTE: when the keys are not there, the read mpc will not complete them
        self.assertTupleEqual(self.mpc14['gencost'].shape, (5, 7))
        self.assertNotIn('gentype', self.mpc14)
        self.assertNotIn('genfuel', self.mpc14)

    def test_mpc2system(self):
        """Test conversion from MPC to AMS System."""
        system5 = ams.system.System()
        ams.io.matpower.mpc2system(self.mpc5, system5)
        # In case5.m, the gencost has type 2 cost model, with 2 parameters.
        np.testing.assert_array_equal(system5.GCost.c2.v,
                                      np.zeros(system5.StaticGen.n))

        # Check if Area is added
        self.assertGreater(system5.Area.n, 0)

        # Check if Zone is added
        self.assertGreater(system5.Zone.n, 0)

        system14 = ams.system.System()
        # Test gentype length check
        mpc14 = self.mpc14.copy()
        mpc14['gentype'] = np.array(['WT'] * 6)
        with self.assertRaises(ValueError, msg='gentype length check failed!'):
            ams.io.matpower.mpc2system(mpc14, system14)

        system14 = ams.system.System()
        ams.io.matpower.mpc2system(self.mpc14, system14)
        # In case14.m, the gencost has type 2 cost model, with 3 parameters.
        np.testing.assert_array_less(np.zeros(system14.StaticGen.n),
                                     system14.GCost.c2.v,)

    def test_system2mpc(self):
        """Test conversion from AMS System to MPC."""
        system5 = ams.system.System()
        ams.io.matpower.mpc2system(self.mpc5, system5)
        mpc5 = ams.io.matpower.system2mpc(system5)

        self.assertEqual(mpc5['baseMVA'], self.mpc5['baseMVA'])

        # Bus
        # type, PD, QD, GS,BS, VM, VA. BASE_KV, VMAX, VMIN
        bus_cols = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12]
        np.testing.assert_array_equal(mpc5['bus'][:, bus_cols],
                                      self.mpc5['bus'][:, bus_cols])

        # Branch, Gen, Gencost, can have minor differences but is okay

        # String type data
        np.testing.assert_array_equal(mpc5['gentype'], self.mpc5['gentype'])
        np.testing.assert_array_equal(mpc5['genfuel'], self.mpc5['genfuel'])
        np.testing.assert_array_equal(mpc5['bus_name'], self.mpc5['bus_name'])

        # Area quantity
        self.assertEqual(np.unique(mpc5['bus'][:, 6]).shape[0],
                         np.unique(self.mpc5['bus'][:, 6]).shape[0])

        # Zone quantity
        self.assertEqual(np.unique(mpc5['bus'][:, 10]).shape[0],
                         np.unique(self.mpc5['bus'][:, 10]).shape[0])

    def test_gencost1(self):
        """Test when gencost is type 1."""
        mpcgc1 = self.mpc14.copy()
        mpcgc1['gencost'] = np.repeat(np.array([[1, 0, 0, 3, 0.01, 40, 0]]), 5, axis=0)

        system = ams.system.System()
        ams.io.matpower.mpc2system(mpcgc1, system)
        self.assertEqual(system.GCost.n, 5)

    def test_mpc2m(self):
        """Test conversion from MPC to M file."""
        mpc5 = ams.io.matpower.m2mpc(ams.get_case('matpower/case5.m'))
        mpc14 = ams.io.matpower.m2mpc(ams.get_case('matpower/case14.m'))

        # Test conversion to M file
        mfile5 = ams.io.matpower.mpc2m(mpc5, './case5out.m')
        mfile14 = ams.io.matpower.mpc2m(mpc14, './case14out.m')

        # Check if the files exist
        self.assertTrue(os.path.exists(mfile5))
        self.assertTrue(os.path.exists(mfile14))

        mpc5read = ams.io.matpower.m2mpc(mfile5)
        mpc14read = ams.io.matpower.m2mpc(mfile14)

        # Check if the numerical values are the same
        for key in mpc5:
            if key in ['bus_name', 'gentype', 'genfuel']:
                continue
            np.testing.assert_array_almost_equal(
                mpc5[key], mpc5read[key], decimal=5,
                err_msg=f"Mismatch in {key} when converting case5.m"
            )
        for key in mpc14:
            if key in ['bus_name', 'gentype', 'genfuel']:
                continue
            np.testing.assert_array_almost_equal(
                mpc14[key], mpc14read[key], decimal=5,
                err_msg=f"Mismatch in {key} when converting case14.m"
            )

        # Clean up the generated files
        os.remove(mfile5)
        os.remove(mfile14)

    def test_system2m(self):
        """Test conversion from AMS System to M file."""
        s5 = ams.load(ams.get_case('matpower/case5.m'),
                      no_output=True)
        mfile5 = './case5out.m'
        ams.io.matpower.write(s5, mfile5)

        # Clean up the generated file
        os.remove(mfile5)


class TestSystemExport(unittest.TestCase):
    """
    Test system export functions.
    """

    def test_system_to(self):
        """Test conversion from AMS System to several formats."""
        ss = ams.load(ams.get_case('matpower/case5.m'),
                      no_output=True)

        mpc = ss.to_mpc()

        self.assertIsInstance(mpc, dict)
        self.assertIn('bus', mpc)
        self.assertIn('branch', mpc)
        self.assertIn('gen', mpc)
        self.assertIn('gencost', mpc)
        self.assertIn('baseMVA', mpc)
        self.assertIn('bus_name', mpc)
        self.assertIn('gentype', mpc)
        self.assertIn('genfuel', mpc)

        ss.to_m("./case5_out.m")
        self.assertTrue(os.path.exists("./case5_out.m"))
        os.remove("./case5_out.m")

        ss.to_json("./case5_out.json")
        self.assertTrue(os.path.exists("./case5_out.json"))
        os.remove("./case5_out.json")

        ss.to_xlsx("./case5_out.xlsx")
        self.assertTrue(os.path.exists("./case5_out.xlsx"))
        os.remove("./case5_out.xlsx")

        ss.to_raw("./case5_out.raw", overwrite=True)
        self.assertTrue(os.path.exists("./case5_out.raw"))
        os.remove("./case5_out.raw")
