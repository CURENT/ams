"""
Test file input/output.
"""

import unittest
import numpy as np

import ams


class TestMATPOWER(unittest.TestCase):

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

        system14 = ams.system.System()
        # Test gentype length check
        mpc14 = self.mpc14.copy()
        mpc14['gentype'] = np.array(['WT'] * 6)
        with self.assertRaises(ValueError, msg='gentype length check failed!'):
            ams.io.matpower.mpc2system(mpc14, system14)

        ams.io.matpower.mpc2system(self.mpc14, system14)
        # In case14.m, the gencost has type 2 cost model, with 3 parameters.
        np.testing.assert_array_less(np.zeros(system14.StaticGen.n),
                                     system14.GCost.c2.v,)

    def test_gencost1(self):
        """Test when gencost is type 1."""
        mpcgc1 = self.mpc14.copy()
        mpcgc1['gencost'] = np.repeat(np.array([[1, 0, 0, 3, 0.01, 40, 0]]), 5, axis=0)

        system = ams.system.System()
        ams.io.matpower.mpc2system(mpcgc1, system)
        self.assertEqual(system.GCost.n, 5)
