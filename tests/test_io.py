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
        self.assertTupleEqual(self.mpc5['gencost'].shape, (5, 6))
        self.assertTupleEqual(self.mpc14['gencost'].shape, (5, 7))

    def test_mpc2system(self):
        system5 = ams.system.System()
        ams.io.matpower.mpc2system(self.mpc5, system5)
        # In case5.m, the gencost has type 2 cost model, with 2 parameters.
        np.testing.assert_array_equal(system5.GCost.c2.v,
                                      np.zeros(system5.StaticGen.n))

        system14 = ams.system.System()
        ams.io.matpower.mpc2system(self.mpc14, system14)
        # In case14.m, the gencost has type 2 cost model, with 3 parameters.
        np.testing.assert_array_less(np.zeros(system14.StaticGen.n),
                                     system14.GCost.c2.v,)
