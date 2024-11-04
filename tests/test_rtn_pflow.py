import unittest

import json
import numpy as np

from andes.shared import rad2deg

import ams


class TestDCPF(unittest.TestCase):
    """
    Test routine `DCPF`.
    """

    def setUp(self) -> None:
        with open(ams.get_case('matpower/benchmark.json'), 'r') as file:
            self.mpres = json.load(file)

        self.ss = ams.load(ams.get_case('matpower/case14.m'),
                           setup=True, no_output=True, default_config=True)

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.DCPF.init()
        self.assertTrue(self.ss.DCPF.initialized, "DCPF initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 2
        self.ss.StaticGen.set(src='u', attr='v', idx=stg, value=0)

        self.ss.DCPF.update()
        self.ss.DCPF.run()
        self.assertTrue(self.ss.DCPF.converged, "DCPF did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.DCPF.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.alter(src='u', idx=stg, value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.DCPF.update()
        self.ss.DCPF.run()
        self.assertTrue(self.ss.DCPF.converged, "DCPF did not converge under line trip!")
        self.assertAlmostEqual(self.ss.DCPF.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run DCPF ---
        self.ss.DCPF.run()
        pgs = self.ss.DCPF.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.DCPF.update()

        self.ss.DCPF.run()
        pgs_pqt = self.ss.DCPF.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0)
        self.ss.DCPF.update()

        self.ss.DCPF.run()
        pgs_pqt2 = self.ss.DCPF.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")
