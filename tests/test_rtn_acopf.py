import unittest

import ams


class TestACOPF(unittest.TestCase):
    """
    Test routine `ACOPF`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case('matpower/case14.m'),
                           setup=True, no_output=True, default_config=True)

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.ACOPF.init()
        self.assertTrue(self.ss.ACOPF.initialized, "ACOPF initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 2
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.ACOPF.update()
        self.ss.ACOPF.run()
        self.assertTrue(self.ss.ACOPF.converged, "ACOPF did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.ACOPF.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.ACOPF.update()
        self.ss.ACOPF.run()
        self.assertTrue(self.ss.ACOPF.converged, "ACOPF did not converge under line trip!")
        self.assertAlmostEqual(self.ss.ACOPF.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run ACOPF ---
        self.ss.ACOPF.run()
        pgs = self.ss.ACOPF.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.ACOPF.update()

        self.ss.ACOPF.run()
        pgs_pqt = self.ss.ACOPF.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0)
        self.ss.ACOPF.update()

        self.ss.ACOPF.run()
        pgs_pqt2 = self.ss.ACOPF.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")
