import unittest

import ams


class TestDCOPF(unittest.TestCase):
    """
    Test routine `DCOPF`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.xlsx"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.DCOPF.init()
        self.assertTrue(self.ss.DCOPF.initialized, "DCOPF initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.DCOPF.update()
        self.ss.DCOPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCOPF.converged, "DCOPF did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.DCOPF.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.DCOPF.update()
        self.ss.DCOPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCOPF.converged, "DCOPF did not converge under line trip!")
        self.assertAlmostEqual(self.ss.DCOPF.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run DCOPF ---
        self.ss.DCOPF.run(solver='CLARABEL')
        pgs = self.ss.DCOPF.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='CLARABEL')
        pgs_pqt = self.ss.DCOPF.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.DCOPF.update()

        self.ss.DCOPF.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.DCOPF.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")
