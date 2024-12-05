import unittest

import ams


class TestDCPF(unittest.TestCase):
    """
    Test routine `DCPF`.
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
        self.ss.DCPF.init()
        self.assertTrue(self.ss.DCPF.initialized, "DCPF initialization failed!")

    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.DCPF.update()
        self.ss.DCPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCPF.converged, "DCPF did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.DCPF.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.DCPF.update()
        self.ss.DCPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.DCPF.converged, "DCPF did not converge under line trip!")
        self.assertAlmostEqual(self.ss.DCPF.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run DCPF ---
        self.ss.DCPF.run(solver='CLARABEL')
        pgs = self.ss.DCPF.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.DCPF.update()

        self.ss.DCPF.run(solver='CLARABEL')
        pgs_pqt = self.ss.DCPF.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.DCPF.update()

        self.ss.DCPF.run(solver='CLARABEL')
        pgs_pqt2 = self.ss.DCPF.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")
