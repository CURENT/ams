"""
Test wrapper routines of gurobi_optimodes.
"""
import unittest

import ams
from ams.shared import skip_unittest_without_gurobi_optimods


class TestOPF(unittest.TestCase):
    """
    Test routine `OPF` in DC.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("5bus/pjm5bus_demo.json"),
                           setup=True, default_config=True, no_output=True)
        # decrease load first
        self.ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])

    def test_init(self):
        """
        Test initialization.
        """
        self.ss.OPF.init()
        self.assertTrue(self.ss.OPF.initialized, "OPF initialization failed!")

    @skip_unittest_without_gurobi_optimods
    def test_trip_gen(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.OPF.update()
        self.ss.OPF.run(opftype='DC')
        self.assertTrue(self.ss.OPF.converged, "OPF in DC did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.OPF.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    @skip_unittest_without_gurobi_optimods
    def test_trip_gen_ac(self):
        """
        Test generator tripping.
        """
        stg = 'PV_1'
        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

        self.ss.OPF.update()
        self.ss.OPF.run(opftype='AC')
        self.assertTrue(self.ss.OPF.converged, "OPF in AC did not converge under generator trip!")
        self.assertAlmostEqual(self.ss.OPF.get(src='pg', attr='v', idx=stg),
                               0, places=6,
                               msg="Generator trip does not take effect!")

        self.ss.StaticGen.set(src='u', idx=stg, attr='v', value=1)  # reset

    @skip_unittest_without_gurobi_optimods
    def test_trip_line(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.OPF.update()
        self.ss.OPF.run(opftype='DC')
        self.assertTrue(self.ss.OPF.converged, "OPF in DC did not converge under line trip!")
        self.assertAlmostEqual(self.ss.OPF.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    @skip_unittest_without_gurobi_optimods
    def test_trip_line_ac(self):
        """
        Test line tripping.
        """
        self.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

        self.ss.OPF.update()
        self.ss.OPF.run(opftype='AC')
        self.assertTrue(self.ss.OPF.converged, "OPF in AC did not converge under line trip!")
        self.assertAlmostEqual(self.ss.OPF.get(src='plf', attr='v', idx='Line_3'),
                               0, places=6,
                               msg="Line trip does not take effect!")

        self.ss.Line.alter(src='u', idx='Line_3', value=1)  # reset

    @skip_unittest_without_gurobi_optimods
    def test_set_load(self):
        """
        Test setting and tripping load.
        """
        # --- run OPF ---
        self.ss.OPF.run(opftype='DC')
        pgs = self.ss.OPF.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.OPF.update()

        self.ss.OPF.run(opftype='DC')
        pgs_pqt = self.ss.OPF.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.OPF.update()

        self.ss.OPF.run(opftype='DC')
        pgs_pqt2 = self.ss.OPF.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")

    @skip_unittest_without_gurobi_optimods
    def test_set_load_ac(self):
        """
        Test setting and tripping load.
        """
        # --- run OPF ---
        self.ss.OPF.run(opftype='AC')
        pgs = self.ss.OPF.pg.v.sum()

        # --- set load ---
        self.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
        self.ss.OPF.update()

        self.ss.OPF.run(opftype='AC')
        pgs_pqt = self.ss.OPF.pg.v.sum()
        self.assertLess(pgs_pqt, pgs, "Load set does not take effect!")

        # --- trip load ---
        self.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
        self.ss.OPF.update()

        self.ss.OPF.run(opftype='AC')
        pgs_pqt2 = self.ss.OPF.pg.v.sum()
        self.assertLess(pgs_pqt2, pgs_pqt, "Load trip does not take effect!")
