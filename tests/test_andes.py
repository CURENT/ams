"""
Test ANDES interface.
"""

import unittest
import numpy as np

import andes
import ams

from ams.interop.andes import build_group_table, to_andes, parse_addfile


class TestInteropBase(unittest.TestCase):
    """
    Tests for basic function of ANDES interface.

    # TODO: add case that involve PVD1 or REGCV1
    """
    ad_cases = [
        'ieee14/ieee14_full.xlsx',
        'ieee39/ieee39_full.xlsx',
    ]
    am_cases = [
        'ieee14/ieee14.json',
        'ieee39/ieee39.xlsx',
    ]

    def setUp(self) -> None:
        """
        Test setup.
        """

    def test_build_group_table(self):
        for ad_case in self.ad_cases:
            sa = andes.load(andes.get_case(ad_case),
                            setup=True,
                            no_output=True,
                            default_config=True,)
            ssa_stg = build_group_table(adsys=sa,
                                        grp_name='StaticGen',
                                        param_name=['u', 'name', 'idx', 'bus'],
                                        mdl_name=[])
            self.assertEqual(ssa_stg.shape,
                             (sa.StaticGen.n, 4))

            ssa_gov = build_group_table(adsys=sa, grp_name='TurbineGov',
                                        param_name=['idx', 'syn'],
                                        mdl_name=[])
            self.assertEqual(ssa_gov.shape,
                             (sa.TurbineGov.n, 2))

    def test_convert(self):
        for ad_case, am_case in zip(self.ad_cases, self.am_cases):
            sp = ams.load(ams.get_case(am_case),
                          setup=True,
                          no_output=True,
                          default_config=True,)
            # before addfile
            sa = to_andes(sp, setup=False)
            self.assertEqual(set(sp.PV.idx.v), set(sa.PV.idx.v))
            self.assertEqual(set(sp.Bus.idx.v), set(sa.Bus.idx.v))
            self.assertEqual(set(sp.Line.idx.v), set(sa.Line.idx.v))
            self.assertEqual(np.sum(sp.PQ.p0.v), np.sum(sa.PQ.p0.v))

            # after addfile
            sa = parse_addfile(adsys=sa, amsys=sp,
                               addfile=andes.get_case(ad_case))
            sa.setup()
            set1 = set(sa.GENROU.gen.v)
            set2 = set(sp.StaticGen.get_idx())
            # set2 includes set1, ensure GENROU.gen are all in StaticGen.idx
            self.assertEqual(set1, set1 & set2)


class TestANDES(unittest.TestCase):
    """
    Tests for ANDES interface.

    # TODO: add tests for ANDES interface functions.
    """
    cases = [
        '5bus/pjm5bus_demo.xlsx',
        'ieee14/ieee14_uced.xlsx',
        'ieee39/ieee39_uced_esd1.xlsx',
        'matpower/case118.m',
    ]

    def setUp(self) -> None:
        """
        Test setup. This is executed before each test case.
        """
