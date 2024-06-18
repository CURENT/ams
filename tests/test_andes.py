"""
Test ANDES interface.
"""

import unittest
import numpy as np
import pkg_resources
from pkg_resources import parse_version

import andes
import ams

from ams.interop.andes import (build_group_table, make_link_table, to_andes,
                               parse_addfile, verify_pf)


class TestMatrices(unittest.TestCase):
    """
    Tests for system matrices consistency.
    """

    andes_version = pkg_resources.get_distribution("andes").version
    if parse_version(andes_version) < parse_version('1.9.2'):
        raise unittest.SkipTest("Requires ANDES version >= 1.9.2")

    sp = ams.load(ams.get_case('5bus/pjm5bus_demo.xlsx'),
                  setup=True, no_output=True, default_config=True,)
    sa = sp.to_andes(setup=True, no_output=True, default_config=True,)

    def setUp(self) -> None:
        """
        Test setup.
        """

    def test_build_y(self):
        """
        Test build_y consistency.
        """
        ysp = self.sp.Line.build_y()
        ysa = self.sa.Line.build_y()
        np.testing.assert_equal(np.array(ysp.V), np.array(ysa.V))

    def test_build_Bp(self):
        """
        Test build_Bp consistency.
        """
        Bp_sp = self.sp.Line.build_Bp()
        Bp_sa = self.sa.Line.build_Bp()
        np.testing.assert_equal(np.array(Bp_sp.V), np.array(Bp_sa.V))

    def test_build_Bpp(self):
        """
        Test build_Bpp consistency.
        """
        Bpp_sp = self.sp.Line.build_Bpp()
        Bpp_sa = self.sa.Line.build_Bpp()
        np.testing.assert_equal(np.array(Bpp_sp.V), np.array(Bpp_sa.V))

    def test_build_Bdc(self):
        """
        Test build_Bdc consistency.
        """
        Bdc_sp = self.sp.Line.build_Bdc()
        Bdc_sa = self.sa.Line.build_Bdc()
        np.testing.assert_equal(np.array(Bdc_sp.V), np.array(Bdc_sa.V))


class TestInteropBase(unittest.TestCase):
    """
    Tests for basic function of ANDES interface.
    """
    ad_cases = [
        'ieee14/ieee14_full.xlsx',
        'ieee39/ieee39_full.xlsx',
        'npcc/npcc.xlsx',
    ]
    am_cases = [
        'ieee14/ieee14.json',
        'ieee39/ieee39.xlsx',
        'npcc/npcc_uced.xlsx',
    ]

    def setUp(self) -> None:
        """
        Test setup.
        """

    def test_basic_functions(self):
        """
        Test basic functions defined in ANDES Interop.
        """
        for ad_case in self.ad_cases:
            sa = andes.load(andes.get_case(ad_case),
                            setup=True, no_output=True, default_config=True,)
            # --- test build_group_table ---
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
            # --- test make_link_table ---
            link_table = make_link_table(adsys=sa)
            stg_idx = [str(i) for i in sa.PV.idx.v + sa.Slack.idx.v]
            self.assertSetEqual(set(stg_idx),
                                set(link_table['stg_idx'].values))
            bus_idx = [str(i) for i in sa.PV.bus.v + sa.Slack.bus.v]
            self.assertSetEqual(set(bus_idx),
                                set(link_table['bus_idx'].values))

    def test_convert(self):
        """
        Test conversion from AMS case to ANDES case.
        """
        for ad_case, am_case in zip(self.ad_cases, self.am_cases):
            sp = ams.load(ams.get_case(am_case),
                          setup=True, no_output=True, default_config=True,)
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

            # ensure PFlow models consistency
            pflow_mdls = list(sa.PFlow.models.keys())
            for mdl in pflow_mdls:
                self.assertTrue(sp.models[mdl].as_df().equals(sa.PFlow.models[mdl].as_df()))

    def test_convert_after_update(self):
        """
        Test conversion from AMS case to ANDES case after updating parameters.
        """
        for am_case in self.am_cases:
            sp = ams.load(ams.get_case(am_case),
                          setup=True, no_output=True, default_config=True,)
        # record initial values
        pq_idx = sp.PQ.idx.v
        p0 = sp.PQ.p0.v.copy()
        sa = to_andes(sp, setup=False, no_output=True, default_config=True)
        # before update
        np.testing.assert_array_equal(sp.PQ.p0.v, sa.PQ.p0.v)
        # after update
        sp.PQ.alter(src='p0', idx=pq_idx, value=0.9*p0)
        sa = to_andes(sp, setup=False, no_output=True, default_config=True)
        np.testing.assert_array_equal(sp.PQ.p0.v, sa.PQ.p0.v)

    def test_extra_dyn(self):
        """
        Test conversion when extra dynamic models exist.
        """
        sp = ams.load(ams.get_case('ieee14/ieee14_uced.xlsx'),
                      setup=True, no_output=True, default_config=True,)
        sa = to_andes(sp, addfile=andes.get_case('ieee14/ieee14_full.xlsx'),
                      setup=True, no_output=True, default_config=True,
                      verify=False, tol=1e-3)
        self.assertGreaterEqual(sa.PVD1.n, 0)

    def test_verify_pf(self):
        """
        Test verification of power flow results.
        """
        sp = ams.load(ams.get_case('matpower/case300.m'),
                      setup=True, no_output=True, default_config=True,)
        sa = to_andes(sp,
                      setup=True, no_output=True, default_config=True,
                      verify=False, tol=1e-3)
        # NOTE: it is known that there is 1e-7~1e-6 diff in case300.m
        self.assertFalse(verify_pf(amsys=sp, adsys=sa, tol=1e-6))
        self.assertTrue(verify_pf(amsys=sp, adsys=sa, tol=1e-3))


class TestDataExchange(unittest.TestCase):
    """
    Tests for data exchange between AMS and ANDES.
    """

    def setUp(self) -> None:
        """
        Test setup. This is executed before each test case.
        """
        self.sp = ams.load(ams.get_case('ieee14/ieee14_uced.xlsx'),
                           setup=True,
                           no_output=True,
                           default_config=True,)
        self.sp.RTED.run(solver='CLARABEL')
        self.sp.RTED.dc2ac()
        self.stg_idx = self.sp.RTED.pg.get_idx()

    def test_data_exchange(self):
        """
        Test data exchange between AMS and ANDES.
        """
        sa = to_andes(self.sp, setup=True,
                      addfile=andes.get_case('ieee14/ieee14_full.xlsx'),
                      no_output=True,
                      default_config=True,)
        # alleviate limiter
        sa.TGOV1.set(src='VMAX', attr='v', idx=sa.TGOV1.idx.v, value=100*np.ones(sa.TGOV1.n))
        sa.TGOV1.set(src='VMIN', attr='v', idx=sa.TGOV1.idx.v, value=np.zeros(sa.TGOV1.n))

        # --- test before PFlow ---
        self.sp.dyn.send(adsys=sa, routine='RTED')
        p0 = sa.StaticGen.get(src='p0', attr='v', idx=self.stg_idx)
        pg = self.sp.RTED.get(src='pg', attr='v', idx=self.stg_idx)
        np.testing.assert_array_equal(p0, pg)

        # --- test after TDS ---
        self.assertFalse(self.sp.dyn.is_tds)
        sa.PFlow.run()
        sa.TDS.init()
        self.assertTrue(self.sp.dyn.is_tds)

        sa.TDS.config.tf = 1
        sa.TDS.run()
        self.sp.dyn.send(adsys=sa, routine='RTED')
        self.sp.dyn.receive(adsys=sa, routine='RTED', no_update=False)
