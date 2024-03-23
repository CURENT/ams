import logging
import unittest

import numpy as np

import ams

logger = logging.getLogger(__name__)


class TestKnownResults(unittest.TestCase):
    # NOTE: Baseline comes from MATPOWER v8.0b1
    # NOTE: when using ECOS in this case, the accuracy is ralatively low

    def test_known_case14(self):
        case = ams.get_case('matpower/case14.m')
        sp = ams.load(case, setup=True, no_output=True)
        sp.DCOPF.run(solver='ECOS')
        self.assertAlmostEqual(sp.DCOPF.obj.v, 7642.59177715644, places=2)

        aBus_mp = np.array([
            -0.0, -0.088451789, -0.22695322, -0.18549695,
            -0.15942929, -0.25995018, -0.24348912, -0.24348912,
            -0.27468284, -0.2795552, -0.27334388, -0.27941265,
            -0.28242718, -0.30074116,])
        aBus_mp -= aBus_mp[0]

        aBus = sp.DCOPF.aBus.v - sp.DCOPF.aBus.v[0]
        self.assertTrue(np.allclose(aBus, aBus_mp, atol=1e-4))

        pg_mp = np.array([
            2.2096769, 0.38032305, 0.0, 0.0, 0.0])
        pg = sp.DCOPF.pg.v
        self.assertTrue(np.allclose(pg, pg_mp, atol=1e-4))

    def test_known_case39(self):
        case = ams.get_case('matpower/case39.m')
        sp = ams.load(case, setup=True, no_output=True)
        sp.DCOPF.run(solver='ECOS')
        self.assertAlmostEqual(sp.DCOPF.obj.v, 41263.94078592735, places=2)

        pi = sp.DCOPF.pi.v / sp.config.mva
        self.assertTrue(np.all(np.isclose(pi, 13.51692, atol=1e-2)))

    def test_known_case118(self):
        case = ams.get_case('matpower/case118.m')
        sp = ams.load(case, setup=True, no_output=True)
        sp.DCOPF.run(solver='ECOS')
        self.assertAlmostEqual(sp.DCOPF.obj.v, 125947.8814181522, places=2)

        pi = sp.DCOPF.pi.v / sp.config.mva
        self.assertTrue(np.all(np.isclose(pi, 39.38136794580036, atol=1e-2)))
