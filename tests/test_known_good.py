import logging
import unittest

import ams

logger = logging.getLogger(__name__)


class TestKnownResults(unittest.TestCase):
    # NOTE: DCOPF objective values are from MATPOWER v8.0b1
    sets = (
        (ams.get_case('matpower/case14.m'), 7642.59177699),
        (ams.get_case('matpower/case39.m'), 41263.94078588),
        (ams.get_case('matpower/case118.m'), 125947.8814179),
    )

    def test_known_results(self):
        for case, obj in self.sets:
            sp = ams.load(case, setup=True, no_output=True)
            sp.DCOPF.run(solver='ECOS')
            msg = f'Case: {case}; Result: {sp.DCOPF.obj.v}; Expected: {obj}'
            logger.info(msg)
            self.assertAlmostEqual(sp.DCOPF.obj.v, obj, places=2)
