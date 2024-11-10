import unittest

import ams
import ams.benchmarks as bp


def are_packages_available(*packages):
    """
    Check if the specified packages are available.

    Parameters
    ----------
    *packages : str
        Names of the packages to check.

    Returns
    -------
    bool
        True if all packages are available, False otherwise.
    """
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            return False
    return True


def require_packages(*packages):
    """
    Decorator to skip a test if the specified packages are not available.

    Parameters
    ----------
    *packages : str
        Names of the packages required for the test.

    Returns
    -------
    function
        The decorated test function.
    """
    def decorator(test_func):
        return unittest.skipIf(
            not are_packages_available(*packages),
            f"Skipping test because one or more required packages are not available: {', '.join(packages)}"
        )(test_func)
    return decorator


class TestBenchmarks(unittest.TestCase):
    """
    Test module for benchmarks.py.
    """

    def test_get_tool_versions(self):
        self.assertIsInstance(bp.get_tool_versions(), dict)

    def test_pre_solve_rted(self):
        ss = ams.load(ams.get_case('5bus/pjm5bus_demo.xlsx'),
                      setup=True, default_config=True, no_output=True)
        pre_time = bp.pre_solve(ss, 'RTED')
        self.assertIsInstance(pre_time, dict)
        self.assertEqual(len(pre_time), len(bp.cols_pre))
        for v in pre_time.values():
            self.assertIsInstance(v, float)

    def test_dcopf_solve(self):
        ss = ams.load(ams.get_case('matpower/case5.m'),
                      setup=True, default_config=True, no_output=True)
        s, obj = bp.time_routine_solve(ss, routine='DCOPF', solver='CLARABEL', ignore_dpp=False)
        self.assertGreater(s, 0)
        self.assertGreater(obj, 0)

    def test_pre_solve_ed(self):
        ss = ams.load(ams.get_case('5bus/pjm5bus_demo.xlsx'),
                      setup=True, default_config=True, no_output=True)
        pre_time = bp.pre_solve(ss, 'ED')
        self.assertIsInstance(pre_time, dict)
        self.assertEqual(len(pre_time), len(bp.cols_pre))
        for v in pre_time.values():
            self.assertIsInstance(v, float)

    def test_ed_solve(self):
        ss = ams.load(ams.get_case('5bus/pjm5bus_demo.xlsx'),
                      setup=True, default_config=True, no_output=True)
        s, obj = bp.time_routine_solve(ss, 'ED', solver='CLARABEL', ignore_dpp=True)
        self.assertGreater(s, 0)
        self.assertGreater(obj, 0)

    @require_packages('pandapower')
    def time_pdp_dcopf(self):
        ss = ams.load(ams.get_case('matpower/case5.m'),
                      setup=True, default_config=True, no_output=True)
        ppc = ams.io.pypower.system2ppc(ss)
        ppn = ams.io.pandapower.converter.from_ppc(ppc, f_hz=ss.config.freq)
        s, obj = bp.time_pdp_dcopf(ppn)
        self.assertGreater(s, 0)
        self.assertGreater(obj, 0)

    def test_time_dcopf(self):
        ss = ams.load(ams.get_case('matpower/case5.m'),
                      setup=True, default_config=True, no_output=True)
        pre_time, sol = bp.time_routine(ss,
                                        routine='DCOPF',
                                        solvers=['CLARABEL', 'SCIP', 'pandapower'], ignore_dpp=False)
        for v in pre_time.values():
            self.assertGreaterEqual(v, 0)

        self.assertGreater(sol['CLARABEL']['time'], 0)
        self.assertGreater(sol['SCIP']['time'], 0)
        self.assertAlmostEqual(sol['CLARABEL']['obj'],
                               sol['SCIP']['obj'],
                               places=2)
        if not are_packages_available('pandapower'):
            self.assertEqual(sol['pandapower']['time'], -1)
            self.assertEqual(sol['pandapower']['obj'], -1)
        else:
            self.assertGreater(sol['pandapower']['obj'], 0)
            self.assertAlmostEqual(sol['CLARABEL']['obj'],
                                   sol['pandapower']['obj'],
                                   places=2)

    def test_time_rted(self):
        ss = ams.load(ams.get_case('5bus/pjm5bus_demo.xlsx'),
                      setup=True, default_config=True, no_output=True)
        pre_time, sol = bp.time_routine(ss,
                                        routine='RTED',
                                        solvers=['CLARABEL', 'SCIP', 'pandapower'], ignore_dpp=False)
        for v in pre_time.values():
            self.assertGreaterEqual(v, 0)

        self.assertGreater(sol['CLARABEL']['time'], 0)
        self.assertGreater(sol['SCIP']['time'], 0)
        self.assertAlmostEqual(sol['CLARABEL']['obj'],
                               sol['SCIP']['obj'],
                               places=2)
        self.assertEqual(sol['pandapower']['time'], -1)
        self.assertEqual(sol['pandapower']['obj'], -1)

    def test_time_dcopf_with_lf(self):
        ss = ams.load(ams.get_case('matpower/case5.m'),
                      setup=True, default_config=True, no_output=True)
        pre_time, sol = bp.time_dcopf_with_lf(ss, solvers=['CLARABEL', 'SCIP'], load_factors=[
                                              1, 0.5, 0.25], ignore_dpp=False)
        self.assertEqual(len(pre_time), len(bp.cols_pre))
        self.assertAlmostEqual(sol['CLARABEL']['obj'],
                               sol['SCIP']['obj'],
                               places=2)
