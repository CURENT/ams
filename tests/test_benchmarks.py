import unittest
import numpy as np

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

    def setUp(self):
        self.case = ams.get_case('matpower/case5.m')

    def test_get_tool_versions(self):
        self.assertIsInstance(bp.get_tool_versions(), dict)

    @require_packages('mosek', 'gurobipy', 'pandapower')
    def test_run_routine(self):
        ss = ams.load(self.case, setup=True, default_config=True, no_output=True)
        _, _obj = bp.run_routine(ss, routine='DCOPF', solver='CLARABEL', ignore_dpp=False)

        np.testing.assert_array_less(np.zeros_like(_obj), _obj)

    @require_packages('mosek', 'gurobipy', 'pandapower')
    def test_run_dcopf_with_load_factors(self):
        ss = ams.load(self.case, setup=True, default_config=True, no_output=True)
        _, _obj = bp.run_dcopf_with_load_factors(ss, solver='CLARABEL',
                                                 load_factors=np.array([1.1, 1.2]), ignore_dpp=True)

        np.testing.assert_array_less(np.zeros_like(_obj), _obj)

    @require_packages('mosek', 'gurobipy', 'pandapower')
    def test_test_time(self):
        _, _obj = bp.test_time(self.case, routine='DCOPF', ignore_dpp=False)

        np.testing.assert_array_less(np.zeros_like(_obj), _obj)

    @require_packages('mosek', 'gurobipy', 'pandapower')
    def test_test_mtime(self):
        _, _obj = bp.test_mtime(self.case, load_factors=np.array([1.1, 1.2]), ignore_dpp=False)

        np.testing.assert_array_less(np.zeros_like(_obj), _obj)
