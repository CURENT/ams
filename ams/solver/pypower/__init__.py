# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
PYPOWER solves power flow and Optimal Power Flow (OPF) problems.

This module is developed from module pandapower.pypower.
"""

# TODO: add a comparasion function of this module with pandapower.pypower
from numpy import array


class pp_system():
    """
    PyPower system class, as the interface to PyPower
    """

    def __init__(self, case):
        """
        PyPower system class, as the interface to PyPower.

        Parameters
        ----------
        case: str
            PyPower case file name, such as 'case14.py'

        Examples
        --------
        >>> import ams
        >>> spp = pp_system(case=ams.get_case('pypower/case14.py'))
        """
        exec(open(f"{case}").read())
        func_name = case.split('/')[-1].rstrip('.py')
        self.ppc = eval(f"{func_name}()")  # the dict of PyPower case

    def set_case(self, case):
        pass

    def set_solver(self, solver):
        pass

    def solve(self):
        pass

    def get_results(self):
        pass
