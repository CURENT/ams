"""
PYPOWER solves power flow and Optimal Power Flow (OPF) problems.

This module is developed from PyPower.
"""

# TODO: add a comparasion function of this module with pandapower.pypower
from numpy import array


class pp_system():
    """
        PyPower system class, as the interface to PyPower.

        It has the following methods:
        ``from_ppc``: set the PyPower case dict ``ppc``

        Examples
        --------
        >>> spp = pp_system()
        >>> spp.from_ppc(ams.get_case('pypower/case14.py'))
    """

    def __init__(self):
        self.ppc = None
        # 'ppc' for PyPower case, 'mpc' for Matpower case
        # 'ads' for ANDES system, 'ams' for AMS system
        # TODO: this source_type seems not necessary
        self.source_type = None

    def from_ppc(self, case):
        exec(open(f"{case}").read())
        # NOTE: the following line is not robust
        func_name = case.split('/')[-1].rstrip('.py')
        self.ppc = eval(f"{func_name}()")
        self.source_type = 'ppc'

    def set_case(self, case):
        pass

    def set_solver(self, solver):
        pass

    def solve(self):
        pass

    def get_results(self):
        pass
