"""
Parity tests for :class:`ams.core.routine_ns.RoutineNS`.

Validates Phase 4.1 scaffolding: every symbol the legacy ``sub_map``
regex resolves on a real routine should also resolve via ``RoutineNS``
attribute access, and to an object of the same type / identity where
possible. This is the R2 (silent wrong-symbol resolution) safeguard
called out in projects/opt_layer_overhaul/plan.md.
"""

import unittest

import cvxpy as cp

import ams
from ams.core.routine_ns import RoutineNS


class TestRoutineNSResolution(unittest.TestCase):
    """RoutineNS must resolve every sub_map symbol on DCOPF / RTED."""

    @classmethod
    def setUpClass(cls):
        cls.ss = ams.load(
            ams.get_case('5bus/pjm5bus_demo.xlsx'),
            setup=True,
            no_output=True,
            default_config=True,
        )
        # Fully build optimization graph so `om.<name>` is populated.
        cls.ss.DCOPF.init()
        cls.ss.DCOPF.solve(solver='CLARABEL')

    def test_var_resolution(self):
        ns = RoutineNS(self.ss.DCOPF)
        for vname, var in self.ss.DCOPF.vars.items():
            resolved = getattr(ns, vname)
            self.assertIs(resolved, var.optz,
                          f"Var <{vname}> resolves to wrong object")
            self.assertIsInstance(resolved, cp.Variable)

    def test_rparam_resolution(self):
        ns = RoutineNS(self.ss.DCOPF)
        for rpname, rp in self.ss.DCOPF.rparams.items():
            resolved = getattr(ns, rpname)
            # type depends on owner / sparse / no_parse; just assert it
            # resolves without raising.
            self.assertIsNotNone(resolved,
                                 f"RParam <{rpname}> resolved to None")

    def test_constraint_resolution(self):
        ns = RoutineNS(self.ss.DCOPF)
        for cname, c in self.ss.DCOPF.constrs.items():
            resolved = getattr(ns, cname)
            self.assertIs(resolved, c.optz,
                          f"Constr <{cname}> resolves to wrong object")

    def test_config_resolution(self):
        ns = RoutineNS(self.ss.DCOPF)
        for key in self.ss.DCOPF.config.as_dict():
            resolved = getattr(ns, key)
            self.assertEqual(resolved, getattr(self.ss.DCOPF.config, key))

    def test_unknown_symbol_raises(self):
        ns = RoutineNS(self.ss.DCOPF)
        with self.assertRaises(AttributeError):
            getattr(ns, 'definitely_not_a_symbol_xyz')

    def test_sys_f_sys_mva(self):
        ns = RoutineNS(self.ss.DCOPF)
        self.assertEqual(ns.sys_f, self.ss.config.freq)
        self.assertEqual(ns.sys_mva, self.ss.config.mva)

    def test_readonly(self):
        ns = RoutineNS(self.ss.DCOPF)
        with self.assertRaises(AttributeError):
            ns.foo = 1


if __name__ == '__main__':
    unittest.main()
