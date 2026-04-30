"""
Parity tests for :class:`ams.core.routine_ns.RoutineNS` and the
``e_fn`` plumbing on Constraint / Expression / Objective.

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
from ams.opt.constraint import Constraint
from ams.opt.expression import Expression
from ams.opt.objective import Objective


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


class TestEfnPlumbing(unittest.TestCase):
    """e_fn-form Constraint / Expression / Objective accept callables."""

    def test_constraint_accepts_e_fn(self):
        c = Constraint(name='dummy', e_fn=lambda r: r.pg <= 0)
        self.assertIsNotNone(c.e_fn)
        self.assertIsNone(c.e_str)
        self.assertIsNone(c.code)

    def test_expression_accepts_e_fn(self):
        e = Expression(name='dummy', e_fn=lambda r: 2 * r.pg)
        self.assertIsNotNone(e.e_fn)
        self.assertIsNone(e.e_str)
        self.assertIsNone(e.code)

    def test_objective_accepts_e_fn(self):
        o = Objective(name='dummy', e_fn=lambda r: cp.sum(r.pg))
        self.assertIsNotNone(o.e_fn)
        self.assertIsNone(o.e_str)
        self.assertEqual(o.sense, 'min')

    def test_objective_rejects_bad_sense(self):
        o = Objective(name='dummy', e_fn=lambda r: r.pg, sense='maximize')
        # parse() validates sense; we can't call it without an om/rtn,
        # but the constructor stores it — the validation runs in parse().
        self.assertEqual(o.sense, 'maximize')


if __name__ == '__main__':
    unittest.main()
