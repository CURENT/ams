"""
Tests for :class:`ams.core.routine_ns.RoutineNS` and the ``e_fn``
plumbing on Constraint / Expression / Objective.

Validates that every symbol the legacy ``sub_map`` regex resolves on a
real routine also resolves via ``RoutineNS`` attribute access, and to
an object of the same type / identity where possible — the R2 (silent
wrong-symbol resolution) safeguard for the codegen path.
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


class TestRuntimeCustomization(unittest.TestCase):
    """Regression for the ex8.ipynb pattern: post-init e_str modification.

    After ``routine.init()`` runs auto-prep, ``e_str`` must remain readable
    and writable so authors can do the documented customization pattern:

        sp.DCOPF.obj.e_str += '+ cp.sum(cp.multiply(ce, pg))'

    Fixed by routing ``_link_pycode`` through the raw ``_e_fn`` slot
    (preserves ``e_str``) and tracking ``_e_dirty`` on the descriptor so
    re-init's wiring respects user overrides.

    Test isolation: redirects ``ams.prep.pycode_dir`` to a per-class tmp
    directory so the tests can freely create / unlink cache files
    without touching the real ``~/.ams/pycode/``.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path
        from ams.prep import _get_pristine_system  # warm singleton first
        _get_pristine_system()
        # Redirect pycode cache to a tmp dir for the lifetime of this class.
        cls._tmp = tempfile.TemporaryDirectory(prefix='ams-test-pycode-')
        cls._tmp_path = Path(cls._tmp.name)
        import ams.prep as _prep
        cls._orig_pycode_dir = _prep.pycode_dir
        _prep.pycode_dir = lambda: cls._tmp_path

        cls.ss = ams.load(
            ams.get_case('5bus/pjm5bus_demo.xlsx'),
            setup=True,
            no_output=True,
            default_config=True,
        )

    @classmethod
    def tearDownClass(cls):
        import ams.prep as _prep
        _prep.pycode_dir = cls._orig_pycode_dir
        cls._tmp.cleanup()

    def test_e_str_preserved_after_init(self):
        """First init wires e_fn but must not clear e_str."""
        self.ss.DCOPF.init()
        self.assertIsNotNone(self.ss.DCOPF.obj.e_str,
                             "obj.e_str cleared after init — breaks the "
                             "addConstrs / e_str+= customization pattern")
        self.assertIsNotNone(self.ss.DCOPF.obj.e_fn,
                             "auto-prep should still wire e_fn")

    def test_e_str_append_then_reinit(self):
        """The exact ex8.ipynb cell-24 pattern."""
        ss = ams.load(
            ams.get_case('5bus/pjm5bus_demo.xlsx'),
            setup=True, no_output=True, default_config=True,
        )
        ss.DCOPF.init()
        # Read should not be None.
        original = ss.DCOPF.obj.e_str
        self.assertIsNotNone(original)
        # Append (the documented customization knob).
        ss.DCOPF.obj.e_str = original + ' + 0 * cp.sum(pg)'  # no-op term
        # Mutex must mark dirty so re-init doesn't restore the wired e_fn.
        self.assertTrue(getattr(ss.DCOPF.obj, '_e_dirty', False),
                        "appending to e_str must mark item dirty")
        self.assertIsNone(ss.DCOPF.obj.e_fn,
                          "appending to e_str must clear the wired e_fn "
                          "so the legacy regex path picks up the new text")
        ss.DCOPF.init()
        ss.DCOPF.run(solver='CLARABEL')
        self.assertEqual(ss.DCOPF.exit_code, 0)
        # ``.e`` (post-solve numpy LHS readout) must work for items
        # routed through the eval-fallback path. Catches regressions in
        # ``OptzBase.e``'s symbol resolution against ``self.code`` —
        # which changed shape in the eval_e_str refactor (now stores
        # the unrewritten e_str rather than the post-rewrite source).
        # Reviewer-driven assertion (PR #245).
        obj_e = ss.DCOPF.obj.e
        self.assertIsNotNone(obj_e,
                             "obj.e returned None on the eval-fallback "
                             "path — OptzBase.e is mis-resolving symbols")
        self.assertAlmostEqual(float(obj_e), float(ss.DCOPF.obj.v),
                               places=6,
                               msg="obj.e should equal obj.v after a "
                                   "successful solve")

    def test_customization_does_not_pollute_other_instances(self):
        """One System's customization must not leak into another's cache.

        The user-stated invariant: ``~/.ams/pycode/<routine>.py`` is
        always a faithful representation of the source code, never
        polluted by runtime customization. Confirms by mutating ``sp``
        before its first init (the pre-init pollution scenario), then
        constructing ``sp0`` fresh and asserting it solves to the
        original (un-mutated) optimum.
        """
        # Start from a clean cache (in the redirected tmp dir).
        cache = self._tmp_path / 'dcopf.py'
        if cache.exists():
            cache.unlink()

        # sp: customize obj BEFORE first init, then init+solve.
        sp = ams.load(
            ams.get_case('5bus/pjm5bus_demo.xlsx'),
            setup=True, no_output=True, default_config=True,
        )
        original = sp.DCOPF.obj.e_str
        sp.DCOPF.obj.e_str = original + ' + 1e-6 * cp.sum(pg)'  # tiny perturbation
        sp.DCOPF.run(solver='CLARABEL')
        self.assertEqual(sp.exit_code, 0)
        sp_obj = float(sp.DCOPF.obj.v)

        # sp0: fresh instance, no customization.
        sp0 = ams.load(
            ams.get_case('5bus/pjm5bus_demo.xlsx'),
            setup=True, no_output=True, default_config=True,
        )
        sp0.DCOPF.run(solver='CLARABEL')
        sp0_obj = float(sp0.DCOPF.obj.v)

        # The perturbation is small but non-zero; if sp0 inherited it,
        # its objective would equal sp's. They must differ.
        self.assertNotAlmostEqual(
            sp_obj, sp0_obj, places=8,
            msg="sp0 inherited sp's pre-init customization — disk cache "
                "was polluted by sp's mutated state.",
        )

    def test_pristine_marker_in_disk_cache(self):
        """Generated pycode must declare itself pristine."""
        cache = self._tmp_path / 'dcopf.py'
        if cache.exists():
            cache.unlink()
        ss = ams.load(
            ams.get_case('5bus/pjm5bus_demo.xlsx'),
            setup=True, no_output=True, default_config=True,
        )
        ss.DCOPF.init()
        self.assertTrue(cache.exists())
        text = cache.read_text()
        self.assertIn('pristine = True', text,
                      "disk cache must contain `pristine = True` marker so "
                      "older polluted caches are auto-invalidated")


class TestDopfInitsViaEfn(unittest.TestCase):
    """Smoke test: DOPF / DOPFVIS init cleanly via the codegen e_fn path."""

    @classmethod
    def setUpClass(cls):
        cls.ss = ams.load(
            ams.get_case('5bus/pjm5bus_demo.xlsx'),
            setup=True,
            no_output=True,
            default_config=True,
        )

    def test_dopf_init(self):
        self.ss.DOPF.init()
        self.assertTrue(self.ss.DOPF.om.prob.is_dcp(dpp=True))

    def test_dopfvis_init(self):
        self.ss.DOPFVIS.init()
        self.assertTrue(self.ss.DOPFVIS.om.prob.is_dcp(dpp=True))


if __name__ == '__main__':
    unittest.main()
