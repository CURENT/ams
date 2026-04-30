"""
Codegen parity tests for the AMS opt-layer generator (Phase 4.5-B).

For each routine in the test set, the generator's emitted callables
must produce a CVXPY problem whose solution matches the legacy
``e_str``-eval path bit-for-bit (or within solver tolerance for any
non-affine paths).

These tests don't yet hook the generator into ``RoutineBase.__init__``
(that's 4.5-C) — they instead instantiate two systems, run the e_str
path on one, generate-and-wire on the other, and compare. Once
auto-prep lands, these turn into one-system tests.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

import ams
from ams.prep import generate_for_routine


CASE = '5bus/pjm5bus_demo.xlsx'
SOLVER = 'CLARABEL'


def _load_module_from_source(src: str, name: str = 'pycode_under_test'):
    """Compile and import a generated module from a source string."""
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / f'{name}.py'
        target.write_text(src)
        spec = importlib.util.spec_from_file_location(name, target)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _wire_efn_from(routine, mod) -> int:
    """Wire ``e_fn`` from the generated module onto routine items.

    Returns the count of items wired (sanity).
    """
    n = 0
    for name, expr in routine.exprs.items():
        fn = getattr(mod, f'_expr_{name}', None)
        if fn is not None:
            expr.e_fn = fn
            n += 1
    for name, constr in routine.constrs.items():
        fn = getattr(mod, f'_constr_{name}', None)
        if fn is not None:
            constr.e_fn = fn
            n += 1
    if routine.obj is not None:
        fn = getattr(mod, f'_obj_{routine.obj.name}', None)
        if fn is not None:
            routine.obj.e_fn = fn
            n += 1
    return n


class TestGeneratorParityDCOPF(unittest.TestCase):
    """Generated DCOPF callables must produce the same solution as e_str."""

    def setUp(self):
        self.ss_estr = ams.load(ams.get_case(CASE), setup=True,
                                no_output=True, default_config=True)
        self.ss_efn = ams.load(ams.get_case(CASE), setup=True,
                               no_output=True, default_config=True)

    def test_solution_parity(self):
        # Run e_str path
        self.ss_estr.DCOPF.init()
        self.ss_estr.DCOPF.solve(solver=SOLVER)

        # Generate, wire, run e_fn path
        src = generate_for_routine(self.ss_estr.DCOPF)
        mod = _load_module_from_source(src, name='dcopf_gen_parity')
        n_wired = _wire_efn_from(self.ss_efn.DCOPF, mod)
        self.assertGreater(n_wired, 5,
                           "expected several items wired from generated module")

        self.ss_efn.DCOPF.init(force=True)
        self.ss_efn.DCOPF.solve(solver=SOLVER)

        # Compare — solutions should be identical for this LP.
        np.testing.assert_array_almost_equal(
            self.ss_estr.DCOPF.pg.v, self.ss_efn.DCOPF.pg.v, decimal=8,
            err_msg='pg differs between e_str and generated e_fn')
        self.assertAlmostEqual(
            float(self.ss_estr.DCOPF.obj.v),
            float(self.ss_efn.DCOPF.obj.v),
            places=8,
            msg='objective differs')

    def test_dpp_preserved(self):
        src = generate_for_routine(self.ss_estr.DCOPF)
        mod = _load_module_from_source(src, name='dcopf_gen_dpp')
        _wire_efn_from(self.ss_efn.DCOPF, mod)
        self.ss_efn.DCOPF.init(force=True)
        self.assertTrue(self.ss_efn.DCOPF.om.prob.is_dcp(dpp=True),
                        "generated problem must remain DPP-compliant")


class TestGeneratorParityRTED(unittest.TestCase):
    """Same parity check on RTED — exercises the SFR mixin chain."""

    def setUp(self):
        self.ss_estr = ams.load(ams.get_case(CASE), setup=True,
                                no_output=True, default_config=True)
        self.ss_efn = ams.load(ams.get_case(CASE), setup=True,
                               no_output=True, default_config=True)

    def test_solution_parity(self):
        self.ss_estr.RTED.init()
        self.ss_estr.RTED.solve(solver=SOLVER)

        src = generate_for_routine(self.ss_estr.RTED)
        mod = _load_module_from_source(src, name='rted_gen_parity')
        _wire_efn_from(self.ss_efn.RTED, mod)
        self.ss_efn.RTED.init(force=True)
        self.ss_efn.RTED.solve(solver=SOLVER)

        np.testing.assert_array_almost_equal(
            self.ss_estr.RTED.pg.v, self.ss_efn.RTED.pg.v, decimal=8,
            err_msg='pg differs')
        self.assertAlmostEqual(
            float(self.ss_estr.RTED.obj.v),
            float(self.ss_efn.RTED.obj.v),
            places=8)


class TestGeneratorBuildsForMultiPeriod(unittest.TestCase):
    """UC has horizon (5x24), big-M, slicing, MinDur — generator must build it.

    UC requires a MISOCP solver to solve, so we only assert the generated
    problem builds and is DPP-compliant (the e_str path's existing solver
    test already validates the optimization itself).
    """

    def test_uc_builds(self):
        ss = ams.load(ams.get_case(CASE), setup=True,
                      no_output=True, default_config=True)
        src = generate_for_routine(ss.UC)
        mod = _load_module_from_source(src, name='uc_gen_build')
        n = _wire_efn_from(ss.UC, mod)
        self.assertGreater(n, 10, "expected many UC items wired")
        ss.UC.init(force=True)
        self.assertTrue(ss.UC.om.prob.is_dcp(dpp=True))


class TestGeneratorContent(unittest.TestCase):
    """Sanity checks on the generated module text."""

    @classmethod
    def setUpClass(cls):
        cls.ss = ams.load(ams.get_case(CASE), setup=True,
                          no_output=True, default_config=True)

    def test_header_metadata(self):
        src = generate_for_routine(self.ss.DCOPF)
        self.assertIn("class_name = 'DCOPF'", src)
        self.assertIn('md5 = ', src)
        self.assertIn('ams_version = ', src)
        self.assertIn('cvxpy_version = ', src)

    def test_compiles(self):
        for rname in ('DCOPF', 'RTED', 'UC', 'ED'):
            with self.subTest(routine=rname):
                src = generate_for_routine(getattr(self.ss, rname))
                compile(src, f'<generated:{rname}>', 'exec')

    def test_skips_when_e_fn_already_set(self):
        """If a constraint has e_fn pre-set, generator should not emit for it."""
        # Manually set e_fn on one constraint, then regen.
        c = self.ss.DCOPF.constrs['pglb']
        c.e_fn = lambda r: -r.pg + r.pmine <= 0
        try:
            src = generate_for_routine(self.ss.DCOPF)
            self.assertNotIn('def _constr_pglb', src,
                             "generator should skip items with manual e_fn")
        finally:
            c.e_fn = None  # reset (descriptor mutex makes e_str re-active)


if __name__ == '__main__':
    unittest.main()
