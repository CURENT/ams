"""
Codegen tests for the AMS opt-layer generator.

Validates the generator's output: every routine must produce
compilable, DPP-compliant pycode with the expected metadata.

End-to-end correctness (does the codegen-driven solve match what the
user expects) is covered by the broader AMS test suite — auto-prep is
now the default ``OModel.init`` path, so every routine test in
``tests/test_*.py`` exercises the generator implicitly. Comparing a
"manually-wired" run against the auto-prepped run would just be
comparing the codegen against itself.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import ams
from ams.prep import generate_for_routine


CASE = '5bus/pjm5bus_demo.xlsx'


def _load_module_from_source(src: str, name: str = 'pycode_under_test'):
    """Compile and import a generated module from a source string."""
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / f'{name}.py'
        target.write_text(src)
        spec = importlib.util.spec_from_file_location(name, target)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


class TestGeneratorContent(unittest.TestCase):
    """Sanity checks on the generated module text."""

    @classmethod
    def setUpClass(cls):
        # Pristine routines (no init) — items still hold their e_str.
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

    def test_emits_constraint_and_objective(self):
        """Generator must emit at least the canonical DCOPF items."""
        src = generate_for_routine(self.ss.DCOPF)
        # DCOPF has constraints `pb`, `pglb`, `pgub` and an objective.
        self.assertIn('def _constr_pb', src)
        self.assertIn('def _constr_pglb', src)
        self.assertIn('def _obj_obj', src)

    def test_skips_when_e_fn_already_set(self):
        """Generator should skip items with a manually-set e_fn."""
        # Use a fresh system so the global setUpClass instance isn't mutated.
        ss = ams.load(ams.get_case(CASE), setup=True,
                      no_output=True, default_config=True)
        c = ss.DCOPF.constrs['pglb']
        c.e_fn = lambda r: -r.pg + r.pmine <= 0
        src = generate_for_routine(ss.DCOPF)
        self.assertNotIn('def _constr_pglb', src,
                         "generator should skip items with manual e_fn")


class TestGeneratorBuildsDPP(unittest.TestCase):
    """Auto-prep must produce a DPP-compliant problem for every routine."""

    @classmethod
    def setUpClass(cls):
        cls.ss = ams.load(ams.get_case(CASE), setup=True,
                          no_output=True, default_config=True)

    def test_dcopf_dpp(self):
        self.ss.DCOPF.init()
        self.assertTrue(self.ss.DCOPF.om.prob.is_dcp(dpp=True),
                        "DCOPF problem must be DPP-compliant after auto-prep")

    def test_rted_dpp(self):
        self.ss.RTED.init()
        self.assertTrue(self.ss.RTED.om.prob.is_dcp(dpp=True),
                        "RTED problem must be DPP-compliant after auto-prep")

    def test_uc_dpp(self):
        self.ss.UC.init()
        self.assertTrue(self.ss.UC.om.prob.is_dcp(dpp=True),
                        "UC problem must be DPP-compliant after auto-prep")


if __name__ == '__main__':
    unittest.main()
