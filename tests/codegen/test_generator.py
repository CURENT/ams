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
        # pycode_format_version drives the auto-invalidation of caches
        # whose layout predates the current generator. Its absence
        # would silently re-enable use of stale caches.
        self.assertIn('pycode_format_version = ', src)

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


class TestStaleCacheRegen(unittest.TestCase):
    """``pycode_format_version`` mismatch must trigger a fresh codegen.

    Auto-invalidation is the entire point of bumping the version
    constant when the generated layout changes; without coverage, a
    future change that silently skips the staleness check would only
    surface as production-time mismatches.
    """

    @classmethod
    def setUpClass(cls):
        from ams.prep import _get_pristine_system
        _get_pristine_system()  # warm singleton before redirecting
        cls._tmp = tempfile.TemporaryDirectory(prefix='ams-test-pycode-')
        cls._tmp_path = Path(cls._tmp.name)
        import ams.prep as _prep
        cls._orig_pycode_dir = _prep.pycode_dir
        _prep.pycode_dir = lambda: cls._tmp_path

    @classmethod
    def tearDownClass(cls):
        import ams.prep as _prep
        _prep.pycode_dir = cls._orig_pycode_dir
        cls._tmp.cleanup()

    @staticmethod
    def _force_source_reload(target: Path) -> None:
        """Defeat Python's bytecode cache between in-test rewrites.

        ``write_text`` updates the source mtime, but Python's importer
        caches by ``(size, mtime)`` at second precision; a write
        followed by another write inside the same second leaves the
        ``__pycache__/<name>.pyc`` shadow in place and the next
        ``exec_module`` quietly returns the *old* content. Production
        sees seconds between regen and reload, but tests don't.
        Wipe the shadow and call ``importlib.invalidate_caches`` so
        the next load re-compiles from disk.
        """
        import importlib
        import shutil
        cache_dir = target.parent / '__pycache__'
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        importlib.invalidate_caches()

    def test_stale_format_version_triggers_regen(self):
        """A cache with a wrong pycode_format_version is rejected."""
        from ams.prep.generator import PYCODE_FORMAT_VERSION

        ss = ams.load(ams.get_case(CASE), setup=True,
                      no_output=True, default_config=True)

        # Generate a real cache once so the source layout is correct
        # for everything except the version field.
        ss.DCOPF.init()
        target = self._tmp_path / 'dcopf.py'
        self.assertTrue(target.exists(),
                        "first init should produce the cached pycode")

        # Tamper with the version field — simulate a cache produced by
        # an older AMS whose generated layout differs from today's.
        original = target.read_text()
        stale = PYCODE_FORMAT_VERSION - 1
        tampered = original.replace(
            f"pycode_format_version = {PYCODE_FORMAT_VERSION!r}",
            f"pycode_format_version = {stale!r}",
        )
        self.assertNotEqual(tampered, original,
                            "expected pycode_format_version assignment "
                            "in the generated header")
        target.write_text(tampered)
        self._force_source_reload(target)

        # Re-init a fresh System so _link_pycode runs again. It must
        # detect the stale version, regenerate, and overwrite back to
        # the current version.
        ss2 = ams.load(ams.get_case(CASE), setup=True,
                       no_output=True, default_config=True)
        ss2.DCOPF.init()

        regenerated = target.read_text()
        self.assertIn(
            f"pycode_format_version = {PYCODE_FORMAT_VERSION!r}",
            regenerated,
            "stale pycode_format_version should have triggered a regen"
        )

    def test_absent_format_version_triggers_regen(self):
        """A cache with no pycode_format_version field is rejected."""
        from ams.prep.generator import PYCODE_FORMAT_VERSION

        ss = ams.load(ams.get_case(CASE), setup=True,
                      no_output=True, default_config=True)
        ss.DCOPF.init()
        target = self._tmp_path / 'dcopf.py'

        # Strip the field entirely to simulate a cache from a pre-Step-2
        # AMS that doesn't know about pycode_format_version.
        import re
        original = target.read_text()
        without_field = re.sub(
            r'^pycode_format_version = .*\n', '', original, flags=re.M
        )
        self.assertNotIn('pycode_format_version', without_field,
                         "field should have been stripped for the test")
        target.write_text(without_field)
        self._force_source_reload(target)

        ss2 = ams.load(ams.get_case(CASE), setup=True,
                       no_output=True, default_config=True)
        ss2.DCOPF.init()

        regenerated = target.read_text()
        self.assertIn(
            f"pycode_format_version = {PYCODE_FORMAT_VERSION!r}",
            regenerated,
            "missing pycode_format_version should have triggered a regen"
        )


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
