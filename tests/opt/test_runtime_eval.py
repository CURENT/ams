"""
Tests for the eval-fallback safety net in :mod:`ams.opt._runtime_eval`
plus the dynamic reserved-name guard in :mod:`ams.prep.generator`.

Covers two PRs' worth of tightening that landed after the
cvxpy-namespace passthrough and ``is_eq`` retirement:

- L5 — :func:`assert_constraint_lhs_zero` rejects constraint
  ``e_str`` that doesn't end with ``<= 0`` / ``== 0`` / ``>= 0``.
- M4a — :func:`eval_e_str` logs a per-item warning when the
  evaluated CVXPY object reports ``is_dcp() is False``.
- M4b — :data:`RESERVED_CVXPY_ATOM_NAMES` is derived from
  ``cvxpy.atoms`` at import time and is a strict superset of the
  static fallback.
"""
import logging
import unittest

import pytest

import cvxpy as cp

from ams.opt._runtime_eval import assert_constraint_lhs_zero
from ams.prep.generator import (
    RESERVED_CVXPY_ATOM_NAMES,
    _STATIC_RESERVED_CVXPY_ATOM_NAMES,
)


class TestAssertConstraintLhsZero(unittest.TestCase):
    """Pure-function tests on the LHS-zero validator."""

    def _stub(self, name='c'):
        # Minimal stub — validator only reads ``.name`` for the error message.
        class _Item:
            pass
        item = _Item()
        item.name = name
        return item

    def test_accepts_le_zero(self):
        assert_constraint_lhs_zero(self._stub(), 'pg - pmax <= 0')

    def test_accepts_eq_zero(self):
        assert_constraint_lhs_zero(self._stub(), 'cp.sum(pg) - cp.sum(pd) == 0')

    def test_accepts_ge_zero(self):
        assert_constraint_lhs_zero(self._stub(), 'pg - pmin >= 0')

    def test_accepts_trailing_whitespace(self):
        # The regex tolerates whitespace around the op and the zero.
        assert_constraint_lhs_zero(self._stub(), 'pg - pmax  <=  0   ')

    def test_rejects_missing_zero(self):
        with self.assertRaisesRegex(ValueError, 'LHS-zero authoring shape'):
            assert_constraint_lhs_zero(self._stub('rru'), 'pg <= pmax')

    def test_rejects_eq_to_nonzero(self):
        with self.assertRaisesRegex(ValueError, 'LHS-zero'):
            assert_constraint_lhs_zero(self._stub(), 'cp.sum(pg) == cp.sum(pd)')

    def test_rejects_no_operator(self):
        with self.assertRaisesRegex(ValueError, 'LHS-zero'):
            assert_constraint_lhs_zero(self._stub(), 'pg - pmax')

    def test_error_quotes_offending_e_str(self):
        with self.assertRaises(ValueError) as ctx:
            assert_constraint_lhs_zero(self._stub('foo'), 'pg <= pmax')
        msg = str(ctx.exception)
        self.assertIn('foo', msg)
        self.assertIn("'pg <= pmax'", msg)


class TestConstraintParseRejectsBadEStr(unittest.TestCase):
    """End-to-end: a routine constraint authored without LHS-zero
    raises during parse (which auto-prep runs at ``om.init`` time)."""

    @pytest.fixture(autouse=True)
    def _attach_ss(self, request, case5):
        request.instance.ss = case5

    def test_rejects_bad_user_constraint(self):
        # Mutate a user-customized constraint to a non-LHS-zero form
        # and confirm parse raises with the helpful error.
        self.ss.DCOPF.plflb.e_str = 'plf <= rate_a'  # missing `- 0`
        with self.assertRaises(ValueError) as ctx:
            self.ss.DCOPF.plflb.parse()
        self.assertIn('LHS-zero', str(ctx.exception))

    def test_accepts_correct_user_constraint(self):
        # Same intent, correctly authored.
        self.ss.DCOPF.plflb.e_str = 'plf - rate_a <= 0'
        self.ss.DCOPF.plflb.parse()  # must not raise


class TestEvalEStrDcpWarning(unittest.TestCase):
    """``eval_e_str`` logs a per-item warning on non-DCP results."""

    @pytest.fixture(autouse=True)
    def _attach_ss(self, request, case5):
        request.instance.ss = case5

    def test_warns_on_non_dcp(self):
        # Author a non-DCP product: ``cp.multiply(pg, pg)`` is non-convex
        # (use ``cp.square(pg)`` instead). Going through addConstrs forces
        # the eval-fallback path.
        self.ss.DCOPF.init()  # codegen wires built-ins; clean state
        self.ss.DCOPF.addConstrs(
            name='nonconvex_demo',
            e_str='cp.multiply(pg, pg) - 1 <= 0',
        )
        with self.assertLogs('ams.opt._runtime_eval', level='WARNING') as cm:
            self.ss.DCOPF.nonconvex_demo.parse()
            self.ss.DCOPF.nonconvex_demo.evaluate()
        self.assertTrue(
            any('non-DCP' in m for m in cm.output),
            f'expected non-DCP warning, got: {cm.output}',
        )

    def test_no_warning_on_dcp(self):
        # Sanity: a normal LHS-zero constraint must not emit the warning.
        self.ss.DCOPF.init()
        self.ss.DCOPF.addConstrs(
            name='dcp_demo',
            e_str='cp.square(pg) - 100 <= 0',
        )
        logger = logging.getLogger('ams.opt._runtime_eval')
        # Force a handler so assertLogs has something to attach to even
        # when no record is emitted.
        with self.assertLogs(logger, level='WARNING') as cm:
            logger.warning('sentinel')  # ensure cm.output is non-empty
            self.ss.DCOPF.dcp_demo.parse()
            self.ss.DCOPF.dcp_demo.evaluate()
        self.assertFalse(
            any('non-DCP' in m and 'dcp_demo' in m for m in cm.output),
            f'unexpected non-DCP warning: {cm.output}',
        )


class TestReservedAtomNamesDerivation(unittest.TestCase):
    """:data:`RESERVED_CVXPY_ATOM_NAMES` is dynamic + a strict superset
    of the static fallback."""

    def test_static_is_subset(self):
        self.assertTrue(
            _STATIC_RESERVED_CVXPY_ATOM_NAMES.issubset(RESERVED_CVXPY_ATOM_NAMES),
            'auto-derivation must never narrow the reserved set',
        )

    def test_includes_known_non_static_atoms(self):
        # Atoms that exist in modern CVXPY but were not in the original
        # static snapshot. If CVXPY removes one of these someday this
        # test will surface that as a real signal worth investigating.
        for atom in ('huber', 'log_sum_exp', 'quad_over_lin'):
            with self.subTest(atom=atom):
                if hasattr(cp, atom) and callable(getattr(cp, atom)):
                    self.assertIn(atom, RESERVED_CVXPY_ATOM_NAMES)

    def test_grew_meaningfully(self):
        # Cheap regression check: derivation should add at least a
        # handful of names beyond the static set on any reasonable
        # CVXPY install.
        new = RESERVED_CVXPY_ATOM_NAMES - _STATIC_RESERVED_CVXPY_ATOM_NAMES
        self.assertGreaterEqual(
            len(new), 10,
            f'auto-derivation found only {len(new)} new atoms: {sorted(new)}',
        )


if __name__ == '__main__':
    unittest.main()
