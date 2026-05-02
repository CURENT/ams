"""
Runtime ``e_str`` evaluation helpers.

Used by :class:`Constraint`, :class:`Objective`, :class:`Expression`,
and :class:`ExpressionCalc` whenever the author/user supplies an
``e_str`` string instead of (or to override) the codegen-wired
``e_fn``. Replaces the four near-duplicate ``regex + eval`` blocks
that previously lived in those classes — see step 3 of the
cvxpy-namespace passthrough project.

Two helpers, both symbol-resolution-only (no function-name rewrites
— callers must author canonical CVXPY syntax since the
function-rewrite layer was deleted in PR #244):

- :func:`eval_e_str` — CVXPY-side; produces ``cp.Expression`` /
  ``cp.Constraint`` for the optimization model.
- :func:`eval_e_str_numeric` — numeric-side; used by
  :pyattr:`OptzBase.e` to recover the post-solve numpy LHS for
  customized items.
"""

import logging
import re

import cvxpy as cp
import numpy as np

from ams.shared import sps
from ams.core.routine_ns import RoutineNS, NumericRoutineNS
from ams.prep.generator import _build_symbol_regex, _collect_symbol_names

logger = logging.getLogger(__name__)


def _get_symbol_regex(rtn):
    """Memoized symbol-resolution regex for ``rtn``.

    Cached on ``rtn.syms`` so the per-evaluate cost of
    :func:`_collect_symbol_names` (sort + reserved-name set
    intersection) is paid once per symbol-generation pass, not per
    item. ``SymProcessor.generate_symbols`` clears the cache when
    symbols are regenerated.
    """
    syms = rtn.syms
    cached = getattr(syms, '_eval_symbol_regex', None)
    if cached is not None:
        return cached
    regex = _build_symbol_regex(_collect_symbol_names(rtn))
    syms._eval_symbol_regex = regex
    return regex


def _resolve(rtn, e_str):
    """Apply the symbol-resolution regex to ``e_str``."""
    sym_re = _get_symbol_regex(rtn)
    return sym_re.sub(r'r.\1', e_str) if sym_re is not None else e_str


def eval_e_str(item, e_str):
    """Resolve symbols in ``e_str`` and ``eval`` against CVXPY namespace.

    Parameters
    ----------
    item : ams.opt.OptzBase
        The opt-layer item (Constraint/Objective/Expression/
        ExpressionCalc) whose ``e_str`` is being evaluated. Used both
        to locate the owner routine (``item.om.rtn``) and to expose
        ``self`` in the eval namespace for back-compat with any
        author customization that reaches for it.
    e_str : str
        The expression string. Must be in canonical CVXPY syntax
        (``cp.sum(...)``, ``cp.multiply(...)``, ``*`` for element-wise
        multiplication). Bare ``sum``/``multiply``/``mul``/``dot``
        no longer resolve — see the v1.3 release-notes migration.

    Returns
    -------
    object
        The evaluated CVXPY object (``cp.Expression``,
        ``cp.Constraint``, or numeric for literal-only inputs). The
        caller is responsible for any wrapping (``cp.Minimize`` for
        Objective, ``== 0`` / ``<= 0`` for Constraint when the result
        isn't already a ``cp.constraints.Constraint``).
    """
    rtn = item.om.rtn
    code = _resolve(rtn, e_str)
    local_vars = {
        'cp': cp, 'np': np, 'sps': sps,
        'r': RoutineNS(rtn), 'self': item,
    }
    # Trust boundary: e_str is Python source eval'd in a curated
    # namespace (cp/np/sps + the routine's symbol proxy). Authors come
    # from three places — repo source under ``ams/routines/``, mixin
    # overrides, and downstream user code (notebook ``addConstrs``,
    # ``item.e_str = ...`` assignments after cvxpy-namespace passthrough
    # opened the full ``cp.*`` surface). All three ride the same path;
    # treat them with the same trust as the codegen-emitted pycode —
    # both run user-accessible source through Python. No untrusted-
    # input ingestion contract here, so authors are responsible for
    # what their ``e_str`` evaluates to.
    try:
        result = eval(code, {}, local_vars)  # nosec B307
    except Exception as exc:
        raise Exception(
            f"Error evaluating {item.class_name} <{item.name}> via e_str.\n"
            f"  e_str: {e_str}\n"
            f"  rewritten: {code}\n"
            f"  error: {exc}"
        ) from exc
    # Per-item DCP visibility for the eval-fallback path. The codegen
    # path skips eval entirely, so non-DCP drift in built-in routines
    # surfaces at the problem level via ``OModel._log_dpp_diagnostic``.
    # User customizations / addConstrs / mixin overrides are the only
    # things that hit eval_e_str at runtime, and they're exactly where
    # silent DCP drift is most likely (e.g. ``cp.multiply(pg, pg)``
    # instead of ``cp.square(pg)``). Log a warning here so the
    # author sees the offending item by name, instead of a generic
    # solve-time CVXPY DCPError that points at the whole problem.
    is_dcp = getattr(result, 'is_dcp', None)
    if callable(is_dcp):
        try:
            ok = is_dcp()
        except Exception:  # noqa: BLE001 — DCP introspection must not block evaluation
            ok = True
        if not ok:
            logger.warning(
                f"{item.class_name} <{item.name}> e_str produced a "
                f"non-DCP CVXPY object — solver will likely raise.\n"
                f"  e_str: {e_str}"
            )
    return result


_TRAILING_OP_RE = re.compile(r'\s*(==|<=|>=)\s*0\s*$')


def assert_constraint_lhs_zero(item, e_str):
    """Validate constraint ``e_str`` ends with ``<= 0`` / ``== 0`` / ``>= 0``.

    The LHS-zero shape is the only authoring shape after the
    ``is_eq`` retirement (PR #249) — every term goes on the LHS so
    :pyattr:`OptzBase.e` can recover the slack-from-zero LHS by
    stripping the trailing operator. Without the convention,
    ``pg <= pmax`` survives ``Constraint.evaluate`` (CVXPY accepts
    it), but :pyattr:`OptzBase.e` returns a numpy bool array
    instead of the LHS slack — silently wrong post-solve diagnostics.

    Raises
    ------
    ValueError
        If ``e_str`` does not end with the required relational
        operator + ``0`` form.
    """
    if _TRAILING_OP_RE.search(e_str) is None:
        raise ValueError(
            f"Constraint <{getattr(item, 'name', '?')}>: e_str must end "
            f"with `<= 0`, `== 0`, or `>= 0` (LHS-zero authoring shape). "
            f"Got: {e_str!r}. Move every term to the LHS — "
            f"e.g. `pg - pmax <= 0` instead of `pg <= pmax`. "
            f"Without the trailing-zero shape, `OptzBase.e` returns a "
            f"numpy bool array instead of the LHS slack."
        )


def eval_e_str_numeric(item, e_str):
    """Evaluate ``e_str`` numerically for :pyattr:`OptzBase.e`.

    Same shape as :func:`eval_e_str`, but resolves symbols against a
    :class:`NumericRoutineNS` that returns numpy values for Vars,
    RParams, services, and recomputes Expressions through their
    ``e_fn``. CVXPY atoms (``cp.sum``, ``cp.multiply``, …) wrap
    numpy inputs in ``cp.Constant``, so the returned object's
    ``.value`` is the numpy LHS — equivalent to what the legacy
    ``val_map`` + ``np.<atom>`` rewrite produced, but driven by the
    same name-resolution surface as the CVXPY-side helper.

    For an ``e_str`` carrying an embedded relational operator
    (``... <= 0`` / ``... == 0`` / ``... >= 0``), strip the trailing
    operator before evaluating so the returned object is the LHS
    slack — what ``Constraint.v`` reports via CVXPY canonicalization
    on the symbolic side. Without the strip, a numpy-only eval
    short-circuits to a bool array.
    """
    rtn = item.om.rtn
    body = _TRAILING_OP_RE.sub('', e_str)
    code = _resolve(rtn, body)
    local_vars = {
        'cp': cp, 'np': np, 'sps': sps,
        'r': NumericRoutineNS(rtn), 'self': item,
    }
    try:
        result = eval(code, {}, local_vars)  # nosec B307
    except Exception as exc:
        raise Exception(
            f"Error numerically evaluating {item.class_name} "
            f"<{item.name}> via e_str.\n"
            f"  e_str: {e_str}\n"
            f"  rewritten: {code}\n"
            f"  error: {exc}"
        ) from exc
    if isinstance(result, cp.constraints.Constraint):
        inner = getattr(result, '_expr', None)
        if inner is None:
            inner = result.args[0] if result.args else None
        return inner
    return result
