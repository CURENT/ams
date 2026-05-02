"""
Runtime ``e_str`` evaluation helper.

Used by :class:`Constraint`, :class:`Objective`, :class:`Expression`,
and :class:`ExpressionCalc` whenever the author/user supplies an
``e_str`` string instead of (or to override) the codegen-wired
``e_fn``. Replaces the four near-duplicate ``regex + eval`` blocks
that previously lived in those classes — see step 3 of the
cvxpy-namespace passthrough project.

The helper does **only** symbol resolution
(``\\b<name>\\b → r.<name>``), not function-name rewriting: callers
are expected to author canonical CVXPY syntax (``cp.sum(...)``,
``cp.multiply(...)``) since the function-rewrite layer was deleted
in PR #244.
"""

import logging

import cvxpy as cp
import numpy as np

from ams.shared import sps
from ams.core.routine_ns import RoutineNS
from ams.prep.generator import _build_symbol_regex, _collect_symbol_names

logger = logging.getLogger(__name__)


def eval_e_str(item, e_str):
    """Resolve symbols in ``e_str`` and ``eval`` the result.

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
    sym_re = _build_symbol_regex(_collect_symbol_names(rtn))
    code = sym_re.sub(r'r.\1', e_str) if sym_re is not None else e_str
    local_vars = {
        'cp': cp, 'np': np, 'sps': sps,
        'r': RoutineNS(rtn), 'self': item,
    }
    # Trust boundary: e_str is repo-authored in ams/routines/ or
    # comes from trusted downstream code (notebook authors, mixins).
    # Same boundary as the codegen-emitted pycode — both run user-
    # accessible source through Python. No untrusted-input ingestion.
    try:
        return eval(code, {}, local_vars)  # nosec B307
    except Exception as exc:
        raise Exception(
            f"Error evaluating {item.class_name} <{item.name}> via e_str.\n"
            f"  e_str: {e_str}\n"
            f"  rewritten: {code}\n"
            f"  error: {exc}"
        ) from exc
