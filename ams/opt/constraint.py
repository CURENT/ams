"""
Module for optimization Constraint.
"""
import logging

from typing import Callable, Optional

import numpy as np

import cvxpy as cp

from ams.utils import pretty_long_message
from ams.shared import _prefix, _max_length

from ams.core.routine_ns import RoutineNS
from ams.opt import OptzBase, ensure_symbols, ensure_mats_and_parsed
from ams.opt.optzbase import _EFormDescriptor
from ams.opt._runtime_eval import eval_e_str, assert_constraint_lhs_zero

logger = logging.getLogger(__name__)


class Constraint(OptzBase):
    """
    Base class for constraints.

    Routines are authored with ``e_str`` strings; the codegen at
    :func:`ams.prep.generate_for_routine` compiles each ``e_str`` into
    a callable ``e_fn(r)`` taking a :class:`RoutineNS` proxy and
    returning a CVXPY constraint. The mutex descriptor on
    ``e_str`` / ``e_fn`` keeps the two forms exclusive. Authors who
    need to bypass the DSL may pass ``e_fn=`` directly; codegen leaves
    a manually-set ``e_fn`` alone.

    Parameters
    ----------
    name : str, optional
        A user-defined name for the constraint.
    e_str : str, optional
        Mathematical expression in canonical CVXPY syntax with the
        relational operator embedded — ``'<LHS> <= 0'``,
        ``'<LHS> == 0'``, or ``'<LHS> >= 0'``. Authoring style is
        LHS-zero: keep all terms on the left so ``.v`` reports
        slack-from-zero (negative = respected, positive = violated)
        uniformly across every constraint. CVXPY canonicalizes
        every inequality to ``lhs - rhs <= 0`` internally regardless
        of operator direction. Strict ``<`` / ``>`` are forbidden by
        CVXPY (raises ``NotImplementedError``).
    e_fn : callable, optional
        Callable ``e_fn(r) -> cp.Constraint`` taking a
        :class:`RoutineNS`. Must return a fully-formed
        ``cp.constraints.Constraint`` (the codegen convention of
        returning a bare LHS expression is no longer supported —
        the routine source is now the single source of truth for
        the relational shape).
    info : str, optional
        Additional informational text about the constraint.

    Attributes
    ----------
    is_disabled : bool
        Flag indicating if the constraint is disabled, False by default.
    rtn : ams.routines.Routine
        The owner routine instance.
    code : str, optional
        The code string for the constraint
    """

    e_str = _EFormDescriptor('e_str', 'e_fn')
    e_fn = _EFormDescriptor('e_fn', 'e_str')

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 e_fn: Optional[Callable] = None,
                 info: Optional[str] = None,
                 ):
        OptzBase.__init__(self, name=name, info=info)
        self._e_str = None
        self._e_fn = None
        self.e_str = e_str
        self.e_fn = e_fn
        self.is_disabled = False
        self.code = None

    def get_idx(self):
        raise NotImplementedError

    def get_all_idxes(self):
        raise NotImplementedError

    @ensure_symbols
    def parse(self):
        """
        Parse the constraint.

        Validates the LHS-zero authoring shape on every constraint
        carrying an ``e_str`` (covers both the codegen path —
        ``e_str`` is preserved on the item — and the eval-fallback
        path). See :func:`assert_constraint_lhs_zero` for why.

        Codegen-wired ``e_fn`` items then short-circuit; nothing else
        to parse here. For the legacy ``e_str`` path, store the
        source verbatim — symbol rewriting + eval happen together in
        :func:`ams.opt._runtime_eval.eval_e_str` at evaluate time.
        ``self.code`` is preserved (as the unrewritten source) so
        :pyattr:`OptzBase.e` and :meth:`Routine.formulation_summary`
        still have something to read.
        """
        if self.e_str is not None:
            assert_constraint_lhs_zero(self, self.e_str)
        if self.e_fn is not None:
            return True
        self.code = self.e_str
        msg = f" - Constr <{self.name}>: {self.e_str}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the constraint.

        Both the ``e_fn`` (codegen) and ``e_str`` (eval-fallback)
        paths must return a fully-formed ``cp.constraints.Constraint`` —
        the relational operator lives in the source ``e_str`` (or in
        the body of an author-supplied ``e_fn``), not in any
        out-of-band flag. ``Constraint.v`` then reads
        ``self.optz._expr.value`` which CVXPY canonicalizes to the
        slack-from-zero LHS regardless of whether the author wrote
        ``<=`` or ``>=``.
        """
        if self.e_fn is not None:
            try:
                result = self.e_fn(RoutineNS(self.om.rtn))
            except Exception as e:
                raise Exception(f"Error in evaluating Constraint <{self.name}> "
                                f"via e_fn.\n{e}")
        else:
            msg = f" - Constr <{self.name}>: {self.e_str}"
            logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
            result = eval_e_str(self, self.e_str)
        if not isinstance(result, cp.constraints.Constraint):
            source = f"e_str={self.e_str!r}" if self.e_str is not None else "e_fn"
            raise TypeError(
                f"Constraint <{self.name}>: {source} must produce a "
                f"cp.constraints.Constraint (embed the relational "
                f"operator: '<LHS> <= 0', '<LHS> == 0', or '<LHS> >= 0'). "
                f"Got {type(result).__name__}. Stale "
                f"~/.ams/pycode/ from before the v1.2.2 is_eq retirement "
                f"is auto-invalidated by PYCODE_FORMAT_VERSION; if you see "
                f"this from a freshly-regenerated cache, the underlying "
                f"e_str is missing its trailing operator."
            )
        self.optz = result
        return True

    def __repr__(self):
        enabled = 'OFF' if self.is_disabled else 'ON'
        out = f"{self.class_name}: {self.name} [{enabled}]"
        return out

    @property
    def v(self):
        """
        Return the CVXPY constraint LHS value.
        """
        if self.optz is None:
            return None
        if self.optz._expr.value is None:
            # Solver hasn't run / didn't converge; return a zeros array
            # of the right shape rather than ``None`` so callers doing
            # post-solve diagnostics get an array of the expected size.
            try:
                return np.zeros(self.optz._expr.shape)
            except AttributeError:
                return None
        return self.optz._expr.value

    @v.setter
    def v(self, value):
        raise AttributeError("Cannot set the value of the constraint.")
