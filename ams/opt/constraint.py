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
from ams.opt._runtime_eval import eval_e_str

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
        A mathematical expression representing the constraint (legacy form).
    e_fn : callable, optional
        Callable ``e_fn(r) -> cp.Constraint`` taking a :class:`RoutineNS`.
    info : str, optional
        Additional informational text about the constraint.
    is_eq : str, optional
        Flag indicating if the constraint is an equality constraint. False indicates
        an inequality constraint in the form of ``<= 0``. Honored for both the
        ``e_str`` form and the codegen ``e_fn`` form (which returns only the LHS
        — :meth:`evaluate` then applies ``== 0`` or ``<= 0`` based on this flag).
        It is *only* ignored when an author manually passes an ``e_fn`` that
        returns a fully-formed ``cp.Constraint``.

    Attributes
    ----------
    is_disabled : bool
        Flag indicating if the constraint is disabled, False by default.
    rtn : ams.routines.Routine
        The owner routine instance.
    is_disabled : bool, optional
        Flag indicating if the constraint is disabled, False by default.
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
                 is_eq: Optional[bool] = False,
                 ):
        OptzBase.__init__(self, name=name, info=info)
        self._e_str = None
        self._e_fn = None
        self.e_str = e_str
        self.e_fn = e_fn
        self.is_eq = is_eq
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

        Codegen-wired ``e_fn`` items have nothing to parse. For the
        legacy ``e_str`` path, store the source verbatim — symbol
        rewriting + eval happen together in
        :func:`ams.opt._runtime_eval.eval_e_str` at evaluate time.
        ``self.code`` is preserved (as the unrewritten source) so
        :pyattr:`OptzBase.e` and :meth:`Routine.formulation_summary`
        still have something to read.
        """
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

        ``e_fn(r)`` may return either a fully-formed ``cp.Constraint``
        (legacy convention; ``r.pg <= 0``) or just the LHS expression
        (codegen convention; ``r.pg``). When it returns the LHS, the
        relational op is applied here based on ``is_eq``. The LHS form
        is required for ``.e`` to recover a numpy LHS during a failed
        solve — see ``OptzBase.e``. The ``e_str`` path goes through
        :func:`eval_e_str` and is wrapped the same way.
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
        if isinstance(result, cp.constraints.Constraint):
            self.optz = result
        else:
            self.optz = (result == 0) if self.is_eq else (result <= 0)
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
