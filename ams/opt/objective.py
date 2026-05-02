"""
Module for optimization Objective.
"""
import logging

from typing import Callable, Optional
import re

import numpy as np  # noqa: F401  # used by routine `e_str` evaluation context

import cvxpy as cp

from ams.utils import pretty_long_message
from ams.shared import _prefix, _max_length

from ams.core.routine_ns import RoutineNS
from ams.opt import OptzBase, ensure_symbols, ensure_mats_and_parsed
from ams.opt.optzbase import _EFormDescriptor

logger = logging.getLogger(__name__)


class Objective(OptzBase):
    """
    Base class for objective functions.

    Exactly one of ``e_str`` or ``e_fn`` should be provided. ``e_fn(r)``
    is the Phase 4.1+ form: a callable that takes a :class:`RoutineNS`
    and returns the inner CVXPY expression to be minimized/maximized
    (the ``cp.Minimize``/``cp.Maximize`` wrapping is handled by
    ``Objective.evaluate``).

    Parameters
    ----------
    name : str, optional
        A user-defined name for the objective function.
    e_str : str, optional
        A mathematical expression (legacy form).
    e_fn : callable, optional
        Callable ``e_fn(r) -> cp.Expression`` taking a :class:`RoutineNS`.
    info : str, optional
        Additional informational text about the objective function.
    sense : str, optional
        The sense of the objective function, default to 'min'.
        `min` for minimization and `max` for maximization.

    Attributes
    ----------
    v : NoneType
        The value of the objective function. It needs to be set through
        computation.
    rtn : ams.routines.Routine
        The owner routine instance.
    code : str
        The code string for the objective function.
    """

    e_str = _EFormDescriptor('e_str', 'e_fn')
    e_fn = _EFormDescriptor('e_fn', 'e_str')

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 e_fn: Optional[Callable] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 sense: Optional[str] = 'min'):
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self._e_str = None
        self._e_fn = None
        self.e_str = e_str
        self.e_fn = e_fn
        self.sense = sense
        self.code = None
        self._extra_terms = []

    def add_term(self, fn: Callable):
        """
        Register an extra cost term added to this objective at evaluate time.

        Used by mixin subclasses (e.g. :class:`ESD1Base`, :class:`RTEDVIS`)
        that extend a parent's objective without rewriting it. ``fn(r)``
        receives a :class:`RoutineNS` and returns a scalar CVXPY
        expression; that expression is summed onto the base objective
        before the ``cp.Minimize``/``cp.Maximize`` wrap.

        Composes equally with ``e_str``-form and ``e_fn``-form parents,
        so a parent can migrate from one to the other without breaking
        any subclass that registered terms.
        """
        self._extra_terms.append(fn)

    @property
    def v(self):
        """
        Return the CVXPY objective value.
        """
        if self.optz is None:
            return None
        else:
            return self.optz.value

    @v.setter
    def v(self, value):
        raise AttributeError("Cannot set the value of the objective function.")

    @ensure_symbols
    def parse(self):
        """
        Parse the objective function.

        Returns
        -------
        bool
            Returns True if the parsing is successful, False otherwise.
        """
        if self.sense not in ['min', 'max']:
            raise ValueError(f'Objective sense {self.sense} is not supported.')
        if self.e_fn is not None:
            return True
        # parse the expression str. Note: ``self.code`` stores the *inner*
        # expression only; the ``cp.Minimize``/``cp.Maximize`` wrap is
        # applied in :meth:`evaluate` so extra cost terms can be summed
        # onto the inner before wrapping.
        sub_map = self.om.rtn.syms.sub_map
        code_obj = self.e_str
        for pattern, replacement, in sub_map.items():
            try:
                code_obj = re.sub(pattern, replacement, code_obj)
            except Exception as e:
                raise Exception(f"Error in parsing obj <{self.name}>.\n{e}")
        self.code = code_obj
        msg = f" - Objective <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the objective function.

        Composes ``base_inner + sum(extra_term(r) for ...)`` and wraps
        with ``cp.Minimize`` / ``cp.Maximize`` per ``sense``.

        Returns
        -------
        bool
            Returns True if the evaluation is successful, False otherwise.
        """
        # 1) Compute the base inner expression.
        if self.e_fn is not None:
            try:
                inner = self.e_fn(RoutineNS(self.om.rtn))
            except Exception as e:
                raise Exception(f"Error in evaluating Objective <{self.name}> "
                                f"via e_fn.\n{e}")
        else:
            logger.debug(f" - Objective <{self.name}>: {self.e_str}")
            try:
                local_vars = {'self': self, 'cp': cp}
                inner = eval(self.code, {}, local_vars)
            except Exception as e:
                raise Exception(f"Error in evaluating Objective <{self.name}>.\n{e}")

        # 2) Add registered extra terms (from mixins).
        if self._extra_terms:
            ns = RoutineNS(self.om.rtn)
            for term_fn in self._extra_terms:
                try:
                    inner = inner + term_fn(ns)
                except Exception as e:
                    raise Exception(f"Error composing extra cost term in "
                                    f"Objective <{self.name}>.\n{e}")

        # 3) Wrap with sense.
        wrap = cp.Minimize if self.sense == 'min' else cp.Maximize
        self.optz = wrap(inner)
        return True

    def __repr__(self):
        return f"{self.class_name}: {self.name} [{self.sense.upper()}]"
