"""
Module for optimization Constraint.
"""
import logging

from typing import Callable, Optional
import re

import numpy as np

import cvxpy as cp

from ams.utils import pretty_long_message
from ams.shared import _prefix, _max_length

from ams.core.routine_ns import RoutineNS
from ams.opt import OptzBase, ensure_symbols, ensure_mats_and_parsed
from ams.opt.optzbase import _EFormDescriptor

logger = logging.getLogger(__name__)


class Constraint(OptzBase):
    """
    Base class for constraints.

    Exactly one of ``e_str`` or ``e_fn`` should be provided. ``e_fn(r)``
    is the Phase 4.1+ form: a callable that takes a :class:`RoutineNS`
    proxy and returns a CVXPY constraint (e.g., ``r.pg <= r.pmax.cp``).
    ``e_str`` is the legacy regex-rewritten string form, slated for
    deletion in Phase 4.5.

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
        an inequality constraint in the form of `<= 0`. Ignored when ``e_fn`` is used —
        the callable returns the fully-formed constraint.

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
        """
        if self.e_fn is not None:
            return True
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_constr = self.e_str
        for pattern, replacement in sub_map.items():
            try:
                code_constr = re.sub(pattern, replacement, code_constr)
            except TypeError as e:
                raise TypeError(f"Error in parsing constr <{self.name}>.\n{e}")
        # parse the constraint type
        code_constr += " == 0" if self.is_eq else " <= 0"
        # store the parsed expression str code
        self.code = code_constr
        msg = f" - Constr <{self.name}>: {self.e_str}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the constraint.
        """
        if self.e_fn is not None:
            try:
                self.optz = self.e_fn(RoutineNS(self.om.rtn))
            except Exception as e:
                raise Exception(f"Error in evaluating Constraint <{self.name}> "
                                f"via e_fn.\n{e}")
            return True
        msg = f" - Constr <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'cp': cp, 'sub_map': self.om.rtn.syms.val_map}
            self.optz = eval(self.code, {}, local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating Constraint <{self.name}>.\n{e}")

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
            try:
                shape = self._expr.shape
                return np.zeros(shape)
            except AttributeError:
                return None
        else:
            return self.optz._expr.value

    @v.setter
    def v(self, value):
        raise AttributeError("Cannot set the value of the constraint.")
