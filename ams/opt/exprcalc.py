"""
Module for optimization ExpressionCalc.
"""
import logging

from typing import Callable, Optional
import re

import numpy as np

from ams.shared import sps  # NOQA

import cvxpy as cp

from ams.utils import pretty_long_message
from ams.shared import _prefix, _max_length

from ams.opt import OptzBase, ensure_symbols, ensure_mats_and_parsed
from ams.opt.optzbase import _EFormDescriptor

logger = logging.getLogger(__name__)


class ExpressionCalc(OptzBase):
    """
    Class for calculating expressions.

    This class is useful for performing post-solve calculations, but it does not
    participate in the optimization model itself.

    Note: `ExpressionCalc` is not a CVXPY expression and should NOT be referenced
    in `e_str` by any other components, including other instances of `ExpressionCalc`.

    Accepts ``e_fn`` (callable taking :class:`RoutineNS`) alongside the
    legacy ``e_str`` form. The codegen emits ``_exprcalc_<name>``
    callables; the descriptor mutex from ``OptzBase`` keeps the two
    forms exclusive.
    """

    e_str = _EFormDescriptor('e_str', 'e_fn')
    e_fn = _EFormDescriptor('e_fn', 'e_str')

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 e_str: Optional[str] = None,
                 e_fn: Optional[Callable] = None,
                 model: Optional[str] = None,
                 src: Optional[str] = None,
                 ):
        OptzBase.__init__(self, name=name, info=info, unit=unit, model=model)
        self.optz = None
        self._e_str = None
        self._e_fn = None
        self.e_str = e_str
        self.e_fn = e_fn
        self.code = None
        self.src = src

    @ensure_symbols
    def parse(self):
        """
        Parse the Expression.
        """
        if self.e_fn is not None:
            return True
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_expr = self.e_str
        for pattern, replacement in sub_map.items():
            try:
                code_expr = re.sub(pattern, replacement, code_expr)
            except Exception as e:
                raise Exception(f"Error in parsing expr <{self.name}>.\n{e}")
        # store the parsed expression str code
        self.code = code_expr
        msg = f" - ExpressionCalc <{self.name}>: {self.e_str}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the expression.
        """
        if self.e_fn is not None:
            from ams.core.routine_ns import RoutineNS  # local: avoid circular import
            try:
                self.optz = self.e_fn(RoutineNS(self.om.rtn))
            except Exception as e:
                raise Exception(f"Error in evaluating ExpressionCalc <{self.name}> "
                                f"via e_fn.\n{e}")
            return True
        msg = f" - Expression <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'np': np, 'cp': cp, 'sps': sps}
            self.optz = eval(self.code, {}, local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating ExpressionCalc <{self.name}>.\n{e}")
        return True

    @property
    def v(self):
        """
        Return the CVXPY expression value.
        """
        if self.optz is None:
            return None
        else:
            return self.optz.value

    @v.setter
    def v(self, value):
        """
        Set the ExpressionCalc value.
        """
        if self.optz is None:
            raise ValueError("ExpressionCalc is not evaluated yet.")
        if not isinstance(value, (int, float, np.ndarray)):
            raise TypeError(f"Value must be a number or numpy array, got {type(value)}.")
        self.optz.value = value
