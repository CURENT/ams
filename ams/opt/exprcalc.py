"""
Module for optimization ExpressionCalc.
"""
import logging

from typing import Optional
import re

import numpy as np

from ams.shared import sps  # NOQA

import cvxpy as cp

from ams.utils import pretty_long_message
from ams.shared import _prefix, _max_length

from ams.opt import OptzBase, ensure_symbols, ensure_mats_and_parsed

logger = logging.getLogger(__name__)


class ExpressionCalc(OptzBase):
    """
    Class for calculating expressions.

    Note that `ExpressionCalc` is not a CVXPY expression, and it should not be used
    in the optimization model.
    It is used to calculate expression values **after** successful optimization.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 e_str: Optional[str] = None,
                 model: Optional[str] = None,
                 src: Optional[str] = None,
                 ):
        OptzBase.__init__(self, name=name, info=info, unit=unit, model=model)
        self.optz = None
        self.e_str = e_str
        self.code = None
        self.src = src

    @ensure_symbols
    def parse(self):
        """
        Parse the Expression.
        """
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
        msg = f" - Expression <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'np': np, 'cp': cp, 'sps': sps}
            self.optz = self._evaluate_expression(self.code, local_vars=local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating ExpressionCalc <{self.name}>.\n{e}")
        return True

    def _evaluate_expression(self, code, local_vars=None):
        """
        Helper method to evaluate the expression code.

        Parameters
        ----------
        code : str
            The code string representing the expression.

        Returns
        -------
        cp.Expression
            The evaluated cvxpy expression.
        """
        return eval(code, {}, local_vars)

    @property
    def v(self):
        """
        Return the CVXPY expression value.
        """
        if self.optz is None:
            return None
        else:
            return self.optz.value

    @property
    def e(self):
        """
        Return the calculated expression value.
        """
        if self.code is None:
            logger.info(f"ExpressionCalc <{self.name}> is not parsed yet.")
            return None

        val_map = self.om.rtn.syms.val_map
        code = self.code
        for pattern, replacement in val_map.items():
            try:
                code = re.sub(pattern, replacement, code)
            except TypeError as e:
                raise TypeError(e)

        try:
            logger.debug(pretty_long_message(f"Value code: {code}",
                                             _prefix, max_length=_max_length))
            local_vars = {'self': self, 'np': np, 'cp': cp, 'val_map': val_map}
            return self._evaluate_expression(code, local_vars)
        except Exception as e:
            logger.error(f"Error in calculating expr <{self.name}>.\n{e}")
            return None
