"""
Module for optimization Constraint.
"""
import logging

from typing import Optional
import re

import numpy as np

import cvxpy as cp

from ams.utils import pretty_long_message
from ams.shared import _prefix, _max_length

from ams.opt import OptzBase, ensure_symbols, ensure_mats_and_parsed

logger = logging.getLogger(__name__)


class Constraint(OptzBase):
    """
    Base class for constraints.

    This class is used as a template for defining constraints. Each
    instance of this class represents a single constraint.

    Parameters
    ----------
    name : str, optional
        A user-defined name for the constraint.
    e_str : str, optional
        A mathematical expression representing the constraint.
    info : str, optional
        Additional informational text about the constraint.
    is_eq : str, optional
        Flag indicating if the constraint is an equality constraint. False indicates
        an inequality constraint in the form of `<= 0`.

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

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 info: Optional[str] = None,
                 is_eq: Optional[bool] = False,
                 ):
        OptzBase.__init__(self, name=name, info=info)
        self.e_str = e_str
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

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the constraint.
        """
        msg = f" - Constr <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'cp': cp, 'sub_map': self.om.rtn.syms.val_map}
            self.optz = self._evaluate_expression(self.code, local_vars=local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating Constraint <{self.name}>.\n{e}")

    def __repr__(self):
        enabled = 'OFF' if self.is_disabled else 'ON'
        out = f"{self.class_name}: {self.name} [{enabled}]"
        return out

    @property
    def e(self):
        """
        Return the calculated constraint LHS value.
        Note that `v` should be used primarily as it is obtained
        from the solver directly.

        `e` is for debugging purpose. For a successfully solved problem,
        `e` should equal to `v`. However, when a problem is infeasible
        or unbounded, `e` can be used to check the constraint LHS value.
        """
        if self.code is None:
            logger.info(f"Constraint <{self.name}> is not parsed yet.")
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
            logger.error(f"Error in calculating constr <{self.name}>.\n{e}")
            return None

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
