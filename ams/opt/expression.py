"""
Module for optimization Expression.
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


class Expression(OptzBase):
    """
    Base class for expressions used in a routine.

    Parameters
    ----------
    name : str, optional
        Expression name. One should typically assigning the name directly because
        it will be automatically assigned by the model. The value of ``name``
        will be the symbol name to be used in expressions.
    tex_name : str, optional
        LaTeX-formatted variable symbol. Defaults to the value of ``name``.
    info : str, optional
        Descriptive information
    unit : str, optional
        Unit
    e_str : str, optional
        Expression string
    model : str, optional
        Name of the owner model or group.
    src : str, optional
        Source expression name
    vtype : type, optional
        Value type
    horizon : ams.routines.RParam, optional
        Horizon
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 e_str: Optional[str] = None,
                 model: Optional[str] = None,
                 src: Optional[str] = None,
                 vtype: Optional[str] = float,
                 horizon: Optional[str] = None,
                 ):
        OptzBase.__init__(self, name=name, info=info, unit=unit, model=model)
        self.tex_name = tex_name
        self.e_str = e_str
        self.optz = None
        self.code = None
        self.src = src
        self.horizon = horizon

    @ensure_symbols
    def parse(self):
        """
        Parse the expression.

        Returns
        -------
        bool
            Returns True if the parsing is successful, False otherwise.
        """
        sub_map = self.om.rtn.syms.sub_map
        code_expr = self.e_str
        for pattern, replacement in sub_map.items():
            try:
                code_expr = re.sub(pattern, replacement, code_expr)
            except Exception as e:
                raise Exception(f"Error in parsing expr <{self.name}>.\n{e}")
        self.code = code_expr
        msg = f" - Expression <{self.name}>: {self.e_str}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the expression.

        Returns
        -------
        bool
            Returns True if the evaluation is successful, False otherwise.
        """
        msg = f" - Expression <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'np': np, 'cp': cp, 'sub_map': self.om.rtn.syms.val_map}
            self.optz = self._evaluate_expression(self.code, local_vars=local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating Expression <{self.name}>.\n{e}")
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

    @v.setter
    def v(self, value):
        """
        Set the value.
        """
        raise NotImplementedError('Cannot set value to an Expression.')

    @property
    def shape(self):
        """
        Return the shape.
        """
        try:
            return self.om.__dict__[self.name].shape
        except KeyError:
            logger.warning('Shape info is not ready before initialization.')
            return None

    @property
    def size(self):
        """
        Return the size.
        """
        if self.rtn.initialized:
            return self.om.__dict__[self.name].size
        else:
            logger.warning(f'Routine <{self.rtn.class_name}> is not initialized yet.')
            return None

    @property
    def e(self):
        """
        Return the calculated expression value.
        """
        if self.code is None:
            logger.info(f"Expression <{self.name}> is not parsed yet.")
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
