"""
Module for optimization Expression.
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


class Expression(OptzBase):
    """
    Base class for expressions used in a routine.

    Routines are authored with ``e_str`` strings; the codegen at
    :func:`ams.prep.generate_for_routine` compiles each ``e_str`` into
    a callable ``e_fn(r)`` taking a :class:`RoutineNS` proxy and
    returning a CVXPY expression. The mutex descriptor on
    ``e_str`` / ``e_fn`` keeps the two forms exclusive.

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
        Expression string (legacy form).
    e_fn : callable, optional
        Callable ``e_fn(r) -> cp.Expression`` taking a :class:`RoutineNS`.
    model : str, optional
        Name of the owner model or group.
    src : str, optional
        Source expression name
    vtype : type, optional
        Value type
    horizon : ams.routines.RParam, optional
        Horizon
    """

    e_str = _EFormDescriptor('e_str', 'e_fn')
    e_fn = _EFormDescriptor('e_fn', 'e_str')

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 e_str: Optional[str] = None,
                 e_fn: Optional[Callable] = None,
                 model: Optional[str] = None,
                 src: Optional[str] = None,
                 vtype: Optional[str] = float,
                 horizon: Optional[str] = None,
                 ):
        OptzBase.__init__(self, name=name, info=info, unit=unit, model=model)
        self.tex_name = tex_name
        self._e_str = None
        self._e_fn = None
        self.e_str = e_str
        self.e_fn = e_fn
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
        if self.e_fn is not None:
            # e_fn form: nothing to parse — defer everything to evaluate()
            return True
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
        if self.e_fn is not None:
            try:
                self.optz = self.e_fn(RoutineNS(self.om.rtn))
            except Exception as e:
                raise Exception(f"Error in evaluating Expression <{self.name}> "
                                f"via e_fn.\n{e}")
            return True
        msg = f" - Expression <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'np': np, 'cp': cp, 'sub_map': self.om.rtn.syms.val_map}
            self.optz = eval(self.code, {}, local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating Expression <{self.name}>.\n{e}")
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
        Set the value.
        """
        raise NotImplementedError('Cannot set value to an Expression.')
