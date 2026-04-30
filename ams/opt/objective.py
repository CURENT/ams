"""
Module for optimization Objective.
"""
import logging

from typing import Optional
import re

import numpy as np  # noqa: F401  # used by routine `e_str` evaluation context

import cvxpy as cp

from ams.utils import pretty_long_message
from ams.shared import _prefix, _max_length

from ams.opt import OptzBase, ensure_symbols, ensure_mats_and_parsed

logger = logging.getLogger(__name__)


class Objective(OptzBase):
    """
    Base class for objective functions.

    This class serves as a template for defining objective functions. Each
    instance of this class represents a single objective function that can
    be minimized or maximized depending on the sense ('min' or 'max').

    Parameters
    ----------
    name : str, optional
        A user-defined name for the objective function.
    e_str : str, optional
        A mathematical expression representing the objective function.
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

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 sense: Optional[str] = 'min'):
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self.e_str = e_str
        self.sense = sense
        self.code = None

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
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_obj = self.e_str
        for pattern, replacement, in sub_map.items():
            try:
                code_obj = re.sub(pattern, replacement, code_obj)
            except Exception as e:
                raise Exception(f"Error in parsing obj <{self.name}>.\n{e}")
        if self.sense not in ['min', 'max']:
            raise ValueError(f'Objective sense {self.sense} is not supported.')
        sense = 'cp.Minimize' if self.sense == 'min' else 'cp.Maximize'
        self.code = f"{sense}({code_obj})"
        msg = f" - Objective <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the objective function.

        Returns
        -------
        bool
            Returns True if the evaluation is successful, False otherwise.
        """
        logger.debug(f" - Objective <{self.name}>: {self.e_str}")
        try:
            local_vars = {'self': self, 'cp': cp}
            self.optz = eval(self.code, {}, local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating Objective <{self.name}>.\n{e}")
        return True

    def __repr__(self):
        return f"{self.class_name}: {self.name} [{self.sense.upper()}]"
