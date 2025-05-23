"""
Module for optimization Param.
"""
import logging

from typing import Optional
from collections import OrderedDict
import re

import numpy as np

import cvxpy as cp

from ams.core import Config
from ams.shared import sps

from ams.opt import OptzBase, ensure_symbols, ensure_mats_and_parsed

logger = logging.getLogger(__name__)


class Param(OptzBase):
    """
    Base class for parameters used in a routine.

    Parameters
    ----------
    no_parse: bool, optional
        True to skip parsing the parameter.
    nonneg: bool, optional
        True to set the parameter as non-negative.
    nonpos: bool, optional
        True to set the parameter as non-positive.
    cplx: bool, optional
        True to set the parameter as complex, avoiding the use of `complex`.
    imag: bool, optional
        True to set the parameter as imaginary.
    symmetric: bool, optional
        True to set the parameter as symmetric.
    diag: bool, optional
        True to set the parameter as diagonal.
    hermitian: bool, optional
        True to set the parameter as hermitian.
    boolean: bool, optional
        True to set the parameter as boolean.
    integer: bool, optional
        True to set the parameter as integer.
    pos: bool, optional
        True to set the parameter as positive.
    neg: bool, optional
        True to set the parameter as negative.
    sparse: bool, optional
        True to set the parameter as sparse.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 no_parse: Optional[bool] = False,
                 nonneg: Optional[bool] = False,
                 nonpos: Optional[bool] = False,
                 cplx: Optional[bool] = False,
                 imag: Optional[bool] = False,
                 symmetric: Optional[bool] = False,
                 diag: Optional[bool] = False,
                 hermitian: Optional[bool] = False,
                 boolean: Optional[bool] = False,
                 integer: Optional[bool] = False,
                 pos: Optional[bool] = False,
                 neg: Optional[bool] = False,
                 sparse: Optional[bool] = False,
                 ):
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self.no_parse = no_parse  # True to skip parsing the parameter
        self.sparse = sparse

        self.config = Config(name=self.class_name)  # `config` that can be exported

        self.config.add(OrderedDict((('nonneg', nonneg),
                                     ('nonpos', nonpos),
                                     ('complex', cplx),
                                     ('imag', imag),
                                     ('symmetric', symmetric),
                                     ('diag', diag),
                                     ('hermitian', hermitian),
                                     ('boolean', boolean),
                                     ('integer', integer),
                                     ('pos', pos),
                                     ('neg', neg),
                                     )))

    @ensure_symbols
    def parse(self):
        """
        Parse the parameter.

        Returns
        -------
        bool
            Returns True if the parsing is successful, False otherwise.
        """
        sub_map = self.om.rtn.syms.sub_map
        code_param = "param(**config)"
        for pattern, replacement, in sub_map.items():
            try:
                code_param = re.sub(pattern, replacement, code_param)
            except Exception as e:
                raise Exception(f"Error in parsing param <{self.name}>.\n{e}")
        self.code = code_param
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the parameter.
        """
        if self.no_parse:
            return True

        config = self.config.as_dict()
        try:
            msg = f"Parameter <{self.name}> is set as sparse, "
            msg += "but the value is not sparse."
            if self.sparse:
                self.v = sps.csr_matrix(self.v)

            # Create the cvxpy.Parameter object
            if isinstance(self.v, np.ndarray):
                self.optz = cp.Parameter(shape=self.v.shape, **config)
            else:
                self.optz = cp.Parameter(**config)
            self.optz.value = self.v
        except ValueError:
            msg = f"Parameter <{self.name}> has non-numeric value, "
            msg += "set `no_parse=True`."
            logger.warning(msg)
            self.no_parse = True
            return True
        except Exception as e:
            raise Exception(f"Error in evaluating Param <{self.name}>.\n{e}")
        return True

    def update(self):
        """
        Update the Parameter value.
        """
        # NOTE: skip no_parse parameters
        if self.optz is None:
            return None
        self.optz.value = self.v
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.name}'
