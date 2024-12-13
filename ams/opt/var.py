"""
Module for optimization Var.
"""
import logging

from typing import Optional, Union
from collections import OrderedDict
import re

import numpy as np

from andes.core.common import Config

import cvxpy as cp

from ams.utils import pretty_long_message
from ams.shared import _prefix, _max_length

from ams.opt import OptzBase, ensure_symbols, ensure_mats_and_parsed

logger = logging.getLogger(__name__)


class Var(OptzBase):
    """
    Base class for variables used in a routine.

    When `horizon` is provided, the variable will be expanded to a matrix,
    where rows are indexed by the source variable index and columns are
    indexed by the horizon index.

    Parameters
    ----------
    info : str, optional
        Descriptive information
    unit : str, optional
        Unit
    tex_name : str
        LaTeX-formatted variable symbol. Defaults to the value of ``name``.
    name : str, optional
        Variable name. One should typically assigning the name directly because
        it will be automatically assigned by the model. The value of ``name``
        will be the symbol name to be used in expressions.
    src : str, optional
        Source variable name. Defaults to the value of ``name``.
    model : str, optional
        Name of the owner model or group.
    horizon : ams.routines.RParam, optional
        Horizon idx.
    nonneg : bool, optional
        Non-negative variable
    nonpos : bool, optional
        Non-positive variable
    cplx : bool, optional
        Complex variable
    imag : bool, optional
        Imaginary variable
    symmetric : bool, optional
        Symmetric variable
    diag : bool, optional
        Diagonal variable
    psd : bool, optional
        Positive semi-definite variable
    nsd : bool, optional
        Negative semi-definite variable
    hermitian : bool, optional
        Hermitian variable
    boolean : bool, optional
        Boolean variable
    integer : bool, optional
        Integer variable
    pos : bool, optional
        Positive variable
    neg : bool, optional
        Negative variable

    Attributes
    ----------
    a : np.ndarray
        Variable address.
    _v : np.ndarray
        Local-storage of the variable value.
    rtn : ams.routines.Routine
        The owner routine instance.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 src: Optional[str] = None,
                 unit: Optional[str] = None,
                 model: Optional[str] = None,
                 shape: Optional[Union[tuple, int]] = None,
                 v0: Optional[str] = None,
                 horizon: Optional[str] = None,
                 nonneg: Optional[bool] = False,
                 nonpos: Optional[bool] = False,
                 cplx: Optional[bool] = False,
                 imag: Optional[bool] = False,
                 symmetric: Optional[bool] = False,
                 diag: Optional[bool] = False,
                 psd: Optional[bool] = False,
                 nsd: Optional[bool] = False,
                 hermitian: Optional[bool] = False,
                 boolean: Optional[bool] = False,
                 integer: Optional[bool] = False,
                 pos: Optional[bool] = False,
                 neg: Optional[bool] = False,
                 ):
        self.name = name
        self.info = info
        self.unit = unit

        self.tex_name = tex_name if tex_name else name
        # variable internal index inside a model (assigned in run time)
        self.id = None
        OptzBase.__init__(self, name=name, info=info, unit=unit, model=model)
        self.src = src
        self.v0 = v0
        self.horizon = horizon
        self._shape = shape
        self._v = None
        self.a: np.ndarray = np.array([], dtype=int)

        self.config = Config(name=self.class_name)  # `config` that can be exported

        self.config.add(OrderedDict((('nonneg', nonneg),
                                     ('nonpos', nonpos),
                                     ('complex', cplx),
                                     ('imag', imag),
                                     ('symmetric', symmetric),
                                     ('diag', diag),
                                     ('psd', psd),
                                     ('nsd', nsd),
                                     ('hermitian', hermitian),
                                     ('boolean', boolean),
                                     ('integer', integer),
                                     ('pos', pos),
                                     ('neg', neg),
                                     )))

    @property
    def v(self):
        """
        Return the CVXPY variable value.
        """
        if self.optz is None:
            return None
        if self.optz.value is None:
            try:
                shape = self.optz.shape
                return np.zeros(shape)
            except AttributeError:
                return None
        else:
            return self.optz.value

    @v.setter
    def v(self, value):
        if self.optz is None:
            logger.info(f"Variable <{self.name}> is not initialized yet.")
        else:
            self.optz.value = value

    @ensure_symbols
    def parse(self):
        """
        Parse the variable.
        """
        sub_map = self.om.rtn.syms.sub_map
        # NOTE: number of rows is the size of the source variable
        if self.owner is not None:
            nr = self.owner.n
            nc = 0
            if self.horizon:
                # NOTE: numer of columns is the horizon if exists
                nc = int(self.horizon.n)
                shape = (nr, nc)
            else:
                shape = (nr,)
        elif isinstance(self._shape, int):
            shape = (self._shape,)
            nr = shape
            nc = 0
        elif isinstance(self._shape, tuple):
            shape = self._shape
            nr = shape[0]
            nc = shape[1] if len(shape) > 1 else 0
        else:
            raise ValueError(f"Invalid shape {self._shape}.")
        code_var = f"var({shape}, **config)"
        logger.debug(f" - Var <{self.name}>: {self.code}")
        for pattern, replacement, in sub_map.items():
            try:
                code_var = re.sub(pattern, replacement, code_var)
            except Exception as e:
                raise Exception(f"Error in parsing var <{self.name}>.\n{e}")
        self.code = code_var
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the variable.

        Returns
        -------
        bool
            Returns True if the evaluation is successful, False otherwise.
        """
        # NOTE: in CVXPY, Config only allow lower case letters
        config = {}  # used in `self.code`
        for k, v in self.config.as_dict().items():
            if k == 'psd':
                config['PSD'] = v
            elif k == 'nsd':
                config['NSD'] = v
            elif k == 'bool':
                config['boolean'] = v
            else:
                config[k] = v
        msg = f" - Var <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'config': config, 'cp': cp}
            self.optz = eval(self.code, {}, local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating Var <{self.name}>.\n{e}")
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}'
