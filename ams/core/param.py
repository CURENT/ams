"""
Base class for parameters.
"""


import logging

from typing import Callable, Iterable, List, Optional, Tuple, Type, Union
from collections import OrderedDict

import numpy as np

from andes.core.common import Config
from andes.core import BaseParam, DataParam, IdxParam, NumParam, ExtParam
from andes.models.group import GroupBase

from ams.core.var import Algeb

logger = logging.getLogger(__name__)


class RParam:
    """
    Class for parameters used in a routine.

    This class is developed to simplify the routine definition.


    This class is an extension of conventional parameters
    ``BaseParam``, ``DataParam``, ``IdxParam``, and ``NumParam``.
    It contains a `group` attribute to indicate the group.

    Parameters
    ----------
    name : str, optional
        Name of the parameter.
    tex_name : str, optional
        TeX name of the parameter.
    info : str, optional
        Additional information about the parameter.
    src : str, optional
        Source of the parameter.
    unit : str, optional
        Unit of the parameter.
    owner_name : str, optional
        Name of the owner model or group.
    v : np.ndarray, optional
        Value of the parameter.
    v_str : str, optional
        String representation of the parameter value.

    Attributes
    ----------
    name : str
        Name of the parameter.
    tex_name : str
        TeX name of the parameter.
    info : str
        Additional information about the parameter.
    src : str
        Source of the parameter.
    unit : str
        Unit of the parameter.
    is_group : bool
        Indicates if the parameter is a group variable.
    owner_name : str
        Name of the owner model or group.
    owner : object
        Instance of the owner model or group.
    is_set : bool
        Indicates if the value is set externally.
    v_str : str
        String representation of the parameter value.

    Properties
    ----------
    v : np.ndarray
        Value of the parameter.
    n : int
        Number of elements in the parameter.
    class_name : str
        Class name of the parameter.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 src: Optional[str] = None,
                 unit: Optional[str] = None,
                 owner_name: Optional[str] = None,
                 v: Optional[np.ndarray] = None,
                 v_str: Optional[str] = None,
                 ):

        self.name = name
        self.tex_name = tex_name if (tex_name is not None) else name
        self.info = info
        self.src = name if (src is None) else src
        self.unit = unit
        self.is_group = False
        self.owner_name = owner_name  # indicate if this variable is a group variable
        self.owner = None  # instance of the owner model or group
        self.is_set = False  # indicate if the value is set externally
        self.v_str = v_str
        if v is not None:
            self._v = v
            self.is_set = True

    @property
    def v(self):
        """
        The value of the parameter.

        Notes
        -----
        This property is a wrapper for the ``get`` method of the owner class.
        """
        if self.is_set:
            return self._v
        elif self.is_group:
            return self.owner.get(src=self.src, attr='v',
                                  idx=self.owner.get_idx())
        else:
            src_param = getattr(self.owner, self.src)
            return getattr(src_param, 'v')

    @property
    def n(self):
        """
        Return the szie of the parameter.
        """
        if self.is_set:
            return self._v.shape[0]
        else:
            return self.owner.n

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    def __repr__(self):
        if self.is_set:
            span = ''
            if self.v.ndim == 1:
                if len(self.v) <= 20:
                    span = f', v={self.v}'
            else:
                span = f', v=shape{self.v.shape}'
            return f'{self.__class__.__name__}: {self.name}{span}'
        else:
            span = ''
            if 1 <= self.n <= 20:
                span = f', v={self.v}'
                if hasattr(self, 'vin') and (self.vin is not None):
                    span += f', vin={self.vin}'

            if self.v.ndim == 1:
                if len(self.v) <= 20:
                    span = f', v={self.v}'
            else:
                span = f', v=shape{self.v.shape}'

            return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}{span}'

    def get_idx(self):
        if self.is_group:
            return self.owner.get_idx()
        elif self.owner is None:
            logger.info(f'Param <{self.name}> has no owner.')
            return None
        else:
            return self.owner.idx.v
