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

    Parameters
    ----------
    name : str, optional
        Name of this parameter. If not provided, `name` will be set
        to the attribute name.
    tex_name : str, optional
        LaTeX-formatted parameter name. If not provided, `tex_name`
        will be assigned the same as `name`.
    info : str, optional
        A description of this parameter
    src : str, optional
        Source name of the parameter.
    unit : str, optional
        Unit of the parameter.
    model : str, optional
        Name of the owner model or group.
    v : np.ndarray, optional
        External value of the parameter.

    Examples
    --------
    Example 1: Define a routine parameter from a source model or group.

    In this example, we define the parameter `cru` from the source model
    `SFRCost` with the parameter `cru`.

    >>> self.cru = RParam(info='RegUp reserve coefficient',
    >>>                   tex_name=r'c_{r,u}',
    >>>                   unit=r'$/(p.u.)',
    >>>                   name='cru',
    >>>                   src='cru',
    >>>                   model='SFRCost'
    >>>                   )

    Example 2: Define a routine parameter with a user-defined value.

    In this example, we define the parameter with a user-defined value.
    TODO: Add example
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 src: Optional[str] = None,
                 unit: Optional[str] = None,
                 model: Optional[str] = None,
                 v: Optional[np.ndarray] = None,
                 ):

        self.name = name
        self.tex_name = tex_name if (tex_name is not None) else name
        self.info = info
        self.src = name if (src is None) else src
        self.unit = unit
        self.is_group = False
        self.model = model  # name of a group or model
        self.owner = None  # instance of the owner model or group
        self.is_ext = False  # indicate if the value is set externally
        if v is not None:
            self._v = v
            self.is_ext = True

    @property
    def v(self):
        """
        The value of the parameter.

        Notes
        -----
        This property is a wrapper for the ``get`` method of the owner class.
        """
        if self.is_ext:
            return self._v
        elif self.is_group:
            return self.owner.get(src=self.src, attr='v',
                                  idx=self.owner.get_idx())
        else:
            src_param = getattr(self.owner, self.src)
            return getattr(src_param, 'v')

    @property
    def shape(self):
        """
        Return the shape of the parameter.
        """
        return self.v.shape

    @property
    def n(self):
        """
        Return the szie of the parameter.
        """
        if self.is_ext:
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
        if self.is_ext:
            span = ''
            if isinstance(self.v, np.ndarray):
                if 1 in self.v.shape:
                    if len(self.v) <= 20:
                        span = f', v={self.v}'
            else:
                if len(self.v) <= 20:
                    span = f', v={self.v}'
                else:
                    span = f', v in length of {len(self.v)}'
            return f'{self.__class__.__name__}: {self.name}{span}'
        else:
            span = ''
            if 1 <= self.n <= 20:
                span = f', v={self.v}'
                if hasattr(self, 'vin') and (self.vin is not None):
                    span += f', v in length of {self.vin}'

            if isinstance(self.v, np.ndarray):
                if self.v.shape[0] == 1:
                    if len(self.v) <= 20:
                        span = f', v={self.v}'
                else:
                    span = f', v in shape({self.v.shape})'
            elif isinstance(self.v, list):
                if len(self.v) <= 20:
                    span = f', v={self.v}'
                else:
                    span = f', v in length of {len(self.v)}'
            else:
                span = f', v={self.v}'
            return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}{span}'

    def get_idx(self):
        """
        Get the index of the parameter.

        Returns
        -------
        idx : list
            Index of the parameter.
        """
        if self.is_group:
            return self.owner.get_idx()
        elif self.owner is None:
            logger.info(f'Param <{self.name}> has no owner.')
            return None
        else:
            return self.owner.idx.v
