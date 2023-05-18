"""
Base class for parameters.
"""


import logging

from typing import Callable, Iterable, List, Optional, Tuple, Type, Union
from collections import OrderedDict

import numpy as np

from andes.core.common import Config
from andes.core import BaseParam, DataParam, IdxParam, NumParam
from andes.models.group import GroupBase

from ams.core.var import Algeb

logger = logging.getLogger(__name__)


class RParam:
    """
    Class for parameters in a routine.

    This class is an extension of conventional parameters
    `BaseParam`, `DataParam`, `IdxParam`, and `NumParam`.
    It contains a `group` attribute to indicate the group.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 is_group: Optional[bool] = False,
                 group_name: Optional[str] = None,
                 ):

        self.name = name
        self.tex_name = tex_name if (tex_name is not None) else name
        self.info = info
        self.unit = unit
        self.is_group = is_group
        if is_group and not group_name:
            raise ValueError('A group name is required if is_group is True.')
        self.group_name = group_name  # indicate if this variable is a group variable
        self.group = None  # instance of the owner group
