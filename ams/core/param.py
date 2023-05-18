"""
Base class for parameters.
"""


import logging

from typing import Optional, Union
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
                 group: Optional[GroupBase] = None,
                 **kwargs):
        # Initialize the parent classes dynamically based on their type
        for parent_class in RParam.__bases__:
            if issubclass(RParam, parent_class):
                parent_class.__init__(self, **kwargs)

        # Set the group attribute
        self.group = group
