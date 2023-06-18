"""
Service.
"""

import logging
from typing import Callable, Optional, Type, Union

from andes.core.service import BaseService, BackRef, RefFlatten


logger = logging.getLogger(__name__)


class VarSum(BaseService):
    """
    Build sum matrix for a variable.

    #TODO: add example
    """

    def __init__(self, name: str = None, tex_name: str = None, unit: str = None,
                 info: str = None, vtype: Type = None,
                 indexer: Callable = None,
                 model: str = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype)
        self.indexer = indexer
        self.model = model
        self.export = False

    @property
    def v(self):
        return None

    # FIXME: might need to fix assign_memory to fit 2D array initialization
