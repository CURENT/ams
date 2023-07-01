"""
Service.
"""

import logging
from typing import Callable, Optional, Type, Union

import numpy as np

from andes.core.service import BaseService, BackRef, RefFlatten


logger = logging.getLogger(__name__)


class VarSum(BaseService):
    """
    Build sum matrix for a variable vector in the shape of indexer vector.
    The value array is in the shape of (nr, nc), where nr is the number of
    unique values in indexer.v, and nc is the length of the target var.

    See :py:mod:`ams.models.region` for example usage.

    Parameters
    ----------
    name : str
        Instance name.
    tex_name : str
        TeX name.
    unit : str
        Unit.
    info : str
        Description.
    vtype : Type
        Variable type.
    indexer : Callable
        Indexer instance.
    model : str
        Model name.
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
        nr = self.indexer.n
        mdl_or_grp = self.owner.system.__dict__[self.model]
        nc = mdl_or_grp.n

        idx = None
        try:
            idx = mdl_or_grp.idx.v
        except AttributeError:
            idx = mdl_or_grp.get_idx()
        try:
            mdl_indexer_val = mdl_or_grp.get(src='zone', attr='v',
                                             idx=idx, allow_none=True, default=None)
        except KeyError:
            raise KeyError(f'Indexer <zone> not found in model <{self.model}>!')
        row, col = np.meshgrid(mdl_indexer_val, self.indexer.v)
        result = (row == col).astype(int)

        return result
