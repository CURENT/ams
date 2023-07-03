"""
Service.
"""

import logging  # NOQA
from typing import Callable, Optional, Type, Union  # NOQA

import numpy as np  # NOQA

from andes.core.service import BaseService, BackRef, RefFlatten  # NOQA


logger = logging.getLogger(__name__)


class RBaseService(BaseService):
    """
    Base class for services that are used in a routine.
    Revised from :py:class:`andes.core.service.BaseService`.

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
    model : str
        Model name.
    """

    def __init__(self, name: str = None, tex_name: str = None,
                 unit: str = None,
                 info: str = None, vtype: Type = None,
                 model: str = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype)
        self.model = model
        self.export = False
        self.is_group = False
        self.rtn = None

    @property
    def v(self):
        """
        Value of the service.
        """
        return None

    def __repr__(self):
        val_str = ''

        v = self.v

        if v is None:
            return f'{self.class_name}: {self.owner.class_name}.{self.name}'
        elif isinstance(v, np.ndarray):
            if v.ndim == 1:
                if len(self.v) <= 20:
                    val_str = f', v={self.v}'
                else:
                    val_str = f', v=shape{self.v.shape}'
            else:
                val_str = f', v=shape{self.v.shape}'

            return f'{self.class_name}: {self.owner.class_name}.{self.name}{val_str}'


class NumTile(BaseService):
    """
    Tile a numerical vector.
    """

    def __init__(self, name: str = None, tex_name: str = None,
                 unit: str = None,
                 info: str = None, vtype: Type = None,
                 model: str = None,
                 src: str = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype)
        self.src = src
        self.model = model
        self.export = False

    @property
    def v(self):
        pass


class VarSub(BaseService):
    """
    Build substraction matrix for a variable vector in the shape of
    indexer vector.
    """

    def __init__(self, name: str = None, tex_name: str = None,
                 unit: str = None,
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


class VarSum(RBaseService):
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
    model : str
        Model name.
    indexer : Callable
        Indexer instance.
    """

    def __init__(self, name: str = None, tex_name: str = None, unit: str = None,
                 info: str = None, vtype: Type = None,
                 model: str = None,
                 indexer: Callable = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype, model=model)
        self.indexer = indexer

    @property
    def v(self):
        nr = self.indexer.n
        nc = self.owner.n

        idx = None
        try:
            idx = self.owner.idx.v
        except AttributeError:
            idx = self.owner.get_idx()
        try:
            mdl_indexer_val = self.owner.get(src='zone', attr='v',
                                             idx=idx, allow_none=True, default=None)
        except KeyError:
            raise KeyError(f'Indexer <zone> not found in model <{self.model}>!')
        row, col = np.meshgrid(mdl_indexer_val, self.indexer.v)
        result = (row == col).astype(int)

        return result
