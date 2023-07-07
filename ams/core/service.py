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
    Revised from module `andes.core.service.BaseService`.

    Parameters
    ----------
    name : str, optional
        Instance name.
    tex_name : str, optional
        TeX name.
    unit : str, optional
        Unit.
    info : str, optional
        Description.
    vtype : Type, optional
        Variable type.
    model : str, optional
        Model name.
    """

    def __init__(self,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
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

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

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

            return f'{self.class_name}: {self.rtn.class_name}.{self.name}{val_str}'


class ROperationService(RBaseService):
    """
    Base calss for operational services used in routine.
    """

    def __init__(self,
                 u: Callable,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 model: str = None):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype, model=model)
        self.u = u


class VarSum(RBaseService):
    """
    Build sum matrix for a ``Var`` vector in the shape of collection model,
    ``Area`` or ``Region``.
    The value array is in the shape of (nr, nc), where nr is the length of
    rid instance idx, and nc is the length of the cid value.

    In an IEEE-14 Bus system, we have the zonal definition by the
    ``Region`` model. Suppose in it we have two regions, "ZONE1" and
    "ZONE2".

    Follwing it, we have a zonal SFR requirement model ``SFR`` that
    defines the zonal reserve requirements for each zone.

    All 14 buses are classified to a zone by the `IdxParam` ``zone``,
    and the 5 generators are connected to buses
    (idx): [2, 3, 1, 6, 8], and the zone of these generators are thereby:
    ['ZONE1', 'ZONE1', 'ZONE2', 'ZONE2', 'ZONE1'].

    In the `RTED` model, we have the Vars ``pru`` and ``prd`` in the
    shape of generators.

    Then, the Region model has idx ['ZONE1', 'ZONE2'], and the ``gsm`` value
    will be [[1, 1, 0, 0, 1], [0, 0, 1, 1, 0]].

    Finally, the zonal reserve requirements can be formulated as
    constraints in the optimization problem: "gsm @ pru <= du" and
    "gsm @ prd <= dd".

    See ``gsm`` definition in :py:mod:`ams.routines.rted.RTEDModel` for
    more details.

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
    cid : Callable
        Column indexer, usually an RParam that points to an IdxParam
        "zone" or "area" defined in a model.
    rid : str
        Row indexer, usually the name of a collection model, e.g.,
        "Region" or "Area".
    """

    def __init__(self, name: str = None, tex_name: str = None, unit: str = None,
                 info: str = None, vtype: Type = None,
                 model: str = None,
                 cid: Callable = None,
                 rid: str = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype, model=model)
        self.cid = cid
        self.rid = rid

    @property
    def v(self):
        rid = getattr(self.rtn.system, self.rid)
        ridx = None
        try:
            ridx = rid.idx.v
        except AttributeError:
            ridx = rid.get_idx()

        row, col = np.meshgrid(self.cid.v, ridx)
        result = (row == col).astype(int)

        return result


class VarReduce(ROperationService):
    """
    A service that reduces a two dimensional variable matrix
    to one dimensional vector.
    """

    def __init__(self, 
                 u: Callable,
                 fun: Callable,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 model: str = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype, model=model,
                         u=u)
        self.fun = fun

    @property
    def v(self):
        # NOTE: the nc will run into error if input var has no owner
        nc = len(self.u.owner.get_idx())
        shape = (1, nc)
        return self.fun(shape=shape)


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
