"""
Service.
"""

import logging  # NOQA
from typing import Callable, Optional, Type, Union, Iterable  # NOQA

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
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype)
        self.export = False
        self.is_group = False
        self.rtn = None

    @property
    def shape(self):
        """
        Return the shape of the service.
        """
        if isinstance(self.v, np.ndarray):
            return self.v.shape
        else:
            raise TypeError(f'{self.class_name}: {self.name} is not an array.')

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
            if v.shape[0] == 1:
                if len(self.v) <= 20:
                    val_str = f', v={self.v}'
                else:
                    val_str = f', v in shape of {self.v.shape}'
            else:
                val_str = f', v in shape of {self.v.shape}'

            return f'{self.class_name}: {self.rtn.class_name}.{self.name}{val_str}'
        else:
            return f'{self.class_name}: {self.rtn.class_name}.{self.name}'


class ValueService(RBaseService):
    """
    Service to store given numeric values.

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
                 name: str,
                 value: np.ndarray,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype)
        self._v = value

    @property
    def v(self):
        """
        Value of the service.
        """
        return self._v


class ROperationService(RBaseService):
    """
    Base calss for operational services used in routine.

    Parameters
    ----------
    u : Callable
        Input.
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
                 u: Callable,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype)
        self.u = u


class NumOp(ROperationService):
    """
    Perform an operation on a numerical array using the
    function ``fun(u.v, **args)``.

    Note that the scalar output is converted to a 1D array.

    The optional kwargs are passed to the input function.

    Parameters
    ----------
    u : Callable
        Input.
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
    rfun : Callable, optional
        Function to apply to the output of ``fun``.
    rargs : dict, optional
        Keyword arguments to pass to ``rfun``.
    expand_dims : int, optional
        Expand the dimensions of the output array along a specified axis.
    """

    def __init__(self,
                 u: Callable,
                 fun: Callable,
                 args: dict = {},
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 rfun: Callable = None,
                 rargs: dict = {},
                 expand_dims: int = None):
        tex_name = tex_name if tex_name is not None else u.tex_name
        unit = unit if unit is not None else u.unit
        info = info if info is not None else u.info
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype, u=u,)
        self.fun = fun
        self.args = args
        self.rfun = rfun
        self.rargs = rargs
        self.expand_dims = expand_dims

    @property
    def v0(self):
        if isinstance(self.u, Iterable):
            out = self.fun([u.v for u in self.u], **self.args)
        else:
            out = self.fun(self.u.v, **self.args)
        if not isinstance(out, np.ndarray):
            out = np.array([out])
        return out

    @property
    def v1(self):
        if self.rfun is not None:
            return self.rfun(self.v0, **self.rargs)
        else:
            return self.v0

    @property
    def v(self):
        if self.expand_dims is not None:
            return np.expand_dims(self.v1, axis=int(self.expand_dims))
        else:
            return self.v1


class NumExpandDim(NumOp):
    """
    Expand the dimensions of the input array along a specified axis
    using NumPy's ``np.expand_dims(u.v, axis=axis)``.

    Parameters
    ----------
    u : Callable
        Input.
    axis : int
        Axis along which to expand the dimensions (default is 0).
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
                 u: Callable,
                 axis: int = 0,
                 args: dict = {},
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=np.expand_dims, args=args)
        self.axis = axis

    @property
    def v(self):
        return self.fun(self.u.v, axis=self.axis, **self.args)


class NumOpDual(NumOp):
    """
    Performan an operation on two numerical arrays using the
    function ``fun(u.v, u2.v, **args)``.

    Note that the scalar output is converted to a 1D array.

    The optional kwargs are passed to the input function.

    Parameters
    ----------
    u : Callable
        Input.
    u2 : Callable
        Input2.
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
    rfun : Callable, optional
        Function to apply to the output of ``fun``.
    rargs : dict, optional
        Keyword arguments to pass to ``rfun``.
    expand_dims : int, optional
        Expand the dimensions of the output array along a specified axis.
    """

    def __init__(self,
                 u: Callable,
                 u2: Callable,
                 fun: Callable,
                 args: dict = {},
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 rfun: Callable = None,
                 rargs: dict = {},
                 expand_dims: int = None):
        tex_name = tex_name if tex_name is not None else u.tex_name
        unit = unit if unit is not None else u.unit
        info = info if info is not None else u.info
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=fun, args=args,
                         rfun=rfun, rargs=rargs,
                         expand_dims=expand_dims)
        self.u2 = u2

    @property
    def v0(self):
        out = self.fun(self.u.v, self.u2.v, **self.args)
        if not isinstance(out, np.ndarray):
            out = np.array([out])
        return out


class NumHstack(NumOp):
    """
    Repeat an array along the second axis nc times
    using NumPy's hstack function, where nc is the column number of the
    reference array,
    ``np.hstack([u.v[:, np.newaxis] * ref.shape[1]], **kwargs)``.

    Parameters
    ----------
    u : Callable
        Input array.
    ref : Callable
        Reference array used to determine the number of repetitions.
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
                 u: Callable,
                 ref: Callable,
                 args: dict = {},
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 rfun: Callable = None,
                 rargs: dict = {}):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=np.hstack, args=args,
                         rfun=rfun, rargs=rargs)
        self.ref = ref

    @property
    def v0(self):
        nc = 1
        if hasattr(self.ref, "shape"):
            nc = self.ref.shape[1]
        elif isinstance(self.ref.v, (list, tuple)):
            nc = len(self.ref.v)
        else:
            raise AttributeError(f"{self.rtn.class_name}: ref {self.ref.name} has no attribute shape nor length.")
        return self.fun([self.u.v[:, np.newaxis]] * nc,
                        **self.args)


class ZonalSum(NumOp):
    """
    Build zonal sum matrix for a vector in the shape of collection model,
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
    u : Callable
        Input.
    zone : str
        Zonal model name, e.g., "Area" or "Region".
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

    def __init__(self,
                 u: Callable,
                 zone: str,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 rfun: Callable = None,
                 rargs: dict = {},
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=None, args={},
                         rfun=rfun, rargs=rargs)
        self.zone = zone

    @property
    def v0(self):
        try:
            zone_mdl = getattr(self.rtn.system, self.zone)
        except AttributeError:
            raise AttributeError(f'Zonal model <{self.zone}> not found.')
        ridx = None
        try:
            ridx = zone_mdl.idx.v
        except AttributeError:
            ridx = zone_mdl.get_idx()

        row, col = np.meshgrid(self.u.v, ridx)
        # consistency check
        is_subset = set(self.u.v).issubset(set(ridx))
        if not is_subset:
            raise ValueError(f'{self.u.model} contains undefined zone, likey a data error.')
        result = (row == col).astype(int)

        return result


class VarReduction(NumOp):
    """
    A numerical matrix to reduce a 2D variable to 1D,
    ``np.fun(shape=(1, u.n))``.

    Parameters
    ----------
    u : Callable
        The input matrix variable.
    fun : Callable
        The reduction function that takes a shape parameter (1D shape) as input.
    name : str, optional
        The name of the instance.
    tex_name : str, optional
        The TeX name for the instance.
    unit : str, optional
        The unit of the output.
    info : str, optional
        A description of the operation.
    vtype : Type, optional
        The variable type.
    model : str, optional
        The model name associated with the operation.
    """

    def __init__(self,
                 u: Callable,
                 fun: Callable,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 rfun: Callable = None,
                 rargs: dict = {},
                 **kwargs
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=None, rfun=rfun, rargs=rargs,
                         **kwargs)
        self.fun = fun

    @property
    def v0(self):
        return self.fun(shape=(1, self.u.n))


class RampSub(NumOp):
    """
    Build a substraction matrix for a 2D variable in the shape (nr, nr-1),
    where nr is the rows of the input.

    This can be used for generator ramping constraints in multi-period
    optimization problems.

    The subtraction matrix is constructed as follows:
    ``np.eye(nr, nc, k=-1) - np.eye(nr, nc, k=0)``.

    Parameters
    ----------
    u : Callable
        Input.
    horizon : Callable
        Horizon reference.
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

    def __init__(self,
                 u: Callable,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 rfun: Callable = None,
                 rargs: dict = {},
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=None, rfun=rfun, rargs=rargs,)

    @property
    def v0(self):
        return self.v

    @property
    def v1(self):
        return self.v

    @property
    def v(self):
        nr = self.u.horizon.n
        return np.eye(nr, nr-1, k=-1) - np.eye(nr, nr-1, k=0)
