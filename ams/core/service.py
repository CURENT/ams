"""
Service.
"""

import logging
from typing import Callable, Type

import numpy as np
import scipy.sparse as spr

from andes.core.service import BaseService, BackRef, RefFlatten  # NOQA

from ams.opt.omodel import Param


logger = logging.getLogger(__name__)


class RBaseService(BaseService, Param):
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
    no_parse: bool, optional
        True to skip parsing the service.
    sparse: bool, optional
        True to return output as scipy csr_matrix.
    """

    def __init__(self,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 no_parse: bool = False,
                 sparse: bool = False,
                 ):
        Param.__init__(self, name=name, unit=unit, info=info,
                       no_parse=no_parse)
        BaseService.__init__(self, name=name, tex_name=tex_name, unit=unit,
                             info=info, vtype=vtype)
        self.export = False
        self.is_group = False
        self.rtn = None
        self.sparse = sparse

    @property
    def shape(self):
        """
        Return the shape of the service.
        """
        if isinstance(self.v, (np.ndarray, spr.csr_matrix)):
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
        if self.name is None:
            return f'{self.class_name}: {self.rtn.class_name}'
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
    sparse: bool, optional
        True to return output as scipy csr_matrix.
    """

    def __init__(self,
                 name: str,
                 value: np.ndarray,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 no_parse: bool = False,
                 sparse: bool = False,
                 ):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         no_parse=no_parse, sparse=sparse)
        self._v = value

    @property
    def v(self):
        """
        Value of the service.
        """
        if self.sparse:
            return spr.csr_matrix(self._v)
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
    sparse: bool, optional
        True to return output as scipy csr_matrix.
    """

    def __init__(self,
                 u: Callable,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 no_parse: bool = False,
                 sparse: bool = False,):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         no_parse=no_parse, sparse=sparse)
        self.u = u


class LoadScale(ROperationService):
    """
    Return load.

    Parameters
    ----------
    u : Callable
        nodal load.
    sd : Callable
        zonal load factor.
    name : str, optional
        Instance name.
    tex_name : str, optional
        TeX name.
    unit : str, optional
        Unit.
    info : str, optional
        Description.
    sparse: bool, optional
        True to return output as scipy csr_matrix.
    """

    def __init__(self,
                 u: Callable,
                 sd: Callable,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 no_parse: bool = False,
                 sparse: bool = False,
                 ):
        tex_name = tex_name if tex_name is not None else u.tex_name
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, u=u, no_parse=no_parse,
                         sparse=sparse)
        self.sd = sd

    @property
    def v(self):
        sys = self.rtn.system
        u_idx = self.u.get_idx()
        u_bus = self.u.owner.get(src='bus', attr='v', idx=u_idx)
        u_zone = sys.Bus.get(src='zone', attr='v', idx=u_bus)
        u_yloc = np.array(sys.Region.idx2uid(u_zone))
        p0s = np.multiply(self.sd.v[:, u_yloc].transpose(),
                          self.u.v[:, np.newaxis])
        if self.sparse:
            return spr.csr_matrix(p0s)
        return p0s


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
    array_out : bool, optional
        Whether to force the output to be an array.
    sparse: bool, optional
        True to return output as scipy csr_matrix.
    """

    def __init__(self,
                 u: Callable,
                 fun: Callable,
                 args: dict = dict(),
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 rfun: Callable = None,
                 rargs: dict = dict(),
                 expand_dims: int = None,
                 array_out=True,
                 no_parse: bool = False,
                 sparse: bool = False,):
        tex_name = tex_name if tex_name is not None else u.tex_name
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype, u=u,
                         no_parse=no_parse, sparse=sparse)
        self.fun = fun
        self.args = args
        self.rfun = rfun
        self.rargs = rargs
        self.expand_dims = expand_dims
        self.array_out = array_out

    @property
    def v0(self):
        out = self.fun(self.u.v, **self.args)
        if self.array_out:
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
            out = np.expand_dims(self.v1, axis=int(self.expand_dims))
        else:
            out = self.v1
        if self.sparse:
            return spr.csr_matrix(out)
        return out


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
    array_out : bool, optional
        Whether to force the output to be an array.
    sparse: bool, optional
        True to return output as scipy csr_matrix.
    """

    def __init__(self,
                 u: Callable,
                 axis: int = 0,
                 args: dict = {},
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 array_out: bool = True,
                 no_parse: bool = False,
                 sparse: bool = False,):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=np.expand_dims, args=args,
                         array_out=array_out,
                         no_parse=no_parse, sparse=sparse)
        self.axis = axis

    @property
    def v(self):
        out = self.fun(self.u.v, axis=self.axis, **self.args)
        if self.sparse:
            return spr.csr_matrix(out)
        return out


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
    array_out : bool, optional
        Whether to force the output to be an array.
    sparse: bool, optional
        True to return output as scipy csr_matrix.
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
                 expand_dims: int = None,
                 array_out=True,
                 no_parse: bool = False,
                 sparse: bool = False,):
        tex_name = tex_name if tex_name is not None else u.tex_name
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=fun, args=args,
                         rfun=rfun, rargs=rargs,
                         expand_dims=expand_dims,
                         array_out=array_out,
                         no_parse=no_parse, sparse=sparse)
        self.u2 = u2

    @property
    def v0(self):
        out = self.fun(self.u.v, self.u2.v, **self.args)
        if self.array_out:
            if not isinstance(out, np.ndarray):
                out = np.array([out])
        if self.sparse:
            return spr.csr_matrix(out)
        return out


class MinDur(NumOpDual):
    """
    Defined to form minimum on matrix for minimum online/offline
    time constraints used in UC.

    Parameters
    ----------
    u : Callable
        Input, should be a ``Var`` with horizon.
    u2 : Callable
        Input2, should be a ``RParam``.
    name : str, optional
        Instance name.
    tex_name : str, optional
        TeX name.
    unit : str, optional
        Unit.
    info : str, optional
        Description.
    sparse: bool, optional
        True to return output as scipy csr_matrix.
    """

    def __init__(self,
                 u: Callable,
                 u2: Callable,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 no_parse: bool = False,
                 sparse: bool = False,):
        tex_name = tex_name if tex_name is not None else u.tex_name
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, u2=u2, fun=None, args=None,
                         rfun=None, rargs=None,
                         expand_dims=None,
                         no_parse=no_parse, sparse=sparse)
        if self.u.horizon is None:
            msg = f'{self.class_name} <{self.name}>.u: <{self.u.name}> '
            msg += 'has no horizon, likely a modeling error.'
            logger.error(msg)

    @property
    def v(self):
        n_gen = self.u.n
        n_ts = self.u.horizon.n
        tout = np.zeros((n_gen, n_ts))
        t = self.rtn.config.t  # dispatch interval

        # minimum online/offline duration
        td = np.ceil(self.u2.v/t).astype(int)

        # Create index arrays for generators and time periods
        i, t = np.meshgrid(np.arange(n_gen), np.arange(n_ts), indexing='ij')
        # Create a mask for valid time periods based on minimum duration
        valid_mask = (t + td[i] <= n_ts)
        tout[i[valid_mask], t[valid_mask]] = 1
        if self.sparse:
            return spr.csr_matrix(tout)
        return tout


class NumHstack(NumOp):
    """
    Repeat an array along the second axis nc times or the length of
    reference array,
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
    sparse: bool, optional
        True to return output as scipy csr_matrix.
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
                 rargs: dict = {},
                 no_parse: bool = False,
                 sparse: bool = False,):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=np.hstack, args=args,
                         rfun=rfun, rargs=rargs,
                         no_parse=no_parse, sparse=sparse)
        self.ref = ref

    @property
    def v0(self):
        nc = 1
        if isinstance(self.ref.v, (list, tuple)):
            nc = len(self.ref.v)
        elif hasattr(self.ref, "shape"):
            nc = self.ref.shape[1]
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
    sparse: bool, optional
        True to return output as scipy csr_matrix.
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
                 no_parse: bool = False,
                 sparse: bool = False,):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=None, args={},
                         rfun=rfun, rargs=rargs,
                         no_parse=no_parse, sparse=sparse)
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


class VarSelect(NumOp):
    """
    A numerical matrix to select a subset of a 2D variable,
    ``u.v[:, idx]``.

    For example, if nned to select Energy Storage output
    power from StaticGen `pg`, following definition can be used:
    ```python
    class RTED:
    ...
    self.ce = VarSelect(u=self.pg, indexer='genE')
    ...
    ```

    Parameters
    ----------
    u : Callable
        The input matrix variable.
    indexer: str
        The name of the indexer source.
    gamma : str, optional
        The name of the indexer gamma.
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
    rfun : Callable, optional
        Function to apply to the output of ``fun``.
    rargs : dict, optional
        Keyword arguments to pass to ``rfun``.
    array_out : bool, optional
        Whether to force the output to be an array.
    sparse: bool, optional
        True to return output as scipy csr_matrix.
    """

    def __init__(self,
                 u: Callable,
                 indexer: str,
                 gamma: str = None,
                 name: str = None,
                 tex_name: str = None,
                 unit: str = None,
                 info: str = None,
                 vtype: Type = None,
                 rfun: Callable = None,
                 rargs: dict = {},
                 array_out: bool = True,
                 no_parse: bool = False,
                 sparse: bool = False,
                 **kwargs):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype, u=u, fun=None,
                         rfun=rfun, rargs=rargs, array_out=array_out,
                         no_parse=no_parse, sparse=sparse,
                         **kwargs)
        self.indexer = indexer
        self.gamma = gamma

    @property
    def v0(self):
        # FIXME: what if reference source has no idx?
        # data consistency check
        indexer = getattr(self.rtn, self.indexer)
        err_msg = f'Indexer source {indexer.model} has no {indexer.src}.'
        group = model = None
        if indexer.model in self.rtn.system.groups.keys():
            group = self.rtn.system.groups[indexer.model]
            group_idx = group.get_idx()
            try:
                ref = group.get(src=indexer.src, attr='v', idx=group_idx)
            except AttributeError:
                raise AttributeError(err_msg)
        elif indexer.model in self.rtn.system.models.keys():
            model = self.rtn.system.models[indexer.model]
            try:
                ref = model.get(src=indexer.src, attr='v', idx=model.idx.v)
            except AttributeError:
                raise AttributeError(err_msg)
        else:
            raise AttributeError(f'Indexer source model {indexer.model} has no ref.')

        try:
            uidx = self.u.get_idx()
        except AttributeError:
            raise AttributeError(f'Input {self.u.name} has no idx, likey a modeling error.')

        is_empty = len(ref) == 0
        if is_empty:
            raise ValueError(f'{indexer.model} contains no input, likey a data error.')

        is_subset = set(ref).issubset(set(uidx))
        if not is_subset:
            raise ValueError(f'{indexer.model} contains undefined {indexer.src}, likey a data error.')

        row, col = np.meshgrid(uidx, ref)
        out = (row == col).astype(int)
        if self.gamma:
            vgamma = getattr(self.rtn, self.gamma)
            out = vgamma.v[:, np.newaxis] * out
        return out


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
    sparse: bool, optional
        True to return output as scipy csr_matrix.
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
                 no_parse: bool = False,
                 sparse: bool = False,
                 **kwargs):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=None, rfun=rfun, rargs=rargs,
                         no_parse=no_parse, sparse=sparse,
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
    sparse: bool, optional
        True to return output as scipy csr_matrix.
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
                 no_parse: bool = False,
                 sparse: bool = False,):
        super().__init__(name=name, tex_name=tex_name, unit=unit,
                         info=info, vtype=vtype,
                         u=u, fun=None, rfun=rfun, rargs=rargs,
                         no_parse=no_parse, sparse=sparse,)

    @property
    def v0(self):
        return self.v

    @property
    def v1(self):
        return self.v

    @property
    def v(self):
        nr = self.u.horizon.n
        out = np.eye(nr, nr-1, k=-1) - np.eye(nr, nr-1, k=0)
        if self.sparse:
            return spr.csr_matrix(out)
        return out
