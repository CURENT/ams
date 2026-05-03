"""
Base class for parameters.
"""


import logging

from typing import Optional, Iterable

import numpy as np

from ams.opt import Param

from ams.utils.misc import deprec_get_idx

logger = logging.getLogger(__name__)


class RParam(Param):
    """
    Class for parameters used in a routine.
    This class is developed to simplify the routine definition.

    `RParm` is further used to define `Parameter` in the optimization model.

    `no_parse` is used to skip parsing the `RParam` in optimization
    model.
    It means that the `RParam` will not be added to the optimization model.
    This is useful when the RParam contains non-numeric values,
    or it is not necessary to be added to the optimization model.

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
    indexer : str, optional
        Primary-axis indexer of the parameter — name of an
        ``IdxParam`` on the row-owner model whose values get matched
        against ``imodel.get_all_idxes()`` to sort / position rows.
    imodel : str, optional
        Name of the owner model or group of the (primary) indexer.
    horizon : ams.core.param.RParam, optional
        Secondary-axis indexer. Mirrors the
        :attr:`ams.opt.var.Var.horizon` convention used for output
        Vars: when set together with ``indexer`` / ``imodel``, the
        param's :attr:`v` returns a 2D matrix shaped
        ``(imodel.n, horizon.n)`` built by pivoting the long-format
        rows on ``hindexer`` (secondary) and ``indexer`` (primary).
        Cells with no matching row fall back to the source
        ``NumParam.default``.
    hindexer : str, optional
        Name of the ``IdxParam`` on the row-owner model that carries
        the secondary key, matched against ``horizon.v``. Required
        when ``horizon`` is set; ignored otherwise.
    no_parse: bool, optional
        True to skip parsing the parameter.
    nonneg: bool, optional
        True to set the parameter as non-negative.
    nonpos: bool, optional
        True to set the parameter as non-positive.
    cplx: bool, optional
        True to set the parameter as complex.
    imag: bool, optional
        True to set the parameter as imaginary.
    symmetric: bool, optional
        True to set the parameter as symmetric.
    diag: bool, optional
        True to set the parameter as diagonal.
    hermitian: bool, optional
        True to set the parameter as hermitian.
    boolean: bool, optional
        True to set the parameter as boolean.
    integer: bool, optional
        True to set the parameter as integer.
    pos: bool, optional
        True to set the parameter as positive.
    neg: bool, optional
        True to set the parameter as negative.
    sparse: bool, optional
        True to set the parameter as sparse.

    Examples
    --------
    Example 1: Define a routine parameter from a source model or group.

    In this example, we define the parameter `cru` from the source model
    `SFRCost` with the parameter `cru`.
    Note since this parameter comes from model `SFRCost`, but it is used to
    multiply on generator output powers, we need to ensure the value is sorted
    in the same order as generators.
    `gen` is the indexer that comes from model `SFR` itself, and `imodel`
    is the indexer model, i.e., the model that has `idx` as its attribute.
    Then, we can ensure the value of cru is sorted in the same order as the
    indexer `StaticGen`.

    >>> self.cru = RParam(info='RegUp reserve coefficient',
    >>>                   tex_name=r'c_{r,u}',
    >>>                   unit=r'$/(p.u.)',
    >>>                   name='cru',
    >>>                   src='cru',
    >>>                   model='SFRCost',
    >>>                   indexer='gen',
    >>>                   imodel='StaticGen',
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
                 indexer: Optional[str] = None,
                 imodel: Optional[str] = None,
                 horizon: Optional['RParam'] = None,
                 hindexer: Optional[str] = None,
                 expand_dims: Optional[int] = None,
                 no_parse: Optional[bool] = False,
                 nonneg: Optional[bool] = False,
                 nonpos: Optional[bool] = False,
                 cplx: Optional[bool] = False,
                 imag: Optional[bool] = False,
                 symmetric: Optional[bool] = False,
                 diag: Optional[bool] = False,
                 hermitian: Optional[bool] = False,
                 boolean: Optional[bool] = False,
                 integer: Optional[bool] = False,
                 pos: Optional[bool] = False,
                 neg: Optional[bool] = False,
                 sparse: Optional[list] = None,
                 ):
        Param.__init__(self, nonneg=nonneg, nonpos=nonpos,
                       cplx=cplx, imag=imag, symmetric=symmetric,
                       diag=diag, hermitian=hermitian, boolean=boolean,
                       integer=integer, pos=pos, neg=neg, sparse=sparse)
        self.name = name
        self.tex_name = tex_name if (tex_name is not None) else name
        self.info = info
        self.src = src
        self.unit = unit
        self.is_group = False
        self.model = model  # name of a group or model
        self.indexer = indexer  # name of the indexer
        self.imodel = imodel  # name of a group or model of the indexer
        self.horizon = horizon  # secondary-axis RParam (mirrors Var.horizon)
        self.hindexer = hindexer  # secondary indexer column on row-owner model
        if horizon is not None and hindexer is None:
            raise ValueError(
                f"RParam <{name}>: 'hindexer' is required when "
                f"'horizon' is set (mirrors the Var.horizon pattern)."
            )
        self.expand_dims = expand_dims
        self.no_parse = no_parse
        self.owner = None  # instance of the owner model or group
        self.is_ext = False  # indicate if the value is set externally
        self._v = None  # external value
        if v is not None:
            self._v = v
            self.is_ext = True

    # FIXME: might need a better organization
    @property
    def v(self):
        """
        The value of the parameter.

        Notes
        -----
        - This property is a wrapper for the ``get`` method of the owner class.
        - The value will sort by the indexer if indexed, used for optmization modeling.
        """
        out = None
        if self.sparse and self.expand_dims is not None:
            msg = 'Sparse matrix does not support expand_dims.'
            raise NotImplementedError(msg)
        if self.is_ext:
            # User-supplied override always wins, regardless of
            # indexer / horizon kwargs.
            out = self._v
        elif self.indexer is None:
            if self.is_group:
                out = self.owner.get(src=self.src, attr='v',
                                     idx=self.owner.get_all_idxes())
            else:
                src_param = getattr(self.owner, self.src)
                out = getattr(src_param, 'v')
        elif self.horizon is not None:
            out = self._materialize_2d()
        else:
            try:
                imodel = getattr(self.rtn.system, self.imodel)
            except AttributeError:
                msg = f'Indexer source model <{self.imodel}> not found, '
                msg += 'likely a modeling error.'
                raise AttributeError(msg)
            try:
                sorted_idx = self.owner.find_idx(keys=self.indexer, values=imodel.get_all_idxes())
            except AttributeError:
                sorted_idx = self.owner.idx.v
            except Exception as e:
                raise e
            model = getattr(self.rtn.system, self.model)
            out = model.get(src=self.src, attr='v', idx=sorted_idx)
        if self.expand_dims is not None:
            out = np.expand_dims(out, axis=self.expand_dims)
        return out

    def _materialize_2d(self):
        """
        Pivot a long-format row-set into a 2D ``(primary, secondary)``
        matrix.

        Used when ``horizon`` is set — see the class docstring for the
        ``indexer`` / ``imodel`` / ``horizon`` / ``hindexer`` contract.
        Cells with no matching row default to ``NumParam.default``;
        duplicate ``(primary, secondary)`` rows raise.
        """
        if self.hindexer is None:
            raise ValueError(
                f"RParam <{self.name}>: 'hindexer' is not set but "
                f"'horizon' is — required for the 2D pivot. Set both "
                f"together (mirrors the Var.horizon pattern)."
            )
        try:
            primary_imodel = getattr(self.rtn.system, self.imodel)
        except AttributeError:
            msg = (f"RParam <{self.name}>: primary indexer source model "
                   f"<{self.imodel}> not found, likely a modeling error.")
            raise AttributeError(msg)

        primary_keys = list(primary_imodel.get_all_idxes())
        secondary_keys = list(np.asarray(self.horizon.v).tolist())
        nr, nc = len(primary_keys), len(secondary_keys)

        row_model = getattr(self.rtn.system, self.model)
        row_primary = list(getattr(row_model, self.indexer).v)
        row_secondary = list(getattr(row_model, self.hindexer).v)
        row_values = np.asarray(getattr(row_model, self.src).v)

        src_param = getattr(row_model, self.src)
        default = getattr(src_param, 'default', 0)
        dtype = row_values.dtype if row_values.size else np.float64
        out = np.full((nr, nc), default, dtype=dtype)

        seen = {}
        for i, (p, s) in enumerate(zip(row_primary, row_secondary)):
            key = (p, s)
            if key in seen:
                msg = (f"RParam <{self.name}>: duplicate row in "
                       f"<{self.model}> at (primary={p!r}, "
                       f"secondary={s!r}); each (primary, secondary) "
                       f"pair must appear at most once.")
                raise ValueError(msg)
            seen[key] = i

        primary_uid = {k: u for u, k in enumerate(primary_keys)}
        secondary_uid = {k: u for u, k in enumerate(secondary_keys)}
        for (p, s), i in seen.items():
            pu = primary_uid.get(p)
            su = secondary_uid.get(s)
            if pu is None or su is None:
                continue
            out[pu, su] = row_values[i]
        return out

    @property
    def shape(self):
        """
        Return the shape of the parameter.

        ``no_parse=True`` short-circuits to ``None`` only when the
        param has no resolvable value yet — for parsed-but-unwired
        2D params (e.g. ``ED.ug`` after the late-bind to
        ``EDSlotGen``), defer to ``np.shape(self.v)`` so the true
        shape is reported.
        """
        if self.is_ext:
            return np.shape(self._v)
        if self.no_parse and self.horizon is None:
            return None
        return np.shape(self.v)

    @property
    def dtype(self):
        """
        Return the data type of the parameter value.
        """
        if isinstance(self.v, (str, bytes)):
            return str
        elif isinstance(self.v, Iterable):
            return type(self.v[0])
        else:
            return type(self.v)

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
        owner = self.owner.__class__.__name__ if self.owner is not None else self.rtn.__class__.__name__
        postfix = '' if self.src is None else f'.{self.src}'
        return f'{self.__class__.__name__}: {owner}' + postfix

    @deprec_get_idx
    def get_idx(self):
        """
        Get the index of the parameter.

        Returns
        -------
        idx : list
            Index of the parameter.

        Notes
        -----
        - The value will sort by the indexer if indexed.

        .. deprecated:: 1.0.0
           Use ``get_all_idxes`` instead.
        """
        if self.indexer is None:
            if self.is_group:
                return self.owner.get_all_idxes()
            elif self.owner is None:
                logger.info(f'Param <{self.name}> has no owner.')
                return None
            elif hasattr(self.owner, 'idx'):
                return self.owner.idx.v
            else:
                logger.info(f'Param <{self.name}> owner <{self.owner.class_name}> has no idx.')
                return None
        else:
            try:
                imodel = getattr(self.rtn.system, self.imodel)
            except AttributeError:
                msg = f'Indexer source model <{self.imodel}> not found, '
                msg += 'likely a modeling error.'
                raise AttributeError(msg)
            try:
                sorted_idx = self.owner.find_idx(keys=self.indexer, values=imodel.get_all_idxes())
            except AttributeError:
                msg = f'Indexer <{self.indexer}> not found in <{self.imodel}>, '
                msg += 'likely a modeling error.'
                raise AttributeError(msg)
            return sorted_idx

    def get_all_idxes(self):
        """
        Get all the indexes of the parameter.

        Returns
        -------
        idx : list
            Index of the parameter.

        Notes
        -----
        - The value will sort by the indexer if indexed.

        .. versionadded:: 1.0.0
        """
        if self.indexer is None:
            if self.is_group:
                return self.owner.get_all_idxes()
            elif self.owner is None:
                logger.info(f'Param <{self.name}> has no owner.')
                return None
            elif hasattr(self.owner, 'idx'):
                return self.owner.idx.v
            else:
                logger.info(f'Param <{self.name}> owner <{self.owner.class_name}> has no idx.')
                return None
        else:
            try:
                imodel = getattr(self.rtn.system, self.imodel)
            except AttributeError:
                msg = f'Indexer source model <{self.imodel}> not found, '
                msg += 'likely a modeling error.'
                raise AttributeError(msg)
            try:
                sorted_idx = self.owner.find_idx(keys=self.indexer, values=imodel.get_all_idxes())
            except AttributeError:
                msg = f'Indexer <{self.indexer}> not found in <{self.imodel}>, '
                msg += 'likely a modeling error.'
                raise AttributeError(msg)
            return sorted_idx
