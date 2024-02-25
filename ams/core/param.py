"""
Base class for parameters.
"""


import logging

from typing import Optional, Iterable

import numpy as np
from scipy.sparse import issparse

from andes.core import BaseParam, DataParam, IdxParam, NumParam, ExtParam  # NOQA

from ams.opt.omodel import Param

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
        Indexer of the parameter.
    imodel : str, optional
        Name of the owner model or group of the indexer.
    no_parse: bool, optional
        True to skip parsing the parameter.
    nonneg: bool, optional
        True to set the parameter as non-negative.
    nonpos: bool, optional
        True to set the parameter as non-positive.
    complex: bool, optional
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
                 indexer: Optional[str] = None,
                 imodel: Optional[str] = None,
                 expand_dims: Optional[int] = None,
                 no_parse: Optional[bool] = False,
                 nonneg: Optional[bool] = False,
                 nonpos: Optional[bool] = False,
                 complex: Optional[bool] = False,
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
                       complex=complex, imag=imag, symmetric=symmetric,
                       diag=diag, hermitian=hermitian, boolean=boolean,
                       integer=integer, pos=pos, neg=neg, sparse=sparse)
        self.name = name
        self.tex_name = tex_name if (tex_name is not None) else name
        self.info = info
        self.src = name if (src is None) else src
        self.unit = unit
        self.is_group = False
        self.model = model  # name of a group or model
        self.indexer = indexer  # name of the indexer
        self.imodel = imodel  # name of a group or model of the indexer
        self.expand_dims = expand_dims
        self.no_parse = no_parse
        self.owner = None  # instance of the owner model or group
        self.rtn = None  # instance of the owner routine
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
        if self.indexer is None:
            if self.is_ext:
                if issparse(self._v):
                    out = self._v.toarray()
                else:
                    out = self._v
            elif self.is_group:
                out = self.owner.get(src=self.src, attr='v',
                                     idx=self.owner.get_idx())
            else:
                src_param = getattr(self.owner, self.src)
                out = getattr(src_param, 'v')
        else:
            try:
                imodel = getattr(self.rtn.system, self.imodel)
            except AttributeError:
                msg = f'Indexer source model <{self.imodel}> not found, '
                msg += 'likely a modeling error.'
                raise AttributeError(msg)
            try:
                sorted_idx = self.owner.find_idx(keys=self.indexer, values=imodel.get_idx())
            except AttributeError:
                sorted_idx = self.owner.idx.v
            except Exception as e:
                raise e
            model = getattr(self.rtn.system, self.model)
            out = model.get(src=self.src, attr='v', idx=sorted_idx)
        if self.expand_dims is not None:
            out = np.expand_dims(out, axis=self.expand_dims)
        return out

    @property
    def shape(self):
        """
        Return the shape of the parameter.
        """
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
        """
        if self.indexer is None:
            if self.is_group:
                return self.owner.get_idx()
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
                sorted_idx = self.owner.find_idx(keys=self.indexer, values=imodel.get_idx())
            except AttributeError:
                msg = f'Indexer <{self.indexer}> not found in <{self.imodel}>, '
                msg += 'likely a modeling error.'
                raise AttributeError(msg)
            return sorted_idx
