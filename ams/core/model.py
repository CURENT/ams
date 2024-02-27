"""
Module for Model class.
"""

import logging
from collections import OrderedDict
from typing import Iterable

import numpy as np
from andes.core.common import Config
from andes.utils.func import list_flatten

from ams.core.documenter import Documenter
from ams.core.var import Algeb
from ams.core.param import ExtParam
from ams.core.service import BaseService, BackRef

logger = logging.getLogger(__name__)


class Model:
    """
    Base class for power system dispatch models.

    This class is revised from ``andes.core.model.Model``.
    """

    def __init__(self, system=None, config=None):

        # --- Model ---
        self.system = system
        self.group = 'Undefined'

        self.algebs = OrderedDict()  # internal algebraic variables
        self.vars_decl_order = OrderedDict()  # variable in the order of declaration

        self.params_ext = OrderedDict()  # external parameters

        self.services = OrderedDict()  # service/temporary variables
        self.services_ref = OrderedDict()  # BackRefs
        self.services_sum = OrderedDict()  # VarSums

        self.config = Config(name=self.class_name)  # `config` that can be exported
        if config is not None:
            self.config.load(config)

        # basic configs
        self.config.add(OrderedDict((('allow_adjust', 1),
                                    ('adjust_lower', 0),
                                    ('adjust_upper', 1),
                                     )))
        self.docum = Documenter(self)

        # TODO: duplicate from ANDES, disable for now
        # self.syms = SymProcessor(self)  # symbolic processor instance
        # self.docum = Documenter(self)

    def _all_vars(self):
        """
        An OrderedDict of States, ExtStates, Algebs, ExtAlgebs
        """
        return OrderedDict(list(self.algebs.items()))

    def _check_attribute(self, key, value):
        """
        Check the attribute pair for valid names while instantiating the class.

        This function assigns `owner` to the model itself, assigns the name and tex_name.
        """
        if isinstance(value, (Algeb, BaseService)):
            if not value.owner:
                value.owner = self
            if not value.name:
                value.name = key
            if not value.tex_name:
                value.tex_name = key
            if key in self.__dict__:
                logger.warning(f"{self.class_name}: redefinition of member <{key}>. Likely a modeling error.")

    def __setattr__(self, key, value):
        """
        Overload the setattr function to register attributes.

        Parameters
        ----------
        key : str
            name of the attribute
        value : [Algeb]
            value of the attribute
        """
        self._check_attribute(key, value)
        self._register_attribute(key, value)

        super(Model, self).__setattr__(key, value)

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to the model instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        Block attributes will be exported and registered recursively.
        """
        if isinstance(value, Algeb):
            self.algebs[key] = value
        elif isinstance(value, ExtParam):
            self.params_ext[key] = value
        elif isinstance(value, BackRef):
            self.services_ref[key] = value
            self.services[key] = value

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    def list2array(self):
        """
        Convert all the value attributes ``v`` to NumPy arrays.

        Value attribute arrays should remain in the same address afterwards.
        Namely, all assignments to value array should be operated in place (e.g., with [:]).
        """

        for instance in self.num_params.values():
            instance.to_array()

    def set_backref(self, name, from_idx, to_idx):
        """
        Helper function for setting idx-es to ``BackRef``.
        """

        if name not in self.services_ref:
            return

        uid = self.idx2uid(to_idx)
        self.services_ref[name].v[uid].append(from_idx)

    def get(self, src: str, idx, attr: str = 'v', allow_none=False, default=0.0):
        """
        Get the value of an attribute of a model property.

        The return value is ``self.<src>.<attr>[idx]``

        Parameters
        ----------
        src : str
            Name of the model property
        idx : str, int, float, array-like
            Indices of the devices
        attr : str, optional, default='v'
            The attribute of the property to get.
            ``v`` for values, ``a`` for address, and ``e`` for equation value.
        allow_none : bool
            True to allow None values in the indexer
        default : float
            If `allow_none` is true, the default value to use for None indexer.

        Returns
        -------
        array-like
            ``self.<src>.<attr>[idx]``

        """
        uid = self.idx2uid(idx)
        if isinstance(self.__dict__[src].__dict__[attr], list):
            if isinstance(uid, Iterable):
                if not allow_none and (uid is None or None in uid):
                    raise KeyError('None not allowed in uid/idx. Enable through '
                                   '`allow_none` and provide a `default` if needed.')
                return [self.__dict__[src].__dict__[attr][i] if i is not None else default
                        for i in uid]
        # FIXME: this seems to be an unexpected case originted from ANDES
        if isinstance(uid, Iterable):
            if None in uid:
                return [self.__dict__[src].__dict__[attr][i] if i is not None else default
                        for i in uid]
        return self.__dict__[src].__dict__[attr][uid]

    def set(self, src, idx, attr, value):
        """
        Set the value of an attribute of a model property.

        Performs ``self.<src>.<attr>[idx] = value``. This method will not modify
        the input values from the case file that have not been converted to the
        system base. As a result, changes applied by this method will not affect
        the dumped case file.

        To alter parameters and reflect it in the case file, use :meth:`alter`
        instead.

        Parameters
        ----------
        src : str
            Name of the model property
        idx : str, int, float, array-like
            Indices of the devices
        attr : str, optional, default='v'
            The internal attribute of the property to get.
            ``v`` for values, ``a`` for address, and ``e`` for equation value.
        value : array-like
            New values to be set

        Returns
        -------
        bool
            True when successful.
        """
        uid = self.idx2uid(idx)
        self.__dict__[src].__dict__[attr][uid] = value
        return True

    def alter(self, src, idx, value):
        """
        Alter values of input parameters or constant service.

        If the method operates on an input parameter, the new data should be in
        the same base as that in the input file. This function will convert the
        new value to per unit in the system base.

        The values for storing the input data, i.e., the ``vin`` field of the
        parameter, will be overwritten, thus the update will be reflected in the
        dumped case file.

        Parameters
        ----------
        src : str
            The parameter name to alter
        idx : str, float, int
            The device to alter
        value : float
            The desired value
        """
        instance = self.__dict__[src]

        if hasattr(instance, 'vin') and (instance.vin is not None):
            self.set(src, idx, 'vin', value)
            instance.v[:] = instance.vin * instance.pu_coeff
        else:
            self.set(src, idx, 'v', value)

    def idx2uid(self, idx):
        """
        Convert idx to the 0-indexed unique index.

        Parameters
        ----------
        idx : array-like, numbers, or str
            idx of devices

        Returns
        -------
        list
            A list containing the unique indices of the devices
        """
        if idx is None:
            logger.debug("idx2uid returned None for idx None")
            return None
        if isinstance(idx, (float, int, str, np.integer, np.floating)):
            return self._one_idx2uid(idx)
        elif isinstance(idx, Iterable):
            if len(idx) > 0 and isinstance(idx[0], (list, np.ndarray)):
                idx = list_flatten(idx)
            return [self._one_idx2uid(i) if i is not None else None
                    for i in idx]
        else:
            raise NotImplementedError(f'Unknown idx type {type(idx)}')

    def _one_idx2uid(self, idx):
        """
        Helper function for checking if an idx exists and
        converting it to uid.
        """

        if idx not in self.uid:
            raise KeyError("<%s>: device not exist with idx=%s." %
                           (self.class_name, idx))

        return self.uid[idx]

    def doc(self, max_width=78, export='plain'):
        """
        Retrieve model documentation as a string.
        """
        return self.docum.get(max_width=max_width, export=export)

    def get_idx(self):
        """
        Return the index of the model instance.
        Equivalent to ``self.idx.v``, develoepd for consistency with group method
        ``get_idx``.
        """
        return self.idx.v

    def __repr__(self):
        dev_text = 'device' if self.n == 1 else 'devices'

        return f'{self.class_name} ({self.n} {dev_text}) at {hex(id(self))}'
