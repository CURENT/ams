import logging
import importlib
import inspect
from collections import OrderedDict

import numpy as np

from andes.models.group import GroupBase as andes_GroupBase
from andes.core.param import BaseParam

from ams.core.var import Algeb

logger = logging.getLogger(__name__)


class GroupBase(andes_GroupBase):
    """
    Base class for groups.

    Add common_vars and common_params to the group class.

    This class is revised from ``andes.models.group.GroupBase``.
    """
    pass

    def __init__(self):
        super().__init__()
        self.params = OrderedDict()
        self.algebs = OrderedDict()

    def combine(self):
        """
        Summarize the group.
        """
        pass
        logger.debug('Pseudo function GroupBase.combine() is called.')

        # msg = f'Group {self.class_name} has models: {self.models.keys()}, common vars: {self.common_vars}, common params: {self.common_params}'
        # logger.debug(msg)
        # # --- debug: summarize a group ---
        # n_mdls = len(self.models)
        # mname_list = list(self.models.keys())
        # logger.debug(mname_list)
        # for mname, mdl in self.models.items():
        #     # --- common_params ---
        #     n_cparams = len(self.common_params)
        #     if n_cparams > 0:
        #         attrs = {}
        #         # gongjiaoche here to collect all
        #         # v_list
        #         # vin_list
        #         for key in self.common_params:
        #             value = getattr(mdl, key)
        #             if isinstance(value, BaseParam):
        #                 self.params[key] = value
        #                 attr_list = [attr for attr in dir(value) if not attr.startswith('_') and not callable(getattr(value, attr))]
        #                 attrs[mname] = {key: getattr(value, key) for key in attr_list}
        #             # set gongjiaoche to group
        #             # setattr(self, key, value)
        #         logger.debug(f'attr_dict: {attrs}')
        #     # --- common_vars ---
        #     n_cvars = len(self.common_vars)
        #     if n_cvars > 0:
        #         for key in self.common_vars:
        #             value = getattr(mdl, key)
        #             if isinstance(value, Algeb):
        #                 self.algebs[key] = value


class Undefined(GroupBase):
    """
    The undefined group. Holds models with no ``group``.
    """
    pass


class ACTopology(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_vars.extend(('a', 'v'))


class Cost(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params.extend(('gen',))


class Collection(GroupBase):
    """Collection of topology models"""
    pass


class StaticGen(GroupBase):
    """
    Generator group.

    Notes
    -----
    For co-simulation with ANDES, replacing static generators with dynamic generators can be found in
    [ANDES StaticGen](https://docs.andes.app/en/latest/groupdoc/StaticGen.html#staticgen).
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('Sn', 'Vn', 'p0', 'q0', 'ra', 'xs', 'subidx'))
        self.common_vars.extend(('p', 'q'))

    def combine(self):
        """
        Overwrite the combine function of the base class.
        """
        pass
        out = {}
        module = importlib.import_module('andes.core.param')
        for pname in self.common_params:
            # NOTE: get the first model in the group
            mdl0 = list(self.models.values())[0]
            param = getattr(mdl0, pname)
            pattrs = [attr for attr in dir(param) if not attr.startswith('_') and not callable(getattr(param, attr))]
            pattr_kvs = {attr: getattr(param, attr) for attr in pattrs}
            param_class = getattr(module, param.__class__.__name__)
            # param_class = getattr(module, str())
            signature = inspect.signature(param_class)
            in_list = list(signature.parameters.keys())
            in_dict = {k: getattr(param, k) for k in in_list if k in pattrs}
            # FIXME: for non-value type: NOT check if input value consistentency, this might cause error
            # TODO: for value-type: modify v, vin, pu_coeff, n
            logger.debug(f'define an {param_class} named {pname}')
            logger.debug(f'input dict: {in_dict}')
            cparam = param_class(**in_dict)
            setattr(self, pname, cparam)
            self.params[pname] = cparam

class ACLine(GroupBase):
    def __init__(self):
        super(ACLine, self).__init__()
        self.common_params.extend(('bus1', 'bus2', 'r', 'x'))


class StaticLoad(GroupBase):
    """
    Static load group.
    """
    pass


class StaticShunt(GroupBase):
    """
    Static shunt compensator group.
    """
    pass


class DynLoad(GroupBase):
    """
    Dynamic load group.
    """
    pass


class Information(GroupBase):
    """
    Group for information container models.
    """

    def __init__(self):
        GroupBase.__init__(self)
        self.common_params = []
