import logging
import importlib
import inspect
import copy
from collections import OrderedDict

import numpy as np

from andes.models.group import GroupBase as andes_GroupBase

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
        self.common_params.extend(('idx',))

    def combine(self):
        """
        Summarize the group.
        """
        pass

    # TODO: revise ``set`` method to make sure Group.params are
    # also updated if model.params are updated

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
        Overwrites the combine function of the base class.

        Combines the common parameters and variables from the models within a group
        and sets them as attributes of the group.
        """
        

        # mdl_list = list(self.models.values())

        # # FIXME: there is an issue that the alteration on group params will not be reflected on the model params
        # # --- combine the common parameters ---
        # # FIXME: this list is kind of hard-coded, improve it later on
        # param_attr_array_list = ['v', 'vin', 'pu_coeff']  # a list of attributes of an param that are np.ndarray

        # for param_name in self.common_params:
        #     # logger.debug(f'Combining Param {param_name}')

        #     varray_dict = {}
        #     for attr_name in param_attr_array_list:
        #         varray_dict[attr_name] = {mdl.class_name: getattr(
        #             mdl, param_name).__dict__.get(attr_name) for mdl in mdl_list}

        #     # --- Type Consistency Check ---
        #     types = [type(getattr(model, param_name)) for model in mdl_list]
        #     is_identical = all(x == types[0] for x in types)
        #     if not is_identical:
        #         logger.warning(f'Parameter {param_name} has different types within the group.')
        #     # --- Type Consistency Check end ---

        #     # loop through the models and collect the array attributes of the param
        #     for mdl in mdl_list:
        #         param = getattr(mdl, param_name)
        #         for attr_name in param_attr_array_list:
        #             if hasattr(param, attr_name):
        #                 varray_dict[attr_name][mdl.class_name] = getattr(param, attr_name)

        #     # Filter out attributes that are not in the param
        #     varray_dict_valid = OrderedDict((key, val)
        #                                     for key, val in varray_dict.items() if key in param.__dict__)
        #     out_dict = OrderedDict((key, np.concatenate(list(val.values()))) for key, val in varray_dict_valid.items())
        #     logger.debug(f'out_dict {out_dict}')
        #     out_param = copy.copy(param)
        #     for attr_name, attr_val in out_dict.items():
        #         setattr(out_param, attr_name, attr_val)

        #     # set the combined param as the group's attribute
        #     setattr(self, param_name, out_param)
        #     self.params[param_name] = out_param

        # --- combine the common variables ---
        # var_attr_array_list = ['v']  # a list of attributes of an var that are np.ndarray

        # for var_name in self.common_vars:
        #     logger.debug(f'Combining Var {var_name}')

        #     varray_dict = {}
        #     for attr_name in var_attr_array_list:
        #         varray_dict[attr_name] = {mdl.class_name: getattr(
        #             mdl, param_name).__dict__.get(attr_name) for mdl in mdl_list}

        #     # --- Type Consistency Check ---
        #     types = [type(getattr(model, param_name)) for model in mdl_list]
        #     is_identical = all(x == types[0] for x in types)
        #     if not is_identical:
        #         logger.warning(f'Parameter {param_name} has different types within the group.')
        #     # --- Type Consistency Check end ---

        #     # loop through the models and collect the array attributes of the param
        #     for mdl in mdl_list:
        #         var = getattr(mdl, var_name)
        #         for attr_name in var_attr_array_list:
        #             if hasattr(var, attr_name):
        #                 varray_dict[attr_name][mdl.class_name] = getattr(var, attr_name)

        #     # Filter out attributes that are not in the param
        #     varray_dict_valid = OrderedDict((key, val)
        #                                     for key, val in varray_dict.items() if key in var.__dict__)
        #     out_dict = OrderedDict((key, np.concatenate(list(val.values()))) for key, val in varray_dict_valid.items())
        #     out_dict.update({'owner': self})
        #     out_var = copy.copy(var)
        #     for attr_name, attr_val in out_dict.items():
        #         setattr(var, attr_name, attr_val)

        #     # set the combined param as the group's attribute
        #     setattr(self, var_name, var)
        #     self.algebs[var_name] = var


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
