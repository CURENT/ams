import logging

import numpy as np

from andes.models.group import GroupBase as andes_GroupBase
from andes.core.service import BackRef
from andes.utils.func import validate_keys_values

from ams.shared import pd

logger = logging.getLogger(__name__)


class GroupBase(andes_GroupBase):
    """
    Base class for groups.

    Add common_vars and common_params to the group class.

    This class is revised from ``andes.models.group.GroupBase``.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('idx',))

    def get_idx(self):
        """
        Return the value of group idx sorted in a human-readable style.

        Notes
        -----
        This function sorts the idx values using a custom sorting key,
        which handles varying length strings with letters and numbers.
        """
        all_idx = [mdl.idx.v for mdl in self.models.values()]

        # Custom sorting function to handle varying length strings with letters and numbers
        def custom_sort_key(item):
            try:
                return int(item)  # Try to convert to integer for pure numeric strings
            except ValueError:
                try:
                    # Extract numeric part for strings with letters and numbers
                    return int(''.join(filter(str.isdigit, item)))
                except ValueError:
                    return item  # Return as is if not numeric

        group_idx = [sorted(mdl_idx, key=custom_sort_key) for mdl_idx in all_idx]
        flat_list = [item for sublist in group_idx for item in sublist]
        return flat_list

    def _check_src(self, src: str):
        """
        Helper function for checking if ``src`` is a shared field.

        The requirement is not strictly enforced and is only for debugging purposed.

        Disable debug logging in dispath modeling.
        """
        if src not in self.common_vars + self.common_params:
            logger.debug(f'Group <{self.class_name}> does not share property <{src}>.')

    def __repr__(self):
        dev_text = 'device' if self.n == 1 else 'devices'
        return f'{self.class_name} ({self.n} {dev_text}) at {hex(id(self))}'

    def __setattr__(self, key, value):
        if hasattr(value, 'owner'):
            if value.owner is None:
                value.owner = self
        if hasattr(value, 'name'):
            if value.name is None:
                value.name = key

        if isinstance(value, BackRef):
            self.services_ref[key] = value

        super().__setattr__(key, value)

    def set_backref(self, name, from_idx, to_idx):
        """
        Set idxes to ``BackRef``, and set them to models.
        """

        uid = self.idx2uid(to_idx)
        self.services_ref[name].v[uid].append(from_idx)

        model = self.idx2model(to_idx)
        model.set_backref(name, from_idx, to_idx)

    def alter(self, src, idx, value, attr='v'):
        """
        Alter values of input parameters or constant service for a group of models.

        .. note::
            New in version 0.9.14. Duplicate of `andes.models.group.GroupBase.alter`.

        Parameters
        ----------
        src : str
            The parameter name to alter
        idx : str, float, int
            The unique identifier for the device to alter
        value : float
            The desired value
        attr : str, optional
            The attribute to alter. Default is 'v'.
        """
        self._check_src(src)
        self._check_idx(idx)

        idx, _ = self._1d_vectorize(idx)
        models = self.idx2model(idx)

        if isinstance(value, (str, int, float, np.integer, np.floating)):
            value = [value] * len(idx)

        for mdl, ii, val in zip(models, idx, value):
            mdl.alter(src, ii, val, attr=attr)

        return True

    def as_dict(self, vin=False):
        """
        Export group common parameters as a dictionary.

        .. note::
            New in version 0.9.14. Duplicate of `andes.models.group.GroupBase.as_dict`.

        This method returns a dictionary where the keys are the `Model` parameter names
        and the values are array-like structures containing the data in the order they were added.
        Unlike `Model.as_dict()`, this dictionary does not include the `uid` field.

        Parameters
        ----------
        vin : bool, optional
            If True, includes the `vin` attribute in the dictionary. Default is False.

        Returns
        -------
        dict
            A dictionary of common parameters.
        """
        out_all = []
        out_params = self.common_params.copy()
        out_params.insert(2, 'idx')

        for mdl in self.models.values():
            if mdl.n <= 0:
                continue
            mdl_data = mdl.as_df(vin=True) if vin else mdl.as_dict()
            mdl_dict = {k: mdl_data.get(k) for k in out_params if k in mdl_data}
            out_all.append(mdl_dict)

        if not out_all:
            return {}

        out = {key: np.concatenate([item[key] for item in out_all]) for key in out_all[0].keys()}
        return out

    def as_df(self, vin=False):
        """
        Export group common parameters as a `pandas.DataFrame` object.

        .. note::
            New in version 0.9.14. Duplicate of `andes.models.group.GroupBase.as_df`.

        Parameters
        ----------
        vin : bool
            If True, export all parameters from original input (``vin``).

        Returns
        -------
        DataFrame
            A dataframe containing all model data. An `uid` column is added.
        """
        return pd.DataFrame(self.as_dict(vin=vin))

    def find_idx(self, keys, values, allow_none=False, default=None, allow_all=False):
        """
        Find indices of devices that satisfy the given `key=value` condition.

        This method iterates over all models in this group.

        .. note::
            New in version 0.9.14. Duplicate of `andes.models.group.GroupBase.find_idx`.

        Parameters
        ----------
        keys : str, array-like, Sized
            A string or an array-like of strings containing the names of parameters for the search criteria.
        values : array, array of arrays, Sized
            Values for the corresponding key to search for. If keys is a str, values should be an array of
            elements. If keys is a list, values should be an array of arrays, each corresponding to the key.
        allow_none : bool, optional
            Allow key, value to be not found. Used by groups. Default is False.
        default : bool, optional
            Default idx to return if not found (missing). Default is None.
        allow_all : bool, optional
            Return all matches if set to True. Default is False.

        Returns
        -------
        list
            Indices of devices.
        """

        keys, values = validate_keys_values(keys, values)

        n_mdl, n_pair = len(self.models), len(values[0])

        indices_found = []
        # `indices_found` contains found indices returned from all models of this group
        for model in self.models.values():
            indices_found.append(model.find_idx(keys, values, allow_none=True, default=default, allow_all=True))

        # --- find missing pairs ---
        i_val_miss = []
        for i in range(n_pair):
            idx_cross_mdls = [indices_found[j][i] for j in range(n_mdl)]
            if all(item == [default] for item in idx_cross_mdls):
                i_val_miss.append(i)

        if (not allow_none) and i_val_miss:
            miss_pairs = []
            for i in i_val_miss:
                miss_pairs.append([values[j][i] for j in range(len(keys))])
            raise IndexError(f'{keys} = {miss_pairs} not found in {self.class_name}')

        # --- output ---
        out_pre = []
        for i in range(n_pair):
            idx_cross_mdls = [indices_found[j][i] for j in range(n_mdl)]
            if all(item == [default] for item in idx_cross_mdls):
                out_pre.append([default])
                continue
            for item in idx_cross_mdls:
                if item != [default]:
                    out_pre.append(item)
                    break

        if allow_all:
            out = out_pre
        else:
            out = [item[0] for item in out_pre]

        return out


class Undefined(GroupBase):
    """
    The undefined group. Holds models with no ``group``.
    """

    def __init__(self):
        super().__init__()


class ACTopology(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_vars.extend(('a', 'v'))


class RenGen(GroupBase):
    """
    Renewable generator (converter) group.

    See ANDES Documentation SynGen here for the notes on replacing StaticGen and setting the power
    ratio parameters.

    Reference:

    [1] ANDES Documentation, RenGen, [Online]

    Available:

    https://docs.andes.app/en/latest/groupdoc/RenGen.html#rengen
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'gen', 'Sn', 'q0'))
        self.common_vars.extend(('Pe', 'Qe'))


class VSG(GroupBase):
    """
    Renewable generator with virtual synchronous generator (VSG) control group.

    Note that this is a group separate from ``RenGen`` for VSG scheduling study,
    and there is not a group 'VSG' in ANDES.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'gen', 'Sn'))
        self.common_vars.extend(('Pe', 'Qe'))


class DG(GroupBase):
    """
    Distributed generation (small-scale).

    See ANDES Documentation SynGen here for the notes on replacing StaticGen and setting the power
    ratio parameters.

    Reference:

    [1] ANDES Documentation, SynGen, [Online]

    Available:

    https://docs.andes.app/en/latest/groupdoc/SynGen.html#syngen
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'fn'))


class Cost(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params.extend(('gen',))


class Collection(GroupBase):
    """
    Collection of topology models
    """

    def __init__(self):
        super().__init__()


class Horizon(GroupBase):
    """
    Time horizon group.
    """

    def __init__(self):
        super().__init__()


class Reserve(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params.extend(('zone',))


class StaticGen(GroupBase):
    """
    Generator group.

    Notes
    -----
    For co-simulation with ANDES, check
    `ANDES StaticGen <https://docs.andes.app/en/latest/groupdoc/StaticGen.html#staticgen>`_
    for replacing static generators with dynamic generators.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'Sn', 'Vn', 'p0', 'q0', 'ra', 'xs', 'subidx',
                                   'pmax', 'pmin', 'pg0', 'ctrl', 'R10', 'td1', 'td2',
                                   'zone'))
        self.common_vars.extend(('p', 'q'))


class ACLine(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus1', 'bus2', 'r', 'x'))


class ACShort(GroupBase):
    def __init__(self):
        super(ACShort, self).__init__()
        self.common_params.extend(('bus1', 'bus2'))


class StaticLoad(GroupBase):
    """
    Static load group.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'p0', 'q0', 'ctrl', 'zone'))


class StaticShunt(GroupBase):
    """
    Static shunt compensator group.
    """
    pass


class Information(GroupBase):
    """
    Group for information container models.
    """

    def __init__(self):
        GroupBase.__init__(self)
        self.common_params = []
