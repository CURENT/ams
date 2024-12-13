from typing import Iterable, Sized

from ams.shared import np


def validate_keys_values(keys, values):
    """
    Validate the inputs for the func `find_idx`.

    :note:
        New in version 0.9.14. Duplicate from `ams.utils.func.validate_keys_values`.

    Parameters
    ----------
    keys : str, array-like, Sized
        A string or an array-like of strings containing the names of parameters for the search criteria.
    values : array, array of arrays, Sized
        Values for the corresponding key to search for. If keys is a str, values should be an array of
        elements. If keys is a list, values should be an array of arrays, each corresponds to the key.

    Returns
    -------
    tuple
        Sanitized keys and values

    Raises
    ------
    ValueError
        If the inputs are not valid.
    """
    if isinstance(keys, str):
        keys = (keys,)
        if not isinstance(values, (int, float, str, np.floating)) and not isinstance(values, Iterable):
            raise ValueError(f"value must be a string, scalar or an iterable, got {values}")

        if len(values) > 0 and not isinstance(values[0], (list, tuple, np.ndarray)):
            values = (values,)

    elif isinstance(keys, Sized):
        if not isinstance(values, Iterable):
            raise ValueError(f"value must be an iterable, got {values}")

        if len(values) > 0 and not isinstance(values[0], Iterable):
            raise ValueError(f"if keys is an iterable, values must be an iterable of iterables. got {values}")

        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length")

        if isinstance(values[0], Iterable):
            if not all([len(val) == len(values[0]) for val in values]):
                raise ValueError("All items in values must have the same length")

    return keys, values
