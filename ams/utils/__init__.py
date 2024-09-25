from andes.utils.misc import elapsed

from ams.utils import paths  # NOQA
from ams.utils.paths import get_case  # NOQA


def timer(func):
    def wrapper(*args, **kwargs):
        t0, _ = elapsed()
        result = func(*args, **kwargs)
        _, s = elapsed(t0)
        return result, s
    return wrapper


def create_entry(*fields, three_params=True):
    """
    Helper function to create a list of fields for a model entry.

    Parameters
    ----------
    fields : tuple
        Additional fields to include in the list.
    three_params : bool, optional
        Whether to include 'idx', 'u', 'name' in the list (default is True).

    Returns
    -------
    list
        A list of fields for the model entry.
    """
    base_fields = ['idx', 'u', 'name'] if three_params else []
    return base_fields + list(fields)
