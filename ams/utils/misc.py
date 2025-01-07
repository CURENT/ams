import warnings
import functools

from andes.utils.misc import elapsed


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


def pretty_long_message(message, prefix="", max_length=80):
    """
    Pretty print a long message.

    Parameters
    ----------
    message : str
        The message to format.
    prefix : str, optional
        A prefix to add to each line of the message.
    max_length : int, optional
        The maximum length of each line.

    Returns
    -------
    str
        The formatted message.
    """
    if len(message) <= max_length:
        return message
    else:
        lines = [message[i:i+max_length] for i in range(0, len(message), max_length)]
        formatted_message = lines[0] + "\n" + "\n".join([prefix + line for line in lines[1:]])
        return formatted_message


def deprecated(message="This function is deprecated and will be removed in a future version."):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


deprec_get_idx = deprecated(
    "get_idx() is deprecated and will be removed in a future version. Please use get_all_idxes() instead.")
