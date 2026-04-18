import warnings
import functools
from decimal import Decimal, ROUND_DOWN
from time import time


def elapsed(t0=0.0):
    """
    Return the current time and the elapsed time from ``t0``.

    If ``t0`` is not given (default 0.0), returns the Unix epoch time and a
    string of the elapsed seconds from epoch (not useful on its own — pass
    the first return value as ``t0`` on a subsequent call).

    Parameters
    ----------
    t0 : float, optional
        Start time from a previous ``elapsed()`` call. Default 0.0.

    Returns
    -------
    t : float
        Current Unix time.
    s : str
        Elapsed time from ``t0`` formatted as ``'X.XXXX second(s)'``.

    Notes
    -----
    Adapted from ``andes.utils.misc.elapsed``.
    Original author: Hantao Cui. License: GPL-3.0.
    """
    t = time()
    dt = t - t0
    dt_sec = Decimal(str(dt)).quantize(Decimal('.0001'), rounding=ROUND_DOWN)
    s = str(dt_sec) + (' second' if dt_sec == 1 else ' seconds')
    return t, s


def is_interactive():
    """
    Return ``True`` if running inside an interactive Python/IPython shell.

    Notes
    -----
    Adapted from ``andes.utils.misc.is_interactive``.
    Original author: Hantao Cui. License: GPL-3.0.
    """
    ipython = False
    try:
        cls_name = get_ipython().__class__.__name__  # noqa: F821
        if cls_name in ('InteractiveShellEmbed', 'TerminalInteractiveShell'):
            ipython = True
    except NameError:
        pass

    import __main__ as main
    return not hasattr(main, '__file__') or ipython


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
