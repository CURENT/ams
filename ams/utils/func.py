"""
Miscellaneous utility functions for AMS.
"""

import math
import functools
import operator

import numpy as np

# ---------------------------------------------------------------------------
# Math constants
# ---------------------------------------------------------------------------

deg2rad = math.pi / 180
"""Degrees to radians conversion factor."""

rad2deg = 180 / math.pi
"""Radians to degrees conversion factor."""


# ---------------------------------------------------------------------------
# Array utilities
# ---------------------------------------------------------------------------

def list_flatten(input_list):
    """
    Flatten a multi-dimensional list into a flat 1-D list.

    Notes
    -----
    Adapted from ``andes.utils.func.list_flatten``.
    Original author: Hantao Cui. License: GPL-3.0.
    """
    if len(input_list) > 0 and isinstance(input_list[0], (list, np.ndarray)):
        return functools.reduce(operator.iconcat, input_list, [])
    return input_list


def safe_div(a, b, out=None):
    """
    Safe division for NumPy arrays. Division by zero yields zero (or *out*).

    Parameters
    ----------
    a : array-like
        Numerator.
    b : array-like
        Denominator.
    out : array-like or None, optional
        Default values where *b* == 0. If ``None``, zeros are used.

    Notes
    -----
    Cannot be used in ``e_str`` due to unsupported Derivative.
    Adapted from ``andes.thirdparty.npfunc.safe_div``.
    """
    if out is None:
        out = np.zeros_like(a)
    return np.divide(a, b, out=out, where=(b != 0))


# ---------------------------------------------------------------------------
# String / list conversion
# ---------------------------------------------------------------------------

def str_list_iconv(x):
    """
    Convert a comma-separated string into a stripped list of strings.

    Parameters
    ----------
    x : str
        Comma-separated string, e.g. ``'a, b, c'``.

    Returns
    -------
    list of str

    Notes
    -----
    Adapted from ``andes.models.timeseries.str_list_iconv``.
    Original author: Hantao Cui. License: GPL-3.0.
    """
    if isinstance(x, str):
        x = x.split(',')
        return [item.strip() for item in x]
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Numpy operator helpers
# ---------------------------------------------------------------------------

def multiply_left_t(a, b):
    """
    Element-wise multiply ``a.T * b``.

    Lets callers compose a small left-transpose with a multiply when the
    natural shape of ``a`` is the transpose of what the broadcast rule
    needs against ``b`` — used by the per-(area, slot) reserve chain in
    :class:`SRBase` / :class:`NSRBase` so the v1.3.0 ``sd`` shape
    ``(narea, nslot)`` flows through without a NumOp(transpose) shim
    on the source side.
    """
    import numpy as np
    return np.multiply(np.transpose(a), b)
