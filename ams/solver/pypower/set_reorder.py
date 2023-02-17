"""
Assigns B to A with one of the dimensions of A indexed.
"""

from numpy import ndim


def set_reorder(A, B, idx, dim=0):
    """Assigns B to A with one of the dimensions of A indexed.

    @return: A after doing A(:, ..., :, IDX, :, ..., :) = B
    where DIM determines in which dimension to place the IDX.

    @see: L{get_reorder}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    A = A.copy()
    ndims = ndim(A)
    A = A.astype(B.dtype)
    if ndims ==  1:
        A[idx] = B
    elif ndims == 2:
        if dim == 0:
            A[idx, :] = B
        elif dim == 1:
            A[:, idx] = B
        else:
            raise ValueError('dim (%d) may be 0 or 1' % dim)
    else:
        raise ValueError('number of dimensions (%d) may be 1 or 2' % dim)

    return A
