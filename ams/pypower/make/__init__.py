"""
Module for build matrices.
"""
import numpy as np  # NOQA
from random import random  # NOQA
from ams.pypower.make.matrices import (isload, hasPQcap,
                                      makeAang, makeApq, makeAvl, makeAy,
                                      makeB, makeBdc,
                                      makeLODF, makePTDF,
                                      makeSbus, makeYbus)  # NOQA
from ams.pypower.make.pdv import (dSbus_dV, dIbr_dV, dSbr_dV,
                                  d2Sbus_dV2, d2AIbr_dV2, d2ASbr_dV2,
                                  d2Ibr_dV2, d2Sbr_dV2, dAbr_dV)  # NOQA


def fairmax(x):
    """
    Same as built-in C{max}, except breaks ties randomly.

    Takes a vector as an argument and returns the same output as the
    built-in function C{max} with two output parameters, except that
    where the maximum value occurs at more than one position in the
    vector, the index is chosen randomly from these positions as opposed
    to just choosing the first occurance.

    @see: C{max}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    val = max(x)  # find max value
    i = np.nonzero(x == val)  # find all positions where this occurs
    n = len(i)  # number of occurences
    idx = i(np.fix(n * random()) + 1)  # select index randomly among occurances
    return val, idx
