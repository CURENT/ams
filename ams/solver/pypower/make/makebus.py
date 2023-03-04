"""
Builds the vector of complex bus power injections.
"""
import logging

from numpy import ones, flatnonzero as find
from scipy.sparse import csr_matrix as sparse

import ams.solver.pypower.idx.constants as const

logger = logging.getLogger(__name__)

def makeSbus(baseMVA, bus, gen):
    """
    Builds the vector of complex bus power injections.
    
    Parameters
    ----------
    baseMVA : float
        Base MVA.
    bus : NumPy.array
        Bus data.
    gen : NumPy.array
        Generator data.
    
    Returns
    -------
    Sbus : NumPy.array
        Complex bus power injections.
    """
    ## generator info
    on = find(gen[:, const.gen['GEN_STATUS']] > 0)      ## which generators are on?
    gbus = gen[on, const.gen['GEN_BUS']]                   ## what buses are they at?

    ## form net complex bus power injection vector
    nb = bus.shape[0]
    ngon = on.shape[0]
    ## connection matrix, element i, j is 1 if gen on(j) at bus i is ON
    Cg = sparse((ones(ngon), (gbus, range(ngon))), (nb, ngon))

    ## power injected by gens plus power injected by loads converted to p.u.
    Sbus = ( Cg * (gen[on, const.gen['PG']] + 1j * gen[on, const.gen['QG']]) -
             (bus[:, const.bus['PD']] + 1j * bus[:, const.bus['QD']]) ) / baseMVA

    return Sbus


def makeSbus(baseMVA, bus, gen):
    """
    Builds the vector of complex bus power injections in p.u.,
    that is, generation minus load.
    
    Parameters
    ----------
    baseMVA : float
        Base MVA.
    bus : NumPy.array
        Bus data.
    gen : NumPy.array
        Generator data.

    Returns
    -------
    Sbus : NumPy.array
        Complex bus power injections.
    """
    ## generator info
    on = find(gen[:, const.gen['GEN_STATUS']] > 0)      ## which generators are on?
    gbus = gen[on, const.gen['GEN_BUS']]                   ## what buses are they at?

    ## form net complex bus power injection vector
    nb = bus.shape[0]
    ngon = on.shape[0]
    ## connection matrix, element i, j is 1 if gen on(j) at bus i is ON
    Cg = sparse((ones(ngon), (gbus, range(ngon))), (nb, ngon))

    ## power injected by gens plus power injected by loads converted to p.u.
    Sbus = ( Cg * (gen[on, const.gen['PG']] + 1j * gen[on, const.gen['QG']]) -
             (bus[:, const.bus['PD']] + 1j * bus[:, const.bus['QD']]) ) / baseMVA

    return Sbus

