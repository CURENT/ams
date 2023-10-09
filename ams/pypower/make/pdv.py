"""
Make partial derivatives matrices w.r.t. voltage.
"""

import logging  # NOQA

import numpy as np  # NOQA
import scipy.sparse as sp  # NOQA
from scipy.sparse import csr_matrix as c_sparse  # NOQA

from ams.pypower.idx import IDX  # NOQA

logger = logging.getLogger(__name__)


def dSbus_dV(Ybus, V):
    """
    Computes partial derivatives of power injection w.r.t. voltage.

    Returns two matrices containing partial derivatives of the complex bus
    power injections w.r.t voltage magnitude and voltage angle respectively
    (for all buses). If C{Ybus} is a c_sparse matrix, the return values will be
    also. The following explains the expressions used to form the matrices::

        S = diag(V) * conj(Ibus) = diag(conj(Ibus)) * V

    Partials of V & Ibus w.r.t. voltage magnitudes::
        dV/dVm = diag(V / abs(V))
        dI/dVm = Ybus * dV/dVm = Ybus * diag(V / abs(V))

    Partials of V & Ibus w.r.t. voltage angles::
        dV/dVa = j * diag(V)
        dI/dVa = Ybus * dV/dVa = Ybus * j * diag(V)

    Partials of S w.r.t. voltage magnitudes::
        dS/dVm = diag(V) * conj(dI/dVm) + diag(conj(Ibus)) * dV/dVm
               = diag(V) * conj(Ybus * diag(V / abs(V)))
                                        + conj(diag(Ibus)) * diag(V / abs(V))

    Partials of S w.r.t. voltage angles::
        dS/dVa = diag(V) * conj(dI/dVa) + diag(conj(Ibus)) * dV/dVa
               = diag(V) * conj(Ybus * j * diag(V))
                                        + conj(diag(Ibus)) * j * diag(V)
               = -j * diag(V) * conj(Ybus * diag(V))
                                        + conj(diag(Ibus)) * j * diag(V)
               = j * diag(V) * conj(diag(Ibus) - Ybus * diag(V))

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, "AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation", MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ib = range(len(V))

    if sp.issparse(Ybus):
        Ibus = Ybus * V

        diagV = c_sparse((V, (ib, ib)))
        diagIbus = c_sparse((Ibus, (ib, ib)))
        diagVnorm = c_sparse((V / abs(V), (ib, ib)))
    else:
        Ibus = Ybus * np.asmatrix(V).T

        diagV = np.asmatrix(np.diag(V))
        diagIbus = np.asmatrix(np.diag(np.asarray(Ibus).flatten()))
        diagVnorm = np.asmatrix(np.diag(V / abs(V)))

    dS_dVm = diagV * np.conj(Ybus * diagVnorm) + np.conj(diagIbus) * diagVnorm
    dS_dVa = 1j * diagV * np.conj(diagIbus - Ybus * diagV)

    return dS_dVm, dS_dVa


def dIbr_dV(branch, Yf, Yt, V):
    """Computes partial derivatives of branch currents w.r.t. voltage.

    Returns four matrices containing partial derivatives of the complex
    branch currents at "from" and "to" ends of each branch w.r.t voltage
    magnitude and voltage angle respectively (for all buses). If C{Yf} is a
    c_sparse matrix, the partial derivative matrices will be as well. Optionally
    returns vectors containing the currents themselves. The following
    explains the expressions used to form the matrices::

        If = Yf * V

    Partials of V, Vf & If w.r.t. voltage angles::
        dV/dVa  = j * diag(V)
        dVf/dVa = c_sparse(range(nl), f, j*V(f)) = j * c_sparse(range(nl), f, V(f))
        dIf/dVa = Yf * dV/dVa = Yf * j * diag(V)

    Partials of V, Vf & If w.r.t. voltage magnitudes::
        dV/dVm  = diag(V / abs(V))
        dVf/dVm = c_sparse(range(nl), f, V(f) / abs(V(f))
        dIf/dVm = Yf * dV/dVm = Yf * diag(V / abs(V))

    Derivations for "to" bus are similar.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    i = range(len(V))

    Vnorm = V / abs(V)

    if sp.issparse(Yf):
        diagV = c_sparse((V, (i, i)))
        diagVnorm = c_sparse((Vnorm, (i, i)))
    else:
        diagV = np.asmatrix(np.diag(V))
        diagVnorm = np.asmatrix(np.diag(Vnorm))

    dIf_dVa = Yf * 1j * diagV
    dIf_dVm = Yf * diagVnorm
    dIt_dVa = Yt * 1j * diagV
    dIt_dVm = Yt * diagVnorm

    # Compute currents.
    if sp.issparse(Yf):
        If = Yf * V
        It = Yt * V
    else:
        If = np.asarray(Yf * np.asmatrix(V).T).flatten()
        It = np.asarray(Yt * np.asmatrix(V).T).flatten()

    return dIf_dVa, dIf_dVm, dIt_dVa, dIt_dVm, If, It


def dSbr_dV(branch, Yf, Yt, V):
    """Computes partial derivatives of power flows w.r.t. voltage.

    returns four matrices containing partial derivatives of the complex
    branch power flows at "from" and "to" ends of each branch w.r.t voltage
    magnitude and voltage angle respectively (for all buses). If C{Yf} is a
    c_sparse matrix, the partial derivative matrices will be as well. Optionally
    returns vectors containing the power flows themselves. The following
    explains the expressions used to form the matrices::

        If = Yf * V;
        Sf = diag(Vf) * conj(If) = diag(conj(If)) * Vf

    Partials of V, Vf & If w.r.t. voltage angles::
        dV/dVa  = j * diag(V)
        dVf/dVa = c_sparse(range(nl), f, j*V(f)) = j * c_sparse(range(nl), f, V(f))
        dIf/dVa = Yf * dV/dVa = Yf * j * diag(V)

    Partials of V, Vf & If w.r.t. voltage magnitudes::
        dV/dVm  = diag(V / abs(V))
        dVf/dVm = c_sparse(range(nl), f, V(f) / abs(V(f))
        dIf/dVm = Yf * dV/dVm = Yf * diag(V / abs(V))

    Partials of Sf w.r.t. voltage angles::
        dSf/dVa = diag(Vf) * conj(dIf/dVa)
                        + diag(conj(If)) * dVf/dVa
                = diag(Vf) * conj(Yf * j * diag(V))
                        + conj(diag(If)) * j * c_sparse(range(nl), f, V(f))
                = -j * diag(Vf) * conj(Yf * diag(V))
                        + j * conj(diag(If)) * c_sparse(range(nl), f, V(f))
                = j * (conj(diag(If)) * c_sparse(range(nl), f, V(f))
                        - diag(Vf) * conj(Yf * diag(V)))

    Partials of Sf w.r.t. voltage magnitudes::
        dSf/dVm = diag(Vf) * conj(dIf/dVm)
                        + diag(conj(If)) * dVf/dVm
                = diag(Vf) * conj(Yf * diag(V / abs(V)))
                        + conj(diag(If)) * c_sparse(range(nl), f, V(f)/abs(V(f)))

    Derivations for "to" bus are similar.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, "AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation", MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # define
    f = branch[:, IDX.branch.F_BUS].astype(int)  # list of "from" buses
    t = branch[:, IDX.branch.T_BUS].astype(int)  # list of "to" buses
    nl = len(f)
    nb = len(V)
    il = np.arange(nl)
    ib = np.arange(nb)

    Vnorm = V / abs(V)

    if sp.issparse(Yf):
        # compute currents
        If = Yf * V
        It = Yt * V

        diagVf = c_sparse((V[f], (il, il)))
        diagIf = c_sparse((If, (il, il)))
        diagVt = c_sparse((V[t], (il, il)))
        diagIt = c_sparse((It, (il, il)))
        diagV = c_sparse((V, (ib, ib)))
        diagVnorm = c_sparse((Vnorm, (ib, ib)))

        shape = (nl, nb)
        # Partial derivative of S w.r.t voltage phase angle.
        dSf_dVa = 1j * (np.conj(diagIf) *
                        c_sparse((V[f], (il, f)), shape) - diagVf * np.conj(Yf * diagV))

        dSt_dVa = 1j * (np.conj(diagIt) *
                        c_sparse((V[t], (il, t)), shape) - diagVt * np.conj(Yt * diagV))

        # Partial derivative of S w.r.t. voltage amplitude.
        dSf_dVm = diagVf * np.conj(Yf * diagVnorm) + np.conj(diagIf) * \
            c_sparse((Vnorm[f], (il, f)), shape)

        dSt_dVm = diagVt * np.conj(Yt * diagVnorm) + np.conj(diagIt) * \
            c_sparse((Vnorm[t], (il, t)), shape)
    else:  # dense version
        # compute currents
        If = np.asarray(Yf * np.asmatrix(V).T).flatten()
        It = np.asarray(Yt * np.asmatrix(V).T).flatten()

        diagVf = np.asmatrix(np.diag(V[f]))
        diagIf = np.asmatrix(np.diag(If))
        diagVt = np.asmatrix(np.diag(V[t]))
        diagIt = np.asmatrix(np.diag(It))
        diagV = np.asmatrix(np.diag(V))
        diagVnorm = np.asmatrix(np.diag(Vnorm))
        temp1 = np.asmatrix(np.zeros((nl, nb), complex))
        temp2 = np.asmatrix(np.zeros((nl, nb), complex))
        temp3 = np.asmatrix(np.zeros((nl, nb), complex))
        temp4 = np.asmatrix(np.zeros((nl, nb), complex))
        for i in range(nl):
            fi, ti = f[i], t[i]
            temp1[i, fi] = V[fi].item()
            temp2[i, fi] = Vnorm[fi].item()
            temp3[i, ti] = V[ti].item()
            temp4[i, ti] = Vnorm[ti].item()

        dSf_dVa = 1j * (np.conj(diagIf) * temp1 - diagVf * np.conj(Yf * diagV))
        dSf_dVm = diagVf * np.conj(Yf * diagVnorm) + np.conj(diagIf) * temp2
        dSt_dVa = 1j * (np.conj(diagIt) * temp3 - diagVt * np.conj(Yt * diagV))
        dSt_dVm = diagVt * np.conj(Yt * diagVnorm) + np.conj(diagIt) * temp4

    # Compute power flow vectors.
    Sf = V[f] * np.conj(If)
    St = V[t] * np.conj(It)

    return dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St


def d2Sbus_dV2(Ybus, V, lam):
    """Computes 2nd derivatives of power injection w.r.t. voltage.

    Returns 4 matrices containing the partial derivatives w.r.t. voltage angle
    and magnitude of the product of a vector C{lam} with the 1st partial
    derivatives of the complex bus power injections. Takes sparse bus
    admittance matrix C{Ybus}, voltage vector C{V} and C{nb x 1} vector of
    multipliers C{lam}. Output matrices are sparse.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    nb = len(V)
    ib = np.arange(nb)
    Ibus = Ybus * V
    diaglam = c_sparse((lam, (ib, ib)))
    diagV = c_sparse((V, (ib, ib)))

    A = c_sparse((lam * V, (ib, ib)))
    B = Ybus * diagV
    C = A * np.conj(B)
    D = Ybus.H * diagV
    E = diagV.conj() * (D * diaglam - c_sparse((D * lam, (ib, ib))))
    F = C - A * c_sparse((np.conj(Ibus), (ib, ib)))
    G = c_sparse((np.ones(nb) / abs(V), (ib, ib)))

    Gaa = E + F
    Gva = 1j * G * (E - F)
    Gav = Gva.T
    Gvv = G * (C + C.T) * G

    return Gaa, Gav, Gva, Gvv


def d2AIbr_dV2(dIbr_dVa, dIbr_dVm, Ibr, Ybr, V, lam):
    """Computes 2nd derivatives of |complex current|**2 w.r.t. V.

    Returns 4 matrices containing the partial derivatives w.r.t. voltage
    angle and magnitude of the product of a vector C{lam} with the 1st partial
    derivatives of the square of the magnitude of the branch currents.
    Takes sparse first derivative matrices of complex flow, complex flow
    vector, sparse branch admittance matrix C{Ybr}, voltage vector C{V} and
    C{nl x 1} vector of multipliers C{lam}. Output matrices are sparse.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @see: L{dIbr_dV}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # define
    il = range(len(lam))

    diaglam = c_sparse((lam, (il, il)))
    diagIbr_conj = c_sparse((Ibr.conj(), (il, il)))

    Iaa, Iav, Iva, Ivv = d2Ibr_dV2(Ybr, V, diagIbr_conj * lam)

    Haa = 2 * (Iaa + dIbr_dVa.T * diaglam * dIbr_dVa.conj()).real
    Hva = 2 * (Iva + dIbr_dVm.T * diaglam * dIbr_dVa.conj()).real
    Hav = 2 * (Iav + dIbr_dVa.T * diaglam * dIbr_dVm.conj()).real
    Hvv = 2 * (Ivv + dIbr_dVm.T * diaglam * dIbr_dVm.conj()).real

    return Haa, Hav, Hva, Hvv


def d2ASbr_dV2(dSbr_dVa, dSbr_dVm, Sbr, Cbr, Ybr, V, lam):
    """Computes 2nd derivatives of |complex power flow|**2 w.r.t. V.

    Returns 4 matrices containing the partial derivatives w.r.t. voltage
    angle and magnitude of the product of a vector C{lam} with the 1st partial
    derivatives of the square of the magnitude of branch complex power flows.
    Takes sparse first derivative matrices of complex flow, complex flow
    vector, sparse connection matrix C{Cbr}, sparse branch admittance matrix
    C{Ybr}, voltage vector C{V} and C{nl x 1} vector of multipliers C{lam}.
    Output matrices are sparse.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @see: L{dSbr_dV}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    il = range(len(lam))

    diaglam = c_sparse((lam, (il, il)))
    diagSbr_conj = c_sparse((Sbr.conj(), (il, il)))

    Saa, Sav, Sva, Svv = d2Sbr_dV2(Cbr, Ybr, V, diagSbr_conj * lam)

    Haa = 2 * (Saa + dSbr_dVa.T * diaglam * dSbr_dVa.conj()).real
    Hva = 2 * (Sva + dSbr_dVm.T * diaglam * dSbr_dVa.conj()).real
    Hav = 2 * (Sav + dSbr_dVa.T * diaglam * dSbr_dVm.conj()).real
    Hvv = 2 * (Svv + dSbr_dVm.T * diaglam * dSbr_dVm.conj()).real

    return Haa, Hav, Hva, Hvv


def d2Ibr_dV2(Ybr, V, lam):
    """Computes 2nd derivatives of complex branch current w.r.t. voltage.

    Returns 4 matrices containing the partial derivatives w.r.t. voltage
    angle and magnitude of the product of a vector LAM with the 1st partial
    derivatives of the complex branch currents. Takes sparse branch admittance
    matrix C{Ybr}, voltage vector C{V} and C{nl x 1} vector of multipliers
    C{lam}. Output matrices are sparse.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    nb = len(V)
    ib = np.arange(nb)
    diaginvVm = c_sparse((np.ones(nb) / abs(V), (ib, ib)))

    Haa = c_sparse((-(Ybr.T * lam) * V, (ib, ib)))
    Hva = -1j * Haa * diaginvVm
    Hav = Hva.copy()
    Hvv = c_sparse((nb, nb))

    return Haa, Hav, Hva, Hvv


def d2Sbr_dV2(Cbr, Ybr, V, lam):
    """Computes 2nd derivatives of complex power flow w.r.t. voltage.

    Returns 4 matrices containing the partial derivatives w.r.t. voltage angle
    and magnitude of the product of a vector C{lam} with the 1st partial
    derivatives of the complex branch power flows. Takes sparse connection
    matrix C{Cbr}, sparse branch admittance matrix C{Ybr}, voltage vector C{V}
    and C{nl x 1} vector of multipliers C{lam}. Output matrices are sparse.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    nb = len(V)
    nl = len(lam)
    ib = range(nb)
    il = range(nl)

    diaglam = c_sparse((lam, (il, il)))
    diagV = c_sparse((V, (ib, ib)))

    A = Ybr.H * diaglam * Cbr
    B = np.conj(diagV) * A * diagV
    D = c_sparse(((A * V) * np.conj(V), (ib, ib)))
    E = c_sparse(((A.T * np.conj(V) * V), (ib, ib)))
    F = B + B.T
    G = c_sparse((np.ones(nb) / abs(V), (ib, ib)))

    Haa = F - D - E
    Hva = 1j * G * (B - B.T - D + E)
    Hav = Hva.T
    Hvv = G * F * G

    return Haa, Hav, Hva, Hvv


def dAbr_dV(dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St):
    """Partial derivatives of squared flow magnitudes w.r.t voltage.

    Returns four matrices containing partial derivatives of the square of
    the branch flow magnitudes at "from" & "to" ends of each branch w.r.t
    voltage magnitude and voltage angle respectively (for all buses), given
    the flows and flow sensitivities. Flows could be complex current or
    complex or real power. Notation below is based on complex power. The
    following explains the expressions used to form the matrices:

    Let Af refer to the square of the apparent power at the "from" end of
    each branch::

        Af = abs(Sf)**2
           = Sf .* conj(Sf)
           = Pf**2 + Qf**2

    then ...

    Partial w.r.t real power::
        dAf/dPf = 2 * diag(Pf)

    Partial w.r.t reactive power::
        dAf/dQf = 2 * diag(Qf)

    Partial w.r.t Vm & Va::
        dAf/dVm = dAf/dPf * dPf/dVm + dAf/dQf * dQf/dVm
        dAf/dVa = dAf/dPf * dPf/dVa + dAf/dQf * dQf/dVa

    Derivations for "to" bus are similar.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @return: The partial derivatives of the squared flow magnitudes w.r.t
             voltage magnitude and voltage angle given the flows and flow
             sensitivities. Flows could be complex current or complex or
             real power.
    @see: L{dIbr_dV}, L{dSbr_dV}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    il = range(len(Sf))

    dAf_dPf = c_sparse((2 * Sf.real, (il, il)))
    dAf_dQf = c_sparse((2 * Sf.imag, (il, il)))
    dAt_dPt = c_sparse((2 * St.real, (il, il)))
    dAt_dQt = c_sparse((2 * St.imag, (il, il)))

    # Partial derivative of apparent power magnitude w.r.t voltage
    # phase angle.
    dAf_dVa = dAf_dPf * dSf_dVa.real + dAf_dQf * dSf_dVa.imag
    dAt_dVa = dAt_dPt * dSt_dVa.real + dAt_dQt * dSt_dVa.imag
    # Partial derivative of apparent power magnitude w.r.t. voltage
    # amplitude.
    dAf_dVm = dAf_dPf * dSf_dVm.real + dAf_dQf * dSf_dVm.imag
    dAt_dVm = dAt_dPt * dSt_dVm.real + dAt_dQt * dSt_dVm.imag

    return dAf_dVa, dAf_dVm, dAt_dVa, dAt_dVm
