"""Golden e_fn fixtures for AMS routine dcopf (Phase 4.5-A).
Hand-translated callables from the 4.2-4.4 migration. Used as the parity
oracle for the codegen in 4.5-B onward.
"""
import cvxpy as cp

# --- e_fn callables (Phase 4.2 migration; mirrors the prior e_str) ---

def _pmaxe(r):
    """Effective pmax: mul(nctrle, pg0) + mul(ctrle, pmax)."""
    return cp.multiply(r.nctrle, r.pg0) + cp.multiply(r.ctrle, r.pmax)


def _pmine(r):
    """Effective pmin: mul(nctrle, pg0) + mul(ctrle, pmin)."""
    return cp.multiply(r.nctrle, r.pg0) + cp.multiply(r.ctrle, r.pmin)


def _pglb(r):
    return -r.pg + r.pmine <= 0


def _pgub(r):
    return r.pg - r.pmaxe <= 0


def _plflb(r):
    return -r.plf - cp.multiply(r.ul, r.rate_a) <= 0


def _plfub(r):
    return r.plf - cp.multiply(r.ul, r.rate_a) <= 0


def _alflb(r):
    return -r.CftT @ r.aBus + r.amin <= 0


def _alfub(r):
    return r.CftT @ r.aBus - r.amax <= 0


def _obj(r):
    return (cp.sum(cp.multiply(r.c2, r.pg ** 2))
            + cp.sum(cp.multiply(r.c1, r.pg))
            + cp.sum(cp.multiply(r.ug, r.c0)))


