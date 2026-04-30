"""Golden e_fn fixtures for AMS routine dopf (Phase 4.5-A).
Hand-translated callables from the 4.2-4.4 migration. Used as the parity
oracle for the codegen in 4.5-B onward.
"""
import cvxpy as cp

# --- e_fn callables (Phase 4.3 migration) ---

def _qglb(r):
    return -r.qg + cp.multiply(r.ug, r.qmin) <= 0


def _qgub(r):
    return r.qg - cp.multiply(r.ug, r.qmax) <= 0


def _vu(r):
    return r.vsq - r.vmax ** 2 <= 0


def _vl(r):
    return -r.vsq + r.vmin ** 2 <= 0


def _lvd(r):
    return r.CftT @ r.vsq - (cp.multiply(r.r, r.plf) + cp.multiply(r.x, r.qlf)) == 0


def _qb(r):
    return cp.sum(r.qd) - cp.sum(r.qg) == 0


def _dopfvis_obj(r):
    # Original e_str used cvxpy's deprecated `*` which is matrix-multiply
    # for vectors. c2/c1/c0/ug/pg are size 5 (StaticGen); cm/cd/M/D are
    # size 4 (VSG). Each term is a scalar dot product; preserve those
    # semantics with `@`.
    return (r.c2 @ r.pg ** 2 + r.c1 @ r.pg + r.ug @ r.c0
            + r.cm @ r.M + r.cd @ r.D)


