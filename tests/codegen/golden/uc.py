"""Golden e_fn fixtures for AMS routine uc (Phase 4.5-A).
Hand-translated callables from the 4.2-4.4 migration. Used as the parity
oracle for the codegen in 4.5-B onward.
"""
import cvxpy as cp

# --- UC e_fn callables (Phase 4.4) ---


def _uc_pmaxe(r):
    return (cp.multiply(cp.multiply(r.nctrl, r.pg0), r.ugd)
            + cp.multiply(cp.multiply(r.ctrl, r.pmax), r.ugd))


def _uc_pmine(r):
    return (cp.multiply(cp.multiply(r.ctrl, r.pmin), r.ugd)
            + cp.multiply(cp.multiply(r.nctrl, r.pg0), r.ugd))


def _uc_pglb(r):
    return -r.pg + r.pmine <= 0


def _uc_pgub(r):
    return r.pg - r.pmaxe <= 0


def _uc_actv(r):
    return r.ugd @ r.Mr - r.vgd[:, 1:] == 0


def _uc_actv0(r):
    return r.ugd[:, 0] - r.ug[:, 0] - r.vgd[:, 0] == 0


def _uc_actw(r):
    return -r.ugd @ r.Mr - r.wgd[:, 1:] == 0


def _uc_actw0(r):
    return -r.ugd[:, 0] + r.ug[:, 0] - r.wgd[:, 0] == 0


def _uc_prsb(r):
    return cp.multiply(r.ugd, cp.multiply(r.pmax, r.tlv)) - r.zug - r.prs == 0


def _uc_rsr(r):
    return -r.gs @ r.prs + r.dsr <= 0


def _uc_prnsb(r):
    return cp.multiply(1 - r.ugd, cp.multiply(r.pmax, r.tlv)) - r.prns == 0


def _uc_rnsr(r):
    return -r.gs @ r.prns + r.dnsr <= 0


def _uc_zuglb(r):
    return -r.zug + r.pg <= 0


def _uc_zugub(r):
    return r.zug - r.pg - r.Mzug * (1 - r.ugd) <= 0


def _uc_zugub2(r):
    return r.zug - r.Mzug * r.ugd <= 0


def _uc_don(r):
    return cp.multiply(r.Con, r.vgd) - r.ugd <= 0


def _uc_doff(r):
    return cp.multiply(r.Coff, r.wgd) - (1 - r.ugd) <= 0


def _uc_plflb(r):
    return -r.Bf @ r.aBus - r.Pfinj - cp.multiply(r.rate_a, r.tlv) <= 0


def _uc_plfub(r):
    return r.Bf @ r.aBus + r.Pfinj - cp.multiply(r.rate_a, r.tlv) <= 0


def _uc_alflb(r):
    return -r.CftT @ r.aBus - r.amax @ r.tlv <= 0


def _uc_alfub(r):
    return r.CftT @ r.aBus - r.amax @ r.tlv <= 0


def _uc_pdumax(r):
    return r.pdu - cp.multiply(r.pdsp, r.dctrl @ r.tlv) <= 0


def _uc_pb(r):
    return (r.Bbus @ r.aBus + r.Pbusinj @ r.tlv
            + r.Cl @ (r.pds - r.pdu) + r.Csh @ r.gsh @ r.tlv
            - r.Cg @ r.pg) == 0


def _uc_obj(r):
    return (r.t ** 2 * cp.sum(r.c2 @ r.pg ** 2)
            + r.t * cp.sum(r.c1 @ r.pg)
            + cp.sum(cp.multiply(r.ug, r.c0) @ r.tlv)
            + cp.sum(r.csu @ r.vgd + r.csd @ r.wgd)
            + r.t * cp.sum(r.csr @ r.prs + r.cnsr @ r.prns + r.cdp @ r.pdu))


