"""Golden e_fn fixtures for AMS routine ed (Phase 4.5-A).
Hand-translated callables from the 4.2-4.4 migration. Used as the parity
oracle for the codegen in 4.5-B onward.
"""
import cvxpy as cp

# --- ED e_fn callables (Phase 4.4) ---


def _ed_pmaxe(r):
    return (cp.multiply(cp.multiply(r.nctrle, r.pg0), r.tlv)
            + cp.multiply(cp.multiply(r.ctrle, r.tlv), r.pmax))


def _ed_pmine(r):
    return (cp.multiply(cp.multiply(r.nctrle, r.pg0), r.tlv)
            + cp.multiply(cp.multiply(r.ctrle, r.tlv), r.pmin))


def _ed_pglb(r):
    return -r.pg + r.pmine <= 0


def _ed_pgub(r):
    return r.pg - r.pmaxe <= 0


def _ed_prsb(r):
    return cp.multiply(r.ugt, r.pmax @ r.tlv - r.pg) - r.prs == 0


def _ed_rsr(r):
    return -r.gs @ r.prs + r.dsr <= 0


def _ed_plflb(r):
    return -r.Bf @ r.aBus - r.Pfinj @ r.tlv - cp.multiply(r.ul, r.rate_a) @ r.tlv <= 0


def _ed_plfub(r):
    return r.Bf @ r.aBus + r.Pfinj @ r.tlv - cp.multiply(r.ul, r.rate_a) @ r.tlv <= 0


def _ed_alflb(r):
    return -r.CftT @ r.aBus + r.amin @ r.tlv <= 0


def _ed_alfub(r):
    return r.CftT @ r.aBus - r.amax @ r.tlv <= 0


def _ed_plf(r):
    return r.Bf @ r.aBus + r.Pfinj @ r.tlv


def _ed_pb(r):
    return r.Bbus @ r.aBus + r.Pbusinj @ r.tlv + r.Cl @ r.pds + r.Csh @ r.gsh @ r.tlv - r.Cg @ r.pg == 0


def _ed_rbu(r):
    return r.gs @ cp.multiply(r.ugt, r.pru) - cp.multiply(r.dud, r.tlv) == 0


def _ed_rbd(r):
    return r.gs @ cp.multiply(r.ugt, r.prd) - cp.multiply(r.ddd, r.tlv) == 0


def _ed_rru(r):
    return r.pg + r.pru - cp.multiply(cp.multiply(r.ugt, r.pmax), r.tlv) <= 0


def _ed_rrd(r):
    return -r.pg + r.prd + cp.multiply(cp.multiply(r.ugt, r.pmin), r.tlv) <= 0


def _ed_rgu(r):
    return r.pg @ r.Mr - r.t * r.RR30 <= 0


def _ed_rgd(r):
    return -r.pg @ r.Mr - r.t * r.RR30 <= 0


def _ed_rgu0(r):
    return cp.multiply(r.ugt[:, 0], r.pg[:, 0] - r.pg0[:, 0] - r.R30) <= 0


def _ed_rgd0(r):
    return cp.multiply(r.ugt[:, 0], -r.pg[:, 0] + r.pg0[:, 0] - r.R30) <= 0


def _ed_obj(r):
    return (cp.sum(r.t ** 2 * r.c2 @ r.pg ** 2)
            + r.t * cp.sum(r.c1 @ r.pg + r.csr @ r.prs)
            + cp.sum(cp.multiply(r.ugt, cp.multiply(r.c0, r.tlv))))


# ESD1MPBase
def _esd1mp_SOCb(r):
    return (cp.multiply(r.EnR, r.SOC @ r.Mre)
            - r.t * cp.multiply(r.EtaCR, r.pce[:, 1:])
            + r.t * cp.multiply(r.REtaDR, r.pde[:, 1:])) == 0


def _esd1mp_SOCb0(r):
    return (cp.multiply(r.En, r.SOC[:, 0] - r.SOCinit)
            - r.t * cp.multiply(r.EtaC, r.pce[:, 0])
            + r.t * cp.multiply(r.REtaD, r.pde[:, 0])) == 0


def _esd1mp_SOCr(r):
    return r.SOCend - r.SOC[:, -1] <= 0


