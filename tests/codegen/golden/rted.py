"""Golden e_fn fixtures for AMS routine rted (Phase 4.5-A).
Hand-translated callables from the 4.2-4.4 migration. Used as the parity
oracle for the codegen in 4.5-B onward.
"""
import cvxpy as cp

# --- RTED e_fn callables (Phase 4.4) ---


def _rted_rbu(r):
    return r.gs @ cp.multiply(r.ug, r.pru) - r.dud == 0


def _rted_rbd(r):
    return r.gs @ cp.multiply(r.ug, r.prd) - r.ddd == 0


def _rted_rru(r):
    return cp.multiply(r.ug, r.pg + r.pru) - cp.multiply(r.ug, r.pmaxe) <= 0


def _rted_rrd(r):
    return cp.multiply(r.ug, -r.pg + r.prd) + cp.multiply(r.ug, r.pmine) <= 0


def _rted_rgu(r):
    return cp.multiply(r.ug, r.pg - r.pg0 - r.R10) <= 0


def _rted_rgd(r):
    return cp.multiply(r.ug, -r.pg + r.pg0 - r.R10) <= 0


def _rted_obj(r):
    return (r.t ** 2 * cp.sum(cp.multiply(r.c2, r.pg ** 2))
            + cp.sum(cp.multiply(r.ug, r.c0))
            + r.t * cp.sum(r.c1 @ r.pg + r.cru @ r.pru + r.crd @ r.prd))


def _esd1_obj_extra(r):
    """ESD1Base extra objective term (registered via Objective.add_term)."""
    return r.t * cp.sum(-cp.multiply(r.cesdc, r.pce) + cp.multiply(r.cesdd, r.pde))


def _rtedvis_obj_extra(r):
    """RTEDVIS extra objective term."""
    return r.t * cp.sum(cp.multiply(r.cm, r.M) + cp.multiply(r.cd, r.D))


# DGBase
def _dgb_cdgb(r):
    return r.idg @ r.pg - r.pgdg == 0


# ESD1PBase
def _esd1p_cesd(r):
    return r.ies @ r.pg + r.pce - r.pde == 0


def _esd1p_SOClb(r):
    return -r.SOC + r.SOCmin <= 0


def _esd1p_SOCub(r):
    return r.SOC - r.SOCmax <= 0


def _esd1p_SOCb(r):
    return (cp.multiply(r.En, (r.SOC - r.SOCinit))
            - r.t * cp.multiply(r.EtaC, r.pce)
            + r.t * cp.multiply(r.REtaD, r.pde)) == 0


def _esd1p_SOCr(r):
    return r.SOCend - r.SOC <= 0


# RTEDESP
def _rtedesp_zce(r):
    return cp.multiply(1 - r.ucd, r.pce) <= 0


def _rtedesp_zde(r):
    return cp.multiply(1 - r.udd, r.pde) <= 0


# ESD1Base (single-period)
def _esd1b_cdb(r):
    return r.ucd + r.udd - 1 <= 0


def _esd1b_zce1(r):
    return -r.zce + r.pce <= 0


def _esd1b_zce2(r):
    return r.zce - r.pce - r.Mb * (1 - r.ucd) <= 0


def _esd1b_zce3(r):
    return r.zce - r.Mb * r.ucd <= 0


def _esd1b_zde1(r):
    return -r.zde + r.pde <= 0


def _esd1b_zde2(r):
    return r.zde - r.pde - r.Mb * (1 - r.udd) <= 0


def _esd1b_zde3(r):
    return r.zde - r.Mb * r.udd <= 0


def _esd1b_tcdr(r):
    return (r.tdc0 > 0) * (r.tdc > r.tdc0) - r.ucd <= 0


def _esd1b_tddr(r):
    return (r.tdd0 > 0) * (r.tdd > r.tdd0) - r.udd <= 0


# VISBase
def _vis_Mub(r):
    return r.M - r.Mmax <= 0


def _vis_Dub(r):
    return r.D - r.Dmax <= 0


def _vis_Mreq(r):
    return -r.gvsg @ r.M + r.dvm == 0


def _vis_Dreq(r):
    return -r.gvsg @ r.D + r.dvd == 0


