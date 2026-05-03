"""
Known-good MATPOWER cross-checks — parametrized over case file.

Replaces ``tests/test_known_good.py``'s 3 unittest classes
(``TestKnownResults`` / ``TestKnownResultsIEEE39`` /
``TestKnownResultsIEEE118``) — same 4 routines (DCPF, PFlow,
DCOPF, Matrices) × 3 cases (case14, case39, case118), now one
parametrized body per routine.

Per-case differences captured in `_CASES`:

- ``dcpf_pg_sum``: case14 asserts pg element-wise; case39/case118
  only assert the total. Element-wise on case14 keeps the tighter
  coverage that lived there originally.
- ``dcpf_normalize_aBus``: case118 normalizes both sides by
  subtracting the slack-bus angle before comparing — MATPOWER and
  AMS picked different reference angles on that case.
- ``dcopf_places``: tighter (4) for case14, looser (2) for the
  larger systems where solver slop dominates.
- ``matrices_lodf_atol``: ``1e-2`` for case14, ``10`` for the
  larger systems.

Tolerances are preserved verbatim from the legacy bodies.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from andes.shared import rad2deg
from ams.shared import nan


@dataclass(frozen=True)
class _CaseSpec:
    case_id: str
    fixture_name: str
    dcpf_pg_sum: bool
    dcpf_normalize_aBus: bool
    dcopf_places: int
    matrices_lodf_atol: float


_CASES = {
    "case14":  _CaseSpec(case_id="case14",  fixture_name="case14",
                         dcpf_pg_sum=False, dcpf_normalize_aBus=False,
                         dcopf_places=4, matrices_lodf_atol=1e-2),
    "case39":  _CaseSpec(case_id="case39",  fixture_name="case39",
                         dcpf_pg_sum=True,  dcpf_normalize_aBus=False,
                         dcopf_places=2, matrices_lodf_atol=10),
    "case118": _CaseSpec(case_id="case118", fixture_name="case118",
                         dcpf_pg_sum=True,  dcpf_normalize_aBus=True,
                         dcopf_places=2, matrices_lodf_atol=10),
}

_CASE_IDS = list(_CASES)


@dataclass(frozen=True)
class _Ctx:
    sp: object
    spec: _CaseSpec
    mpres_case: dict


@pytest.fixture
def ctx(request, benchmark_mpres):
    spec = _CASES[request.param]
    sp = request.getfixturevalue(spec.fixture_name)
    return _Ctx(sp=sp, spec=spec, mpres_case=benchmark_mpres[spec.case_id])


_PARAMETRIZE_CASES = pytest.mark.parametrize(
    "ctx", _CASE_IDS, indirect=True, ids=_CASE_IDS,
)


def _load_ptdf(mpres_case):
    ptdf_data = np.array(mpres_case['PTDF'])
    return np.array(
        [[0 if val == "_NaN_" else val for val in row] for row in ptdf_data],
        dtype=float,
    )


def _load_lodf(mpres_case):
    lodf_data = np.array(mpres_case['LODF'])
    lodf = np.array(
        [[nan if val in ["_NaN_", "-_Inf_", "_Inf_"] else val for val in row]
         for row in lodf_data],
        dtype=float,
    )
    np.fill_diagonal(lodf, -1)
    return lodf


@_PARAMETRIZE_CASES
def test_dcpf(ctx):
    """DC power flow vs MATPOWER reference."""
    ctx.sp.DCPF.run(solver='CLARABEL')

    aBus_mp = np.array(ctx.mpres_case['DCPF']['aBus']).reshape(-1)
    aBus_sp = ctx.sp.DCPF.aBus.v * rad2deg
    if ctx.spec.dcpf_normalize_aBus:
        aBus_mp = aBus_mp - aBus_mp[0]
        aBus_sp = aBus_sp - aBus_sp[0]
    np.testing.assert_allclose(aBus_sp, aBus_mp, rtol=1e-2, atol=1e-2)

    pg_sp = ctx.sp.DCPF.pg.v * ctx.sp.config.mva
    pg_mp = np.array(ctx.mpres_case['DCPF']['pg']).reshape(-1)
    if ctx.spec.dcpf_pg_sum:
        np.testing.assert_allclose(pg_sp.sum(), pg_mp.sum(),
                                   rtol=1e-2, atol=1e-2)
    else:
        np.testing.assert_allclose(pg_sp, pg_mp, rtol=1e-2, atol=1e-2)


@_PARAMETRIZE_CASES
def test_pflow(ctx):
    """AC power flow vs MATPOWER reference."""
    ctx.sp.PFlow.run()
    np.testing.assert_allclose(
        ctx.sp.PFlow.aBus.v * rad2deg,
        np.array(ctx.mpres_case['PFlow']['aBus']).reshape(-1),
        rtol=1e-2, atol=1e-2,
    )
    np.testing.assert_allclose(
        ctx.sp.PFlow.vBus.v,
        np.array(ctx.mpres_case['PFlow']['vBus']).reshape(-1),
        rtol=1e-2, atol=1e-2,
    )
    np.testing.assert_allclose(
        ctx.sp.PFlow.pg.v.sum() * ctx.sp.config.mva,
        np.array(ctx.mpres_case['PFlow']['pg']).sum(),
        rtol=1e-2, atol=1e-2,
    )


@_PARAMETRIZE_CASES
def test_dcopf(ctx):
    """DCOPF objective + duals vs MATPOWER reference."""
    ctx.sp.DCOPF.run(solver='CLARABEL')
    assert round(ctx.sp.DCOPF.obj.v - ctx.mpres_case['DCOPF']['obj'],
                 ctx.spec.dcopf_places) == 0, \
        f"DCOPF objective mismatch on {ctx.spec.case_id}"
    np.testing.assert_allclose(
        ctx.sp.DCOPF.pi.v / ctx.sp.config.mva,
        np.array(ctx.mpres_case['DCOPF']['pi']).reshape(-1),
        rtol=1e-2, atol=1e-2,
    )


@_PARAMETRIZE_CASES
def test_matrices(ctx):
    """PTDF + LODF vs MATPOWER reference."""
    ptdf = ctx.sp.mats.build_ptdf().todense()
    lodf = ctx.sp.mats.build_lodf().todense()

    ptdf_mp = _load_ptdf(ctx.mpres_case)
    lodf_mp = _load_lodf(ctx.mpres_case)

    ptdf[np.isnan(ptdf_mp)] = nan
    lodf[np.isnan(lodf_mp)] = nan

    np.testing.assert_allclose(ptdf, ptdf_mp,
                               equal_nan=True, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(lodf, lodf_mp,
                               equal_nan=True, rtol=1e-2,
                               atol=ctx.spec.matrices_lodf_atol)
