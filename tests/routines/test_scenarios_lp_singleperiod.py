"""
Single-period DC LP routine scenarios — parametrized.

Family A (this module): DCOPF, DCOPF2, RTED, RTEDDG, RTEDES, RTED2,
RTED2DG, RTED2ES. All eight share the same scenario body modulo
routine name. Differences captured by `_ROUTINES`:

- ``needs_ptdf``: 2nd-generation routines require ``ss.mats.build_ptdf()``
  before the first solve.
- ``has_aBus``: 2nd-generation routines also expose ``aBus`` so the
  vBus scenario asserts both.

OPF is intentionally excluded — its ``_ac`` dual-mode shape doesn't
fit Family A's body and is handled separately in Family B.

Per-test isolation comes from ``conftest.pjm5bus_json`` (function-
scoped deepcopy of a cached load), so trailing ``set(value=1)`` /
``Line.alter`` resets are no longer needed.
"""

from dataclasses import dataclass

import numpy as np
import pytest

try:
    import pypower  # noqa: F401
    _HAS_PYPOWER = True
except ImportError:
    _HAS_PYPOWER = False

import cvxpy as cp

_MISOCP_SOLVERS = {'MOSEK', 'CPLEX', 'GUROBI', 'XPRESS', 'SCIP'}
_HAS_MISOCP = bool(_MISOCP_SOLVERS & set(cp.installed_solvers()))


@dataclass(frozen=True)
class _RoutineSpec:
    needs_ptdf: bool
    has_aBus: bool
    solver: str  # 'CLARABEL' for LP, 'SCIP' for MISOCP storage variants


_ROUTINES = {
    "DCOPF":   _RoutineSpec(needs_ptdf=False, has_aBus=False, solver='CLARABEL'),
    "DCOPF2":  _RoutineSpec(needs_ptdf=True,  has_aBus=True,  solver='CLARABEL'),
    "RTED":    _RoutineSpec(needs_ptdf=False, has_aBus=False, solver='CLARABEL'),
    "RTEDDG":  _RoutineSpec(needs_ptdf=False, has_aBus=False, solver='CLARABEL'),
    "RTEDES":  _RoutineSpec(needs_ptdf=False, has_aBus=False, solver='SCIP'),
    "RTED2":   _RoutineSpec(needs_ptdf=True,  has_aBus=True,  solver='CLARABEL'),
    "RTED2DG": _RoutineSpec(needs_ptdf=True,  has_aBus=True,  solver='CLARABEL'),
    "RTED2ES": _RoutineSpec(needs_ptdf=True,  has_aBus=True,  solver='SCIP'),
}

_ROUTINE_IDS = list(_ROUTINES)


@pytest.fixture
def ss(pjm5bus_json, request):
    """Fresh case5 system with the standard load decrease + PTDF if needed."""
    spec = _ROUTINES[request.param]
    if spec.solver == 'SCIP' and not _HAS_MISOCP:
        pytest.skip("No MISOCP solver is available.")
    ss = pjm5bus_json
    ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
    if spec.needs_ptdf:
        ss.mats.build_ptdf()
    return ss


def _resolve(request):
    routine_id = request.node.callspec.params["ss"]
    return routine_id, _ROUTINES[routine_id]


def _routine(ss, routine_id):
    return getattr(ss, routine_id)


@pytest.mark.parametrize("ss", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS)
def test_init(ss, request):
    routine_id, _ = _resolve(request)
    rtn = _routine(ss, routine_id)
    rtn.init()
    assert rtn.initialized, f"{routine_id} initialization failed!"


@pytest.mark.parametrize("ss", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS)
def test_trip_gen(ss, request):
    routine_id, spec = _resolve(request)
    rtn = _routine(ss, routine_id)
    stg = 'PV_1'
    ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

    rtn.update()
    rtn.run(solver=spec.solver)
    assert rtn.converged, f"{routine_id} did not converge under generator trip!"
    assert abs(rtn.get(src='pg', attr='v', idx=stg)) < 1e-6, \
        "Generator trip does not take effect!"


@pytest.mark.parametrize("ss", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS)
def test_trip_line(ss, request):
    routine_id, spec = _resolve(request)
    rtn = _routine(ss, routine_id)
    ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

    rtn.update()
    rtn.run(solver=spec.solver)
    assert rtn.converged, f"{routine_id} did not converge under line trip!"
    assert abs(rtn.get(src='plf', attr='v', idx='Line_3')) < 1e-6, \
        "Line trip does not take effect!"


@pytest.mark.parametrize("ss", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS)
def test_set_load(ss, request):
    routine_id, spec = _resolve(request)
    rtn = _routine(ss, routine_id)

    rtn.run(solver=spec.solver)
    pgs = rtn.pg.v.sum()

    ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
    rtn.update()
    rtn.run(solver=spec.solver)
    pgs_pqt = rtn.pg.v.sum()
    assert pgs_pqt < pgs, "Load set does not take effect!"

    ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
    rtn.update()
    rtn.run(solver=spec.solver)
    pgs_pqt2 = rtn.pg.v.sum()
    assert pgs_pqt2 < pgs_pqt, "Load trip does not take effect!"


@pytest.mark.parametrize("ss", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS)
def test_vBus(ss, request):
    routine_id, spec = _resolve(request)
    rtn = _routine(ss, routine_id)
    rtn.run(solver=spec.solver)
    assert rtn.converged, f"{routine_id} did not converge!"
    assert np.any(rtn.vBus.v), "vBus is all zero!"
    if spec.has_aBus:
        assert np.any(rtn.aBus.v), "aBus is all zero!"


@pytest.mark.skipif(not _HAS_PYPOWER, reason="PYPOWER is not available.")
@pytest.mark.parametrize("ss", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS)
def test_dc2ac(ss, request):
    routine_id, spec = _resolve(request)
    rtn = _routine(ss, routine_id)
    rtn.run(solver=spec.solver)
    rtn.dc2ac()
    assert rtn.converted, "AC conversion failed!"
    assert rtn.exec_time > 0, "Execution time is not greater than 0."

    stg_idx = ss.StaticGen.get_all_idxes()
    np.testing.assert_almost_equal(
        rtn.get(src='pg', attr='v', idx=stg_idx),
        ss.ACOPF.get(src='pg', attr='v', idx=stg_idx),
        decimal=3,
    )

    bus_idx = ss.Bus.get_all_idxes()
    np.testing.assert_almost_equal(
        rtn.get(src='vBus', attr='v', idx=bus_idx),
        ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx),
        decimal=3,
    )
    np.testing.assert_almost_equal(
        rtn.get(src='aBus', attr='v', idx=bus_idx),
        ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx),
        decimal=3,
    )
