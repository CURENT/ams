"""
Single-period DC LP routine scenarios — parametrized.

Family A (this module): DCOPF, DCOPF2, RTED, RTEDDG, RTEDES, RTED2,
RTED2DG, RTED2ES. All eight share the same scenario body modulo
routine name. Per-routine differences live in `_ROUTINES`:

- ``needs_ptdf``: 2nd-generation routines require ``ss.mats.build_ptdf()``
  before the first solve.
- ``has_aBus``: 2nd-generation routines also expose ``aBus`` so the
  vBus scenario asserts both.
- ``solver``: ``'SCIP'`` for the MISOCP storage variants
  (RTEDES, RTED2ES); ``'CLARABEL'`` otherwise. Tests that call
  ``.run(...)`` are skipped when the matching solver is unavailable;
  ``test_init`` does not invoke a solver and is never skipped on this
  axis.
- ``set_load_pq1``: target PQ_1 ``p0`` value for ``test_set_load``.
  ``0.05`` for storage variants — the larger load reduction of ``0.1``
  is absorbed by ESD1 charging and weakens the ``pgs_pqt < pgs``
  assertion.

OPF is intentionally excluded — its ``_ac`` dual-mode shape doesn't
fit Family A's body and is handled separately.

Per-test isolation comes from ``conftest.pjm5bus_json`` (function-
scoped deepcopy of a cached load), so trailing ``set(value=1)`` /
``Line.alter`` resets are no longer needed.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from tests.conftest import HAS_MISOCP, HAS_PYPOWER


@dataclass(frozen=True)
class _RoutineSpec:
    needs_ptdf: bool
    has_aBus: bool
    solver: str
    set_load_pq1: float = 0.1


_ROUTINES = {
    "DCOPF":   _RoutineSpec(needs_ptdf=False, has_aBus=False, solver='CLARABEL'),
    "DCOPF2":  _RoutineSpec(needs_ptdf=True,  has_aBus=True,  solver='CLARABEL'),
    "RTED":    _RoutineSpec(needs_ptdf=False, has_aBus=False, solver='CLARABEL'),
    "RTEDDG":  _RoutineSpec(needs_ptdf=False, has_aBus=False, solver='CLARABEL'),
    "RTEDES":  _RoutineSpec(needs_ptdf=False, has_aBus=False, solver='SCIP', set_load_pq1=0.05),
    "RTED2":   _RoutineSpec(needs_ptdf=True,  has_aBus=True,  solver='CLARABEL'),
    "RTED2DG": _RoutineSpec(needs_ptdf=True,  has_aBus=True,  solver='CLARABEL'),
    "RTED2ES": _RoutineSpec(needs_ptdf=True,  has_aBus=True,  solver='SCIP', set_load_pq1=0.05),
}

_ROUTINE_IDS = list(_ROUTINES)


@dataclass(frozen=True)
class _Ctx:
    ss: object
    routine_id: str
    spec: _RoutineSpec
    rtn: object


@pytest.fixture
def ctx(pjm5bus_json, request):
    """Family-A context: fresh case5 system + routine handle + spec."""
    routine_id = request.param
    spec = _ROUTINES[routine_id]
    ss = pjm5bus_json
    ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
    if spec.needs_ptdf:
        ss.mats.build_ptdf()
    return _Ctx(ss=ss, routine_id=routine_id, spec=spec, rtn=getattr(ss, routine_id))


def _skip_if_solver_missing(spec):
    if spec.solver == 'SCIP' and not HAS_MISOCP:
        pytest.skip("No MISOCP solver is available.")


_PARAMETRIZE_ROUTINES = pytest.mark.parametrize(
    "ctx", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS,
)


@_PARAMETRIZE_ROUTINES
def test_init(ctx):
    # No solver invoked here — never skip on solver availability.
    ctx.rtn.init()
    assert ctx.rtn.initialized, f"{ctx.routine_id} initialization failed!"


@_PARAMETRIZE_ROUTINES
def test_trip_gen(ctx):
    _skip_if_solver_missing(ctx.spec)
    stg = 'PV_1'
    ctx.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)

    ctx.rtn.update()
    ctx.rtn.run(solver=ctx.spec.solver)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge under generator trip!"
    assert abs(ctx.rtn.get(src='pg', attr='v', idx=stg)) < 1e-6, \
        "Generator trip does not take effect!"


@_PARAMETRIZE_ROUTINES
def test_trip_line(ctx):
    _skip_if_solver_missing(ctx.spec)
    ctx.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

    ctx.rtn.update()
    ctx.rtn.run(solver=ctx.spec.solver)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge under line trip!"
    assert abs(ctx.rtn.get(src='plf', attr='v', idx='Line_3')) < 1e-6, \
        "Line trip does not take effect!"


@_PARAMETRIZE_ROUTINES
def test_set_load(ctx):
    _skip_if_solver_missing(ctx.spec)

    ctx.rtn.run(solver=ctx.spec.solver)
    pgs = ctx.rtn.pg.v.sum()

    ctx.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=ctx.spec.set_load_pq1)
    ctx.rtn.update()
    ctx.rtn.run(solver=ctx.spec.solver)
    pgs_pqt = ctx.rtn.pg.v.sum()
    assert pgs_pqt < pgs, "Load set does not take effect!"

    ctx.ss.PQ.set(src='u', attr='v', idx='PQ_2', value=0)
    ctx.rtn.update()
    ctx.rtn.run(solver=ctx.spec.solver)
    pgs_pqt2 = ctx.rtn.pg.v.sum()
    assert pgs_pqt2 < pgs_pqt, "Load trip does not take effect!"


@_PARAMETRIZE_ROUTINES
def test_vBus(ctx):
    _skip_if_solver_missing(ctx.spec)
    ctx.rtn.run(solver=ctx.spec.solver)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge!"
    assert np.any(ctx.rtn.vBus.v), "vBus is all zero!"
    if ctx.spec.has_aBus:
        assert np.any(ctx.rtn.aBus.v), "aBus is all zero!"


@pytest.mark.skipif(not HAS_PYPOWER, reason="PYPOWER is not available.")
@_PARAMETRIZE_ROUTINES
def test_dc2ac(ctx):
    _skip_if_solver_missing(ctx.spec)
    ctx.rtn.run(solver=ctx.spec.solver)
    ctx.rtn.dc2ac()
    assert ctx.rtn.converted, "AC conversion failed!"
    assert ctx.rtn.exec_time > 0, "Execution time is not greater than 0."

    stg_idx = ctx.ss.StaticGen.get_all_idxes()
    np.testing.assert_almost_equal(
        ctx.rtn.get(src='pg', attr='v', idx=stg_idx),
        ctx.ss.ACOPF.get(src='pg', attr='v', idx=stg_idx),
        decimal=3,
    )

    bus_idx = ctx.ss.Bus.get_all_idxes()
    np.testing.assert_almost_equal(
        ctx.rtn.get(src='vBus', attr='v', idx=bus_idx),
        ctx.ss.ACOPF.get(src='vBus', attr='v', idx=bus_idx),
        decimal=3,
    )
    np.testing.assert_almost_equal(
        ctx.rtn.get(src='aBus', attr='v', idx=bus_idx),
        ctx.ss.ACOPF.get(src='aBus', attr='v', idx=bus_idx),
        decimal=3,
    )
