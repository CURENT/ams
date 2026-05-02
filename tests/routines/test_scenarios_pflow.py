"""
Power-flow / DC-PF routine scenarios — parametrized.

Family C (this module): DCPF, PFlow on ``case14`` and the four
PYPOWER-backed variants (DCPF1, PFlow1, DCOPF1, ACOPF1) on
``pjm5bus_demo``. Per-routine differences live in `_ROUTINES`:

- ``case_fixture``: name of the conftest fixture supplying the
  pre-loaded system (``'pjm5bus_json'`` or ``'case14'``).
- ``stg``: generator idx used for the trip test (str ``'PV_1'`` for
  pjm5bus, int ``2`` for case14).
- ``solver_kwarg``: passed to ``.run(...)`` as ``solver=...``;
  ``None`` for routines that don't accept the kwarg (PFlow uses
  ANDES NR; PYPOWER variants use the PYPOWER backend).
- ``needs_pypower``: PYPOWER-backed routines skip ``.run`` tests
  when PYPOWER is unavailable. ``test_init`` does not invoke a
  backend and is never skipped on this axis.
- ``pre_reduce_load``: pjm5bus needs an initial PQ_1/PQ_2 ``p0``
  reduction to keep the test region feasible; case14 does not.
- ``trip_attr`` / ``trip_idx``: how ``test_set_load`` removes the
  load source. case14 routines use ``set(p0=0, idx='PQ_1')``;
  pjm5bus routines use ``set(u=0, idx='PQ_2')``.
- ``has_vBus``: only DCPF asserts ``vBus`` after a default solve.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from tests.conftest import HAS_PYPOWER


@dataclass(frozen=True)
class _RoutineSpec:
    case_fixture: str
    stg: object
    solver_kwarg: object  # str or None
    needs_pypower: bool
    pre_reduce_load: bool
    trip_attr: str
    trip_idx: str
    has_vBus: bool


_ROUTINES = {
    "DCPF":   _RoutineSpec(case_fixture='case14',       stg=2,      solver_kwarg='CLARABEL',
                           needs_pypower=False, pre_reduce_load=False,
                           trip_attr='p0', trip_idx='PQ_1', has_vBus=True),
    "PFlow":  _RoutineSpec(case_fixture='case14',       stg=2,      solver_kwarg=None,
                           needs_pypower=False, pre_reduce_load=False,
                           trip_attr='p0', trip_idx='PQ_1', has_vBus=False),
    "DCPF1":  _RoutineSpec(case_fixture='pjm5bus_json', stg='PV_1', solver_kwarg=None,
                           needs_pypower=True,  pre_reduce_load=True,
                           trip_attr='u',  trip_idx='PQ_2', has_vBus=False),
    "PFlow1": _RoutineSpec(case_fixture='pjm5bus_json', stg='PV_1', solver_kwarg=None,
                           needs_pypower=True,  pre_reduce_load=True,
                           trip_attr='u',  trip_idx='PQ_2', has_vBus=False),
    "DCOPF1": _RoutineSpec(case_fixture='pjm5bus_json', stg='PV_1', solver_kwarg=None,
                           needs_pypower=True,  pre_reduce_load=True,
                           trip_attr='u',  trip_idx='PQ_2', has_vBus=False),
    "ACOPF1": _RoutineSpec(case_fixture='pjm5bus_json', stg='PV_1', solver_kwarg=None,
                           needs_pypower=True,  pre_reduce_load=True,
                           trip_attr='u',  trip_idx='PQ_2', has_vBus=False),
}

_ROUTINE_IDS = list(_ROUTINES)


@dataclass(frozen=True)
class _Ctx:
    ss: object
    routine_id: str
    spec: _RoutineSpec
    rtn: object


@pytest.fixture
def ctx(request):
    routine_id = request.param
    spec = _ROUTINES[routine_id]
    ss = request.getfixturevalue(spec.case_fixture)
    if spec.pre_reduce_load:
        ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
    return _Ctx(ss=ss, routine_id=routine_id, spec=spec, rtn=getattr(ss, routine_id))


def _skip_if_pypower_missing(spec):
    if spec.needs_pypower and not HAS_PYPOWER:
        pytest.skip("PYPOWER is not available.")


def _run(rtn, spec):
    if spec.solver_kwarg is None:
        rtn.run()
    else:
        rtn.run(solver=spec.solver_kwarg)


_PARAMETRIZE_ROUTINES = pytest.mark.parametrize(
    "ctx", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS,
)


@_PARAMETRIZE_ROUTINES
def test_init(ctx):
    # init() does not invoke the backend solver.
    ctx.rtn.init()
    assert ctx.rtn.initialized, f"{ctx.routine_id} initialization failed!"


@_PARAMETRIZE_ROUTINES
def test_trip_gen(ctx):
    _skip_if_pypower_missing(ctx.spec)
    ctx.ss.StaticGen.set(src='u', idx=ctx.spec.stg, attr='v', value=0)

    ctx.rtn.update()
    _run(ctx.rtn, ctx.spec)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge under generator trip!"
    assert abs(ctx.rtn.get(src='pg', attr='v', idx=ctx.spec.stg)) < 1e-6, \
        "Generator trip does not take effect!"


@_PARAMETRIZE_ROUTINES
def test_trip_line(ctx):
    _skip_if_pypower_missing(ctx.spec)
    ctx.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)

    ctx.rtn.update()
    _run(ctx.rtn, ctx.spec)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge under line trip!"
    assert abs(ctx.rtn.get(src='plf', attr='v', idx='Line_3')) < 1e-6, \
        "Line trip does not take effect!"


@_PARAMETRIZE_ROUTINES
def test_set_load(ctx):
    _skip_if_pypower_missing(ctx.spec)

    _run(ctx.rtn, ctx.spec)
    pgs = ctx.rtn.pg.v.sum()

    ctx.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
    ctx.rtn.update()
    _run(ctx.rtn, ctx.spec)
    pgs_pqt = ctx.rtn.pg.v.sum()
    assert pgs_pqt < pgs, "Load set does not take effect!"

    ctx.ss.PQ.set(src=ctx.spec.trip_attr, attr='v', idx=ctx.spec.trip_idx, value=0)
    ctx.rtn.update()
    _run(ctx.rtn, ctx.spec)
    pgs_pqt2 = ctx.rtn.pg.v.sum()
    assert pgs_pqt2 < pgs_pqt, "Load trip does not take effect!"


@pytest.mark.parametrize(
    "ctx", [r for r, s in _ROUTINES.items() if s.has_vBus], indirect=True,
    ids=[r for r, s in _ROUTINES.items() if s.has_vBus],
)
def test_vBus(ctx):
    _skip_if_pypower_missing(ctx.spec)
    _run(ctx.rtn, ctx.spec)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge!"
    assert np.any(ctx.rtn.vBus.v), "vBus is all zero!"
