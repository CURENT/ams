"""
Multiperiod LP routine scenarios — parametrized.

Family D (this module): ED, EDDG, EDES, ED2, ED2DG, ED2ES on
``pjm5bus_demo``. Test bodies use vector assertions because pg/plf
results are 2-D over (gen, slot). Per-routine differences live in
`_ROUTINES`:

- ``solver``: ``'CLARABEL'`` for LP routines; ``'SCIP'`` for MISOCP
  storage variants (EDES, ED2ES).
- ``has_aBus``: 2nd-generation routines (ED2*) expose ``aBus`` and
  the vBus scenario asserts both.
- ``align_ref``: 2nd-generation routines compare against their 1st-
  generation counterpart (ED2 → ED, ED2DG → EDDG, ED2ES → EDES).
  ``None`` for 1st-generation routines.
- ``align_decimals``: precision for the alignment comparisons.
  Tighter (4) for non-storage; looser (3) for ED2ES.
- ``loc_offtime``: tuple of EDTSlot rows used by ``test_trip_gen``
  to force the gen off. ``(0, 2, 4)`` for everything except EDES
  (``(0, 2)``) — preserved verbatim from the legacy bodies.

``test_init`` does not invoke a solver and is never skipped on
solver availability; ``test_pb_formula`` is a static-string check
on ``pb.e_str`` that also never runs the solver.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from tests.conftest import HAS_MISOCP


@dataclass(frozen=True)
class _RoutineSpec:
    solver: str
    has_aBus: bool
    align_ref: object  # str or None
    align_decimals: int = 4
    loc_offtime: tuple = (0, 2, 4)


_ROUTINES = {
    "ED":     _RoutineSpec(solver='CLARABEL', has_aBus=False, align_ref=None),
    "EDDG":   _RoutineSpec(solver='CLARABEL', has_aBus=False, align_ref=None),
    "EDES":   _RoutineSpec(solver='SCIP',     has_aBus=False, align_ref=None,
                           loc_offtime=(0, 2)),
    "ED2":    _RoutineSpec(solver='CLARABEL', has_aBus=True,  align_ref='ED'),
    "ED2DG":  _RoutineSpec(solver='CLARABEL', has_aBus=True,  align_ref='EDDG'),
    "ED2ES":  _RoutineSpec(solver='SCIP',     has_aBus=True,  align_ref='EDES',
                           align_decimals=3),
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
    routine_id = request.param
    spec = _ROUTINES[routine_id]
    ss = pjm5bus_json
    ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
    return _Ctx(ss=ss, routine_id=routine_id, spec=spec, rtn=getattr(ss, routine_id))


def _skip_if_solver_missing(spec):
    if spec.solver == 'SCIP' and not HAS_MISOCP:
        pytest.skip("No MISOCP solver is available.")


_PARAMETRIZE_ROUTINES = pytest.mark.parametrize(
    "ctx", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS,
)


@_PARAMETRIZE_ROUTINES
def test_init(ctx):
    ctx.rtn.init()
    assert ctx.rtn.initialized, f"{ctx.routine_id} initialization failed!"


@_PARAMETRIZE_ROUTINES
def test_trip_gen(ctx):
    """Verify EDTSlot.ug takes effect, and StaticGen.u is ignored."""
    _skip_if_solver_missing(ctx.spec)

    # a) EDTSlot.ug.v drives per-slot generator status
    stg = 'PV_1'
    stg_uid = ctx.ss.StaticGen.get_all_idxes().index(stg)
    loc_offtime = np.array(ctx.spec.loc_offtime)
    ctx.ss.EDTSlot.ug.v[loc_offtime, stg_uid] = 0

    ctx.rtn.run(solver=ctx.spec.solver)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge under generator trip!"
    pg_pv1 = ctx.rtn.get(src='pg', attr='v', idx=stg)
    np.testing.assert_almost_equal(
        np.zeros_like(loc_offtime), pg_pv1[loc_offtime],
        decimal=6, err_msg="Generator trip does not take effect!",
    )

    ctx.ss.EDTSlot.ug.v[...] = 1  # reset for sub-test b

    # b) StaticGen.u must NOT take effect — multiperiod ED uses EDTSlot.ug
    ctx.ss.StaticGen.set(src='u', idx=stg, attr='v', value=0)
    ctx.rtn.update()

    ctx.rtn.run(solver=ctx.spec.solver)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge under generator trip!"
    pg_pv1 = ctx.rtn.get(src='pg', attr='v', idx=stg)
    np.testing.assert_array_less(
        np.zeros_like(pg_pv1), pg_pv1,
        err_msg="Generator trip via StaticGen.u took effect, which is unexpected!",
    )


@_PARAMETRIZE_ROUTINES
def test_trip_line(ctx):
    _skip_if_solver_missing(ctx.spec)
    ctx.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
    ctx.rtn.update()

    ctx.rtn.run(solver=ctx.spec.solver)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge under line trip!"
    plf_l3 = ctx.rtn.get(src='plf', attr='v', idx='Line_3')
    np.testing.assert_almost_equal(np.zeros_like(plf_l3), plf_l3, decimal=6)


@_PARAMETRIZE_ROUTINES
def test_set_load(ctx):
    _skip_if_solver_missing(ctx.spec)

    ctx.rtn.run(solver=ctx.spec.solver)
    pgs = ctx.rtn.pg.v.sum()

    ctx.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
    ctx.rtn.update()
    ctx.rtn.run(solver=ctx.spec.solver)
    pgs_pqt = ctx.rtn.pg.v.sum()
    assert pgs_pqt < pgs, "Load set does not take effect!"

    ctx.ss.PQ.alter(src='u', idx='PQ_2', value=0)
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


_ALIGN_IDS = [r for r, s in _ROUTINES.items() if s.align_ref is not None]


@pytest.mark.parametrize("ctx", _ALIGN_IDS, indirect=True, ids=_ALIGN_IDS)
def test_align(ctx):
    """2nd-gen routines must match their 1st-gen counterpart."""
    _skip_if_solver_missing(ctx.spec)
    ref = getattr(ctx.ss, ctx.spec.align_ref)
    ctx.rtn.run(solver=ctx.spec.solver)
    ref.run(solver=ctx.spec.solver)

    pg_idx = ctx.ss.StaticGen.get_all_idxes()
    bus_idx = ctx.ss.Bus.idx.v
    line_idx = ctx.ss.Line.idx.v
    decimals = ctx.spec.align_decimals

    np.testing.assert_almost_equal(
        ctx.rtn.obj.v, ref.obj.v, decimal=decimals,
        err_msg=f"Objective value between {ctx.routine_id} and {ctx.spec.align_ref} not match!",
    )
    np.testing.assert_almost_equal(
        ctx.rtn.get(src='pg', attr='v', idx=pg_idx),
        ref.get(src='pg', attr='v', idx=pg_idx),
        decimal=decimals,
        err_msg=f"Generator power between {ctx.routine_id} and {ctx.spec.align_ref} not match!",
    )
    np.testing.assert_almost_equal(
        ctx.rtn.get(src='aBus', attr='v', idx=bus_idx),
        ref.get(src='aBus', attr='v', idx=bus_idx),
        decimal=decimals,
        err_msg=f"Bus angle between {ctx.routine_id} and {ctx.spec.align_ref} not match!",
    )
    np.testing.assert_almost_equal(
        ctx.rtn.get(src='plf', attr='v', idx=line_idx),
        ref.get(src='plf', attr='v', idx=line_idx),
        decimal=decimals,
        err_msg=f"Line flow between {ctx.routine_id} and {ctx.spec.align_ref} not match!",
    )


@pytest.mark.parametrize("ctx", _ALIGN_IDS, indirect=True, ids=_ALIGN_IDS)
def test_pb_formula(ctx):
    """2nd-gen routines must use the non-angle pb formulation."""
    assert 'aBus' not in ctx.rtn.pb.e_str, f"Bus angle is used in {ctx.routine_id}.pb!"
