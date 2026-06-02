"""
Multiperiod MILP routine scenarios — parametrized.

Family E (this module): UC, UCDG, UCES, UC2, UC2DG, UC2ES on
``pjm5bus_demo``. All routines are MISOCP and use SCIP. The
fixture invokes ``_initial_guess()`` once so that ``test_trip_gen``
can verify off-generators stay at zero output.

Per-routine differences live in `_ROUTINES`:

- ``has_aBus``: 2nd-generation routines (UC2*) expose ``aBus`` and
  the vBus scenario asserts both.
- ``align_ref``: 2nd-generation routines compare against their 1st-
  generation counterpart (UC2 → UC, UC2DG → UCDG, UC2ES → UCES).
- ``obj_only_align``: ``True`` for UC2ES — energy storage makes the
  optimal commitment degenerate (equal-cost alternative ugd/pg
  schedules), so only the objective is compared; the others compare the
  full obj/ugd/pg/aBus/plf set.
- ``align_ref_first``: ``True`` for UC2 — the legacy
  ``test_align_uc`` runs ``UC`` before ``UC2``, while
  ``UC2DG``/``UC2ES`` run the 2nd-gen first. Preserved verbatim.

``test_init``, ``test_initial_guess``, and ``test_pb_formula`` do
not invoke the SCIP backend and are not skipped on solver
availability.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from tests.conftest import HAS_MISOCP


@dataclass(frozen=True)
class _RoutineSpec:
    has_aBus: bool
    align_ref: object  # str or None
    align_ref_first: bool = False
    obj_only_align: bool = False


_ROUTINES = {
    "UC":     _RoutineSpec(has_aBus=False, align_ref=None),
    "UCDG":   _RoutineSpec(has_aBus=False, align_ref=None),
    "UCES":   _RoutineSpec(has_aBus=False, align_ref=None),
    "UC2":    _RoutineSpec(has_aBus=True,  align_ref='UC', align_ref_first=True),
    "UC2DG":  _RoutineSpec(has_aBus=True,  align_ref='UCDG'),
    # UC2ES: the energy-storage flexibility makes the optimal commitment
    # schedule degenerate (multiple equal-cost ugd/pg solutions — e.g. a
    # unit's ON window can shift a period at no cost). The *objective*
    # is the stable cross-formulation invariant; exact ugd/pg/aBus/plf
    # are solver-version dependent, so compare obj only.
    "UC2ES":  _RoutineSpec(has_aBus=True,  align_ref='UCES', obj_only_align=True),
}

_ROUTINE_IDS = list(_ROUTINES)

_SOLVER = 'SCIP'


@dataclass(frozen=True)
class _Ctx:
    ss: object
    routine_id: str
    spec: _RoutineSpec
    rtn: object
    off_gen: object


@pytest.fixture
def ctx(pjm5bus_json, request):
    routine_id = request.param
    spec = _ROUTINES[routine_id]
    ss = pjm5bus_json
    ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
    rtn = getattr(ss, routine_id)
    # _initial_guess() is idempotent and solver-free; legacy setUp
    # called it for every test method including test_init, so do
    # the same here.
    off_gen = rtn._initial_guess()
    return _Ctx(ss=ss, routine_id=routine_id, spec=spec, rtn=rtn, off_gen=off_gen)


def _skip_if_solver_missing():
    if not HAS_MISOCP:
        pytest.skip("No MISOCP solver is available.")


_PARAMETRIZE_ROUTINES = pytest.mark.parametrize(
    "ctx", _ROUTINE_IDS, indirect=True, ids=_ROUTINE_IDS,
)


@_PARAMETRIZE_ROUTINES
def test_initial_guess(ctx):
    u_off_gen = ctx.ss.StaticGen.get(src='u', idx=ctx.off_gen)
    np.testing.assert_equal(
        u_off_gen, np.zeros_like(u_off_gen),
        err_msg=f"{ctx.routine_id}._initial_guess() failed!",
    )


@_PARAMETRIZE_ROUTINES
def test_init(ctx):
    ctx.rtn.init()
    assert ctx.rtn.initialized, f"{ctx.routine_id} initialization failed!"


@_PARAMETRIZE_ROUTINES
def test_trip_gen(ctx):
    """An uncommitted generator (``ugd == 0``) must produce no power.

    Unlike the LP/PF families, UC *optimizes* commitment via ``ugd``;
    ``StaticGen.u`` only seeds the pre-horizon state ``ug0`` and there
    is no per-slot status input. The ``_initial_guess()`` warm-start is
    a heuristic the solver is free to override, so a guessed-off unit
    may legitimately be re-committed (e.g. UCES/UC2ES re-commit ``PV_1``
    once ES makes it economical). The portable invariant is therefore
    the commitment-dispatch coupling: wherever the *solved* commitment
    is off, output is zero.
    """
    _skip_if_solver_missing()
    ctx.rtn.run(solver=_SOLVER)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge!"
    gen_idx = ctx.ss.StaticGen.get_all_idxes()
    ugd = np.atleast_2d(np.asarray(ctx.rtn.get(src='ugd', attr='v', idx=gen_idx)))
    pg = np.atleast_2d(np.asarray(ctx.rtn.get(src='pg', attr='v', idx=gen_idx)))
    off = ugd < 0.5
    np.testing.assert_almost_equal(
        pg[off], np.zeros_like(pg[off]), decimal=6,
        err_msg=f"{ctx.routine_id}: uncommitted generators produce power!",
    )


@_PARAMETRIZE_ROUTINES
def test_set_load(ctx):
    _skip_if_solver_missing()
    ctx.rtn.run(solver=_SOLVER)
    pgs = ctx.rtn.pg.v.sum()

    ctx.ss.PQ.set(src='p0', attr='v', idx='PQ_1', value=0.1)
    ctx.rtn.update()
    ctx.rtn.run(solver=_SOLVER)
    pgs_pqt = ctx.rtn.pg.v.sum()
    assert pgs_pqt < pgs, "Load set does not take effect!"

    ctx.ss.PQ.alter(src='u', idx='PQ_2', value=0)
    ctx.rtn.update()
    ctx.rtn.run(solver=_SOLVER)
    pgs_pqt2 = ctx.rtn.pg.v.sum()
    assert pgs_pqt2 < pgs_pqt, "Load trip does not take effect!"


@_PARAMETRIZE_ROUTINES
def test_trip_line(ctx):
    _skip_if_solver_missing()
    ctx.ss.Line.set(src='u', attr='v', idx='Line_3', value=0)
    ctx.rtn.update()

    ctx.rtn.run(solver=_SOLVER)
    assert ctx.rtn.converged, f"{ctx.routine_id} did not converge under line trip!"
    plf_l3 = ctx.rtn.get(src='plf', attr='v', idx='Line_3')
    np.testing.assert_almost_equal(np.zeros_like(plf_l3), plf_l3, decimal=6)


@_PARAMETRIZE_ROUTINES
def test_vBus(ctx):
    _skip_if_solver_missing()
    ctx.rtn.run(solver=_SOLVER)
    assert np.any(ctx.rtn.vBus.v), "vBus is all zero!"
    if ctx.spec.has_aBus:
        assert np.any(ctx.rtn.aBus.v), "aBus is all zero!"


_ALIGN_IDS = [r for r, s in _ROUTINES.items() if s.align_ref is not None]


@pytest.mark.parametrize("ctx", _ALIGN_IDS, indirect=True, ids=_ALIGN_IDS)
def test_align(ctx):
    """2nd-gen routines must match their 1st-gen counterpart."""
    _skip_if_solver_missing()
    ref = getattr(ctx.ss, ctx.spec.align_ref)
    if ctx.spec.align_ref_first:
        ref.run(solver=_SOLVER)
        ctx.rtn.run(solver=_SOLVER)
    else:
        ctx.rtn.run(solver=_SOLVER)
        ref.run(solver=_SOLVER)

    pg_idx = ctx.ss.StaticGen.get_all_idxes()
    bus_idx = ctx.ss.Bus.idx.v
    line_idx = ctx.ss.Line.idx.v
    decimals = 4

    np.testing.assert_almost_equal(
        ctx.rtn.obj.v, ref.obj.v, decimal=decimals,
        err_msg=f"Objective value between {ctx.routine_id} and {ctx.spec.align_ref} not match!",
    )
    if ctx.spec.obj_only_align:
        # Degenerate dispatch (see _ROUTINES note): the objective is the
        # only stable cross-formulation invariant. ugd/pg/aBus/plf can
        # legitimately differ between equal-cost optima.
        return
    np.testing.assert_almost_equal(
        ctx.rtn.get(src='ugd', attr='v', idx=pg_idx),
        ref.get(src='ugd', attr='v', idx=pg_idx),
        decimal=decimals,
        err_msg=f"ugd between {ctx.routine_id} and {ctx.spec.align_ref} not match!",
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


def test_congested_line_limit_uc2(pjm5bus_json):
    """UC2 must enforce line flow limits under network congestion.

    Tighten Line_2's rate_a below its unconstrained peak PTDF flow so the
    limit actually binds during optimization.  Before the fix, the PTDF
    routine constrained Bf@aBus (a free variable) instead of the PTDF flow,
    making the limit vacuous.
    """
    if not HAS_MISOCP:
        pytest.skip("No MISOCP solver is available.")

    ss = pjm5bus_json
    ss.PQ.set(src='p0', attr='v', idx=['PQ_1', 'PQ_2'], value=[0.3, 0.3])
    tight_limit = 0.65   # below peak unconstrained flow (~0.789)
    ss.Line.set(src='rate_a', attr='v', idx='Line_2', value=tight_limit)

    ss.UC.run(solver=_SOLVER)
    assert ss.UC.converged, "UC did not converge under congestion!"
    ss.UC2.update()
    ss.UC2.run(solver=_SOLVER)
    assert ss.UC2.converged, "UC2 did not converge under congestion!"

    # UC2 line flows must respect the tightened limit on Line_2
    plf_l2 = ss.UC2.get(src='plf', attr='v', idx='Line_2')
    assert np.all(np.abs(plf_l2) <= tight_limit + 1e-4), (
        f"UC2 Line_2 flow {np.max(np.abs(plf_l2)):.4f} exceeds limit {tight_limit}"
    )

    # Objectives must align between UC2 and UC
    np.testing.assert_almost_equal(
        ss.UC2.obj.v, ss.UC.obj.v, decimal=3,
        err_msg="UC2 objective does not match UC under congestion!",
    )
