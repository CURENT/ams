"""
Phase 0 — characterization tests for UC minimum on/off duration.

These tests pin **current** behavior so later phases have an
unambiguous baseline.

Headline Phase 0 finding (2026-05-03): the existing
``actv`` / ``actv0`` / ``actw`` / ``actw0`` equalities, combined
with ``boolean=True`` on ``vgd`` / ``wgd``, force ``ugd[g, t]`` to
equal ``ug₀[g]`` for every period — i.e. UC commitment cannot vary
in time. Until that is fixed (Phase 1 of the project), min-up/down
constraints (``td1``, ``td2``, ``ton0``, ``toff0``) cannot be
exercised at all.

Conventions:
- Use ``pjm5bus_json`` from ``tests.conftest``: 5 gens, 24 periods,
  ``UC.config.t = 1`` h.
- ``StaticGen.set(src=..., attr='v', idx=..., value=...)`` to mutate
  parameters; call ``ss.UC.update()`` after to refresh services.
- ``GCost`` has its own idx; map by the ``gen`` field via
  ``_gcost_idx_for``.
- Solver: SCIP, gated on ``HAS_MISOCP``.

These tests are *flip targets*: as Phase 1/2/3 land, the assertion
on each test should invert (or the test renames) to reflect the new
correct behavior.
"""

import numpy as np
import pytest

from tests.conftest import HAS_MISOCP


_SOLVER = 'SCIP'


def _skip_if_solver_missing():
    if not HAS_MISOCP:
        pytest.skip("No MISOCP solver is available.")


def _gidx(ss):
    return ss.StaticGen.get_all_idxes()


def _set_param(ss, src, value):
    ss.StaticGen.set(src=src, attr='v', idx=_gidx(ss), value=value)


def _gcost_idx_for(ss, gen_idx):
    """Return the ``GCost`` idx whose ``gen`` field equals ``gen_idx``."""
    all_gc = ss.GCost.get_all_idxes()
    gen_field = ss.GCost.get(src='gen', attr='v', idx=all_gc)
    for gc, g in zip(all_gc, gen_field):
        if g == gen_idx:
            return gc
    raise AssertionError(f"No GCost row maps to gen {gen_idx!r}")


def test_uc_solves_baseline(pjm5bus_json):
    """UC solves on the demo case as shipped (sanity)."""
    _skip_if_solver_missing()
    ss = pjm5bus_json
    ss.UC.run(solver=_SOLVER)
    assert ss.UC.converged, "Baseline UC did not converge."


def test_baseline_ugd_is_constant_in_time(pjm5bus_json):
    """
    PHASE 0 PIN: every generator's ``ugd[g, :]`` row equals
    ``ug₀[g]`` for all 24 periods. Direct consequence of the
    ``actv`` / ``actw`` equality coupling under ``boolean=True``.

    Phase 1 must overturn this — at least one generator under a
    load-changing scenario should be allowed to switch.
    """
    _skip_if_solver_missing()
    ss = pjm5bus_json
    gidx = _gidx(ss)
    ss.UC._initial_guess()
    ug0 = ss.StaticGen.get(src='u', attr='v', idx=gidx)

    ss.UC.run(solver=_SOLVER)
    assert ss.UC.converged, "Baseline UC did not converge."

    ugd = ss.UC.get(src='ugd', attr='v', idx=gidx)
    for i, g in enumerate(gidx):
        row = ugd[i]
        assert np.allclose(row, ug0[i]), (
            f"Phase-0 pin violated for {g}: row={row} but ug0={ug0[i]}"
        )


def test_min_up_initial_state_blocks_solve(pjm5bus_json):
    """
    Asymmetric ``ug₀`` — only one gen ON — under the constant-
    commitment regime forces every other gen permanently OFF, which
    is below capacity for the demo demand. Solver returns infeasible.

    Phase 1 will allow the others to come ON across the horizon →
    flip to expecting feasibility (but Phase 2 will then enforce the
    actual ``td1`` / ``ton0`` block).
    """
    _skip_if_solver_missing()
    ss = pjm5bus_json
    gidx = _gidx(ss)
    target = 'PV_2'

    _set_param(ss, 'td1', 4.0)
    _set_param(ss, 'ton0',
               [1.0 if g == target else 0.0 for g in gidx])
    _set_param(ss, 'u',
               [1.0 if g == target else 0.0 for g in gidx])

    ss.UC.update()
    ss.UC.run(solver=_SOLVER)

    assert not ss.UC.converged, (
        "Phase-0 pin violated: with only one gen ON at ug₀ and "
        "constant-commitment, UC should be infeasible. If this now "
        "converges, Phase 1 has effectively landed — invert this test."
    )


def test_min_down_initial_state_blocks_startup(pjm5bus_json):
    """
    A gen with ``u₀ = 0`` cannot turn ON at any period under the
    constant-commitment regime, regardless of cost pressure.

    Phase 1 fix → flip: with ``c1 = 0`` the solver brings the unit
    ON. Phase 2 fix → unit cannot turn ON for ``L_dn`` periods
    matching ``td2 / toff0``.
    """
    _skip_if_solver_missing()
    ss = pjm5bus_json
    gidx = _gidx(ss)
    target = 'PV_2'

    # Reset min-down so only the constant-commitment property is
    # being characterized here.
    _set_param(ss, 'td2', 0.0)
    _set_param(ss, 'toff0', 0.0)
    _set_param(ss, 'u',
               [0.0 if g == target else 1.0 for g in gidx])

    gc_target = _gcost_idx_for(ss, target)
    ss.GCost.set(src='c1', attr='v', idx=gc_target, value=0.0)
    ss.GCost.set(src='c0', attr='v', idx=gc_target, value=0.0)

    ss.UC.update()
    ss.UC.run(solver=_SOLVER)
    assert ss.UC.converged, "UC did not converge in min-down init scenario."

    ugd = ss.UC.get(src='ugd', attr='v', idx=target).flatten()
    assert np.allclose(ugd, 0), (
        f"Phase-0 pin violated: gen with ug₀=0 stayed OFF for the "
        f"whole horizon (constant-commitment); got ugd={ugd}. If the "
        f"solver now starts the unit, Phase 1 has effectively landed "
        f"— invert this test."
    )


def test_ton0_toff0_currently_unused(pjm5bus_json):
    """
    Sanity: setting ``ton0`` / ``toff0`` does not change the UC
    objective at all. They are declared on ``StaticGen`` but no
    routine reads them. Phase 2 should make this test fail by
    introducing initial-state min-up/down constraints.
    """
    _skip_if_solver_missing()
    ss = pjm5bus_json

    ss.UC.run(solver=_SOLVER)
    obj_before = float(ss.UC.obj.v)

    _set_param(ss, 'ton0', 5.0)
    _set_param(ss, 'toff0', 5.0)
    ss.UC.update()
    ss.UC.run(solver=_SOLVER)
    obj_after = float(ss.UC.obj.v)

    assert np.isclose(obj_before, obj_after), (
        f"Phase-0 pin violated: setting ton0/toff0 changed objective "
        f"({obj_before} → {obj_after}). Either Phase 2 has landed or "
        f"another consumer of these params has appeared."
    )
