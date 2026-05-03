"""
Tests for UC minimum on/off duration — `uc_min_on_off` project.

Layered by project phase:

- **Phase 1 (state-equation fix, landed):** ``ugd`` is free to vary
  in time; ``vgd`` and ``wgd`` are continuous in [0, 1] with a
  combined ``state`` equality and ``vwexcl`` exclusivity replacing
  the prior ``actv*``/``actw*`` equalities.

- **Phase 2 (initial-state min-up/down via ``ton0``/``toff0``,
  TODO):** placeholder pins that the params are still unused.

- **Phase 3 (interior min-up/down window via Rajan-Takriti,
  TODO):** placeholder pins that ``don``/``doff`` only enforce
  the trivial ``v ≤ u`` reduction today.

Conventions:
- ``pjm5bus_json`` from ``tests.conftest``: 5 gens, 24 periods,
  ``UC.config.t = 1`` h.
- ``StaticGen.set(...)``; ``ss.UC.update()`` after.
- ``GCost`` has its own idx; map by ``gen`` field via
  ``_gcost_idx_for``.
- Solver: SCIP, gated on ``HAS_MISOCP``.
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


# ---------- Phase 1: state-equation fix ----------

def test_uc_solves_baseline(pjm5bus_json):
    """UC solves on the demo case (sanity)."""
    _skip_if_solver_missing()
    ss = pjm5bus_json
    ss.UC.run(solver=_SOLVER)
    assert ss.UC.converged, "Baseline UC did not converge."


def test_state_equation_holds(pjm5bus_json):
    """
    For every (g, t≥1): ``u[g,t] - u[g,t-1] - v[g,t] + w[g,t] == 0``.
    For (g, t=0): ``u[g,0] - ug₀[g] - v[g,0] + w[g,0] == 0``.
    """
    _skip_if_solver_missing()
    ss = pjm5bus_json
    gidx = _gidx(ss)
    ss.UC._initial_guess()
    ug0 = ss.StaticGen.get(src='u', attr='v', idx=gidx)
    ss.UC.run(solver=_SOLVER)
    assert ss.UC.converged

    ugd = ss.UC.get(src='ugd', attr='v', idx=gidx)
    vgd = ss.UC.get(src='vgd', attr='v', idx=gidx)
    wgd = ss.UC.get(src='wgd', attr='v', idx=gidx)

    np.testing.assert_allclose(
        ugd[:, 0] - ug0 - vgd[:, 0] + wgd[:, 0],
        0.0, atol=1e-6,
        err_msg="state0 violated at t=0",
    )
    diff = ugd[:, 1:] - ugd[:, :-1] - vgd[:, 1:] + wgd[:, 1:]
    np.testing.assert_allclose(
        diff, 0.0, atol=1e-6,
        err_msg="state equation violated for t>=1",
    )


def test_vw_exclusivity(pjm5bus_json):
    """``v[g, t] + w[g, t] <= 1`` for all (g, t)."""
    _skip_if_solver_missing()
    ss = pjm5bus_json
    ss.UC.run(solver=_SOLVER)
    assert ss.UC.converged
    gidx = _gidx(ss)
    vgd = ss.UC.get(src='vgd', attr='v', idx=gidx)
    wgd = ss.UC.get(src='wgd', attr='v', idx=gidx)
    assert np.all(vgd + wgd <= 1.0 + 1e-6), \
        f"vwexcl violated: max(v+w) = {(vgd+wgd).max()}"


def test_min_up_initial_state_solves(pjm5bus_json):
    """
    Asymmetric ``ug₀`` (only one gen ON) used to be infeasible under
    stuck commitment. Phase 1 lets the others come online → solve.
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
    assert ss.UC.converged, (
        "UC must solve once Phase 1 lets initially-OFF gens come "
        "online during the horizon."
    )


def test_min_down_zero_cost_starts_unit(pjm5bus_json):
    """
    A gen with ``u₀ = 0`` and zero generation cost should turn ON
    at some period under Phase 1 (was permanently stuck OFF before).
    """
    _skip_if_solver_missing()
    ss = pjm5bus_json
    gidx = _gidx(ss)
    target = 'PV_2'

    _set_param(ss, 'td2', 0.0)
    _set_param(ss, 'toff0', 0.0)
    _set_param(ss, 'u',
               [0.0 if g == target else 1.0 for g in gidx])

    gc_target = _gcost_idx_for(ss, target)
    ss.GCost.set(src='c1', attr='v', idx=gc_target, value=0.0)
    ss.GCost.set(src='c0', attr='v', idx=gc_target, value=0.0)

    ss.UC.update()
    ss.UC.run(solver=_SOLVER)
    assert ss.UC.converged
    ugd = ss.UC.get(src='ugd', attr='v', idx=target).flatten()
    assert (ugd > 0.5).any(), (
        f"Phase-1 fix incomplete: gen with c1=0 stayed permanently "
        f"OFF (got ugd={ugd})."
    )


# ---------- Phase 2 placeholder: initial-state min-up/down ----------

def test_ton0_toff0_currently_unused(pjm5bus_json):
    """
    Pre-Phase-2 pin: setting ``ton0`` / ``toff0`` does not change
    the UC objective. Phase 2 should make this fail.
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
        f"Phase-2 has effectively landed: ton0/toff0 changed obj "
        f"({obj_before} → {obj_after}). Convert this test to assert "
        f"the initial-state min-up/down behavior."
    )
