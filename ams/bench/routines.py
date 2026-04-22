"""Per-routine benchmark harness.

Ports the 5-phase init breakdown and solve-timing pattern from the
removed ``ams/benchmarks.py`` (commit ``45cd1bc8``), adapted to the
new ``Measurement`` schema (structured ``routine`` / ``case`` /
``solver`` / ``phase`` dimensions).

Tier A targets LP-family routines with CLARABEL as the default
solver. UC / UC2 are deferred to Tier B because they're MIPs and
CLARABEL doesn't do MIP.
"""
from __future__ import annotations

import logging
import time

import ams

from ams.bench.cases import CASE_LADDER, ROUTINE_CASES, CaseSpec
from ams.bench.harness import Measurement, summarize

logger = logging.getLogger(__name__)

INIT_PHASES: tuple[str, ...] = (
    "mats", "parse", "evaluate", "finalize", "postinit",
)

TIER_A_ROUTINES: tuple[str, ...] = ("DCOPF", "RTED", "ED", "ACOPF")


def bench_case_load(case: CaseSpec, *, reps: int, warmup_reps: int) -> Measurement:
    """Time ``ams.load(case)`` over ``reps`` measured iterations."""
    from ams.bench.harness import measure

    def _load() -> None:
        ams.load(
            ams.get_case(case.path),
            setup=True, no_output=True, default_config=True,
        )

    return measure(
        _load,
        name=f"case_load/{case.name}",
        group="case_load",
        reps=reps,
        warmup_reps=warmup_reps,
        case=case.name,
    )


def bench_routine_init_phases(
    case: CaseSpec,
    routine: str,
    *,
    reps: int,
    warmup_reps: int,
) -> list[Measurement]:
    """Measure each of the 5 init phases for ``routine`` on ``case``.

    Each iteration = one fresh ``ams.load`` + one full init sequence.
    The first ``warmup_reps`` iterations are discarded; remaining reps
    populate a per-phase raw timings list. Returns one ``Measurement``
    per phase. On iteration failure, remaining iterations are skipped
    and the error is recorded on every phase returned.
    """
    per_phase_raw: dict[str, list[float]] = {p: [] for p in INIT_PHASES}
    err: str | None = None

    for i in range(warmup_reps + reps):
        try:
            sp = ams.load(
                ams.get_case(case.path),
                setup=True, no_output=True, default_config=True,
            )
            rtn = sp.routines[routine]

            phase_times = {}
            t0 = time.perf_counter()
            sp.mats.build(force=True)
            phase_times["mats"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            rtn.om.parse(force=True)
            phase_times["parse"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            rtn.om.evaluate(force=True)
            phase_times["evaluate"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            rtn.om.finalize(force=True)
            phase_times["finalize"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            rtn.init()
            phase_times["postinit"] = time.perf_counter() - t0

            if i >= warmup_reps:
                for phase in INIT_PHASES:
                    per_phase_raw[phase].append(phase_times[phase])
        except Exception as exc:
            logger.warning(
                "bench init-phase %s/%s iter %d failed: %s",
                routine, case.name, i, exc,
            )
            err = str(exc)
            break

    return [
        summarize(
            per_phase_raw[phase],
            name=f"{routine}/{case.name}/{phase}",
            group="routine_init_phase",
            reps=reps,
            warmup_reps=warmup_reps,
            error=err,
            routine=routine,
            case=case.name,
            phase=phase,
        )
        for phase in INIT_PHASES
    ]


def bench_routine_solve(
    case: CaseSpec,
    routine: str,
    solver: str,
    *,
    reps: int,
    warmup_reps: int,
) -> Measurement:
    """Time ``rtn.run(solver=...)`` wall-clock.

    Loads + inits once outside the measured window; each rep calls
    ``rtn.run(solver=...)``. Setup cost is excluded from the timing
    so the measurement reflects solver + model evaluation, not I/O.
    """
    try:
        sp = ams.load(
            ams.get_case(case.path),
            setup=True, no_output=True, default_config=True,
        )
        rtn = sp.routines[routine]
    except Exception as exc:
        logger.warning(
            "bench solve %s/%s/%s: load failed: %s",
            routine, case.name, solver, exc,
        )
        return summarize(
            [],
            name=f"{routine}/{case.name}/{solver}",
            group="routine_solve",
            reps=reps,
            warmup_reps=warmup_reps,
            error=str(exc),
            routine=routine,
            case=case.name,
            solver=solver,
        )

    raw: list[float] = []
    err: str | None = None
    for i in range(warmup_reps + reps):
        try:
            t0 = time.perf_counter()
            rtn.run(solver=solver)
            dt = time.perf_counter() - t0
            if i >= warmup_reps:
                raw.append(dt)
        except Exception as exc:
            logger.warning(
                "bench solve %s/%s/%s iter %d failed: %s",
                routine, case.name, solver, i, exc,
            )
            err = str(exc)
            break

    return summarize(
        raw,
        name=f"{routine}/{case.name}/{solver}",
        group="routine_solve",
        reps=reps,
        warmup_reps=warmup_reps,
        error=err,
        routine=routine,
        case=case.name,
        solver=solver,
    )


def run_smoke(
    *,
    reps: int = 1,
    warmup_reps: int = 0,
    solver: str = "CLARABEL",
) -> list[Measurement]:
    """Minimal suite (~2 measurements) for CI smoke-testing.

    Not a real perf measurement — just validates the end-to-end CLI +
    JSON serialization path. Prefer ``run_tier_a`` for actual baselines.
    """
    return [
        bench_case_load(CASE_LADDER[0], reps=reps, warmup_reps=warmup_reps),
        bench_routine_solve(
            ROUTINE_CASES[0], "DCOPF", solver,
            reps=reps, warmup_reps=warmup_reps,
        ),
    ]


def run_tier_a(
    *,
    reps: int = 5,
    warmup_reps: int = 1,
    solver: str = "CLARABEL",
) -> list[Measurement]:
    """Run the full Tier A suite and return all Measurements.

    Emits ~40 measurements in three groups:

    - ``case_load`` for every entry in ``CASE_LADDER``.
    - ``routine_init_phase`` = 5 phases × 4 routines × 2 routine
      cases = 40 (but collected as per-phase Measurements, so each
      Measurement is one row).
    - ``routine_solve`` = 4 routines × 2 routine cases = 8.
    """
    results: list[Measurement] = []

    for case in CASE_LADDER:
        results.append(bench_case_load(case, reps=reps, warmup_reps=warmup_reps))

    for case in ROUTINE_CASES:
        for routine in TIER_A_ROUTINES:
            results.extend(bench_routine_init_phases(
                case, routine, reps=reps, warmup_reps=warmup_reps,
            ))

    for case in ROUTINE_CASES:
        for routine in TIER_A_ROUTINES:
            results.append(bench_routine_solve(
                case, routine, solver, reps=reps, warmup_reps=warmup_reps,
            ))

    return results
